"""Play Skyfall with a hosted LLM making every decision (the thing Jev/Kev are built to replace).

The key is read from the environment (GEMINI_API_KEY), or from a .env file in the repo
(gitignored). It is never printed and never leaves this process except to the provider.

    export GEMINI_API_KEY=...        # or put GEMINI_API_KEY=... in .env
    python3 -m model.bench --players oracle,kev:http://127.0.0.1:8009,gemini --episodes 20
"""
import json
import os
import re
import time
import urllib.request

from data.questions import QUESTIONS

PROMPT = """You control the skydiver Y in this text grid. Everything scrolls UP one row per tick.
{state}

Question move: {move_q}
Question open: {open_q}

Reply with ONLY a JSON object: {{"move": "left" | "right" | "stay", "open": true | false}}"""

# USD per million tokens. Override with --price_in/--price_out if your plan differs.
PRICES = {                            # list prices found 2026-09-30; check ai.google.dev/pricing before quoting
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3.7-flash": (0.75, 3.75),
    "jev": (0.042, 0.0),              # TypeSafe's published Jev price: input only, output free
}


def load_dotenv(path=".env"):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def parse_answer(text):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    mv, op = obj.get("move"), obj.get("open")
    if mv not in ("left", "right", "stay") or not isinstance(op, bool):
        return None
    return {"move": mv, "open": op}


class GeminiPolicy:
    """One generateContent call per tick, JSON-constrained output, thinking off (fastest honest setting)."""

    def __init__(self, model="gemini-2.5-flash", thinking_budget=0, timeout=60):
        load_dotenv()
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise SystemExit("GEMINI_API_KEY is not set (export it, or put it in .env)")
        self.model, self.timeout = model, timeout
        self.gen_cfg = {"responseMimeType": "application/json",
                        "responseSchema": {"type": "OBJECT", "properties": {"move": {"type": "STRING", "enum": ["left", "right", "stay"]},
                                                                             "open": {"type": "BOOLEAN"}}, "required": ["move", "open"]},
                        "temperature": 0}
        if thinking_budget is not None:
            self.gen_cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        self.name = f"gemini:{model}"

    def __call__(self, w):
        return self.decide(w.render())["action"]

    def decide(self, state):
        body = {"contents": [{"role": "user", "parts": [{"text": PROMPT.format(
            state=state, move_q=QUESTIONS["move"]["instructions"], open_q=QUESTIONS["open"]["instructions"])}]}],
                "generationConfig": self.gen_cfg}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp = json.load(r)
            err = None
        except urllib.error.HTTPError as e:
            resp, err = {}, f"HTTP {e.code}: {e.read()[:200]!r}"
        ms = (time.perf_counter() - t0) * 1000
        text = ""
        try:
            text = resp["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            pass
        usage = resp.get("usageMetadata", {})
        action = parse_answer(text) or {"move": "stay", "open": False}     # unparseable -> do nothing
        return {"action": action, "parsed": parse_answer(text) is not None, "latency_ms": ms, "error": err,
                "input_tokens": usage.get("promptTokenCount", 0), "output_tokens": usage.get("candidatesTokenCount", 0),
                "thinking_tokens": usage.get("thoughtsTokenCount", 0), "raw": text[:200]}
