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

PROMPT_MINIMAL = """You control the skydiver Y in this text grid. Everything scrolls UP one row per tick.
{state}

Question move: {move_q}
Question open: {open_q}

Reply with ONLY a JSON object: {{"move": "left" | "right" | "stay", "open": true | false}}"""

RULES = """You are playing a tiny skydiving game on a grid with 5 columns (c0..c4, left to right) and 6 rows (row0 at the top, row5 at the bottom).
Rules:
- You are Y and you are always on row1.
- Every tick, everything except you scrolls UP by one row: whatever is on row2 now will be on row1 (your row) next tick.
- Letters B (bird), P (other skydiver), A (airplane) and any other letter are deadly if they reach your cell. C is a cloud and is harmless: you fly straight through it.
- Each tick you choose move = left, right or stay (one column at most; the grid edges clamp you). Your move happens BEFORE the scroll.
  You survive the tick only if, after your move, the cell on row1 and the cell on row2 in your column are both '.' or 'C'.
- Open the parachute (open = true) only when altitude <= 1000 m AND the cell directly below you (row2, your column) is '.' or 'C'.
  Opening above 1000 m loses. Opening with something deadly directly below loses. Reaching 0 m without opening loses.
  Altitude drops 100 m per tick."""

PROCEDURE = """Decision procedure, follow it exactly:
1. Find the column c of Y on row1.
2. Look at row2, column c (directly below you). If it is '.' or 'C': move = stay.
   Otherwise move sideways to a neighbouring column c' (c-1 or c+1, inside c0..c4) where BOTH row1[c'] and row2[c'] are '.' or 'C'.
   If both neighbours are safe, prefer the one closer to the centre column c2. If neither neighbour is safe: stay.
3. open = true only if altitude <= 1000 AND row2[c] is '.' or 'C'. Otherwise open = false."""

PROMPT_RULES = RULES + """

Current state:
{labeled}

Reply with ONLY a JSON object: {{"move": "left" | "right" | "stay", "open": true | false}}"""

PROMPT_PROCEDURE = RULES + "\n\n" + PROCEDURE + """

Current state:
{labeled}

Reply with ONLY a JSON object: {{"move": "left" | "right" | "stay", "open": true | false}}"""

PROMPT_COT = RULES + "\n\n" + PROCEDURE + """

Current state:
{labeled}

Reply with ONLY a JSON object with three fields, in this order:
"analysis": one or two sentences applying the procedure (Y's column, what is directly below it, what the neighbouring columns hold on row1 and row2, the altitude check),
"move": "left" | "right" | "stay",
"open": true | false"""

PROMPTS = {"minimal": PROMPT_MINIMAL, "rules": PROMPT_RULES, "procedure": PROMPT_PROCEDURE, "cot": PROMPT_COT}


def labeled_grid(state):
    """The same state with row/column labels, for the LLM only (Kev sees the raw grid)."""
    lines = state.split("\n")
    out = [lines[0] + " m", "       c0 c1 c2 c3 c4"]
    notes = {1: "   <- your row (Y)", 2: "   <- directly below you"}
    for i, row in enumerate(lines[1:]):
        out.append(f"row{i}:  " + "  ".join(row.split(" ")) + notes.get(i, ""))
    return "\n".join(out)


def build_prompt(name, state):
    return PROMPTS[name].format(state=state, labeled=labeled_grid(state),
                                move_q=QUESTIONS["move"]["instructions"], open_q=QUESTIONS["open"]["instructions"])


# USD per million tokens. Override with --price_in/--price_out if your plan differs.
PRICES = {                            # list prices found 2026-09-30; check ai.google.dev/pricing before quoting
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3-flash": (0.50, 3.00),
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.8-flash": (0.75, 3.75),   # output price includes thinking tokens
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-flash-lite-latest": (0.25, 1.50),   # alias; the response's modelVersion says what actually served it
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

    def __init__(self, model="gemini-3.8-flash", thinking=None, timeout=60, prompt="procedure"):
        """thinking: the thinkingConfig dict sent to the API, e.g. {"thinkingBudget": 0} (Gemini 2.x) or
        {"thinkingLevel": "minimal"} (Gemini 3.x); None sends no thinkingConfig at all.
        prompt: one of PROMPTS ("minimal" = two bare questions, "rules" = full rules + labeled grid,
        "procedure" = rules + the decision procedure spelled out)."""
        load_dotenv()
        self.prompt = prompt
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise SystemExit("GEMINI_API_KEY is not set (export it, or put it in .env)")
        self.model, self.timeout = model, timeout
        props = {"move": {"type": "STRING", "enum": ["left", "right", "stay"]}, "open": {"type": "BOOLEAN"}}
        order = ["move", "open"]
        if prompt == "cot":                      # a visible reasoning field, written before the answer
            props = {"analysis": {"type": "STRING"}, **props}
            order = ["analysis", "move", "open"]
        self.gen_cfg = {"responseMimeType": "application/json",
                        "responseSchema": {"type": "OBJECT", "properties": props, "required": order, "propertyOrdering": order},
                        "temperature": 0}
        if thinking:
            self.gen_cfg["thinkingConfig"] = thinking
        self.thinking = thinking
        self.name = f"gemini:{model}" + (f":{'/'.join(str(v) for v in thinking.values())}" if thinking else "") + f":{prompt}"

    def __call__(self, w):
        return self.decide(w.render())["action"]

    def decide(self, state, retries=6):
        body = {"contents": [{"role": "user", "parts": [{"text": build_prompt(self.prompt, state)}]}],
                "generationConfig": self.gen_cfg}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        t_wall = time.perf_counter()
        attempt, err, resp, ms = 0, None, {}, 0.0
        while True:
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "x-goog-api-key": self.key})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.load(r)
                err = None
            except urllib.error.HTTPError as e:
                resp, err = {}, f"HTTP {e.code}: {e.read()[:160]!r}"
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                resp, err = {}, f"{type(e).__name__}: {e}"
            ms = (time.perf_counter() - t0) * 1000
            if err is None or attempt >= retries or not any(c in err for c in ("503", "429", "500", "Timeout", "timed out")):
                break
            attempt += 1
            time.sleep(min(1.0 * 2 ** (attempt - 1), 16.0))      # 503/429: back off and retry
        wall_ms = (time.perf_counter() - t_wall) * 1000
        text = ""
        try:
            text = resp["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            pass
        usage = resp.get("usageMetadata", {})
        action = parse_answer(text) or {"move": "stay", "open": False}     # unparseable -> do nothing
        return {"action": action, "parsed": parse_answer(text) is not None, "latency_ms": ms, "wall_ms": wall_ms, "retries": attempt, "error": err,
                "input_tokens": usage.get("promptTokenCount", 0), "output_tokens": usage.get("candidatesTokenCount", 0),
                "thinking_tokens": usage.get("thoughtsTokenCount", 0), "model_version": resp.get("modelVersion"), "raw": text[:200]}
