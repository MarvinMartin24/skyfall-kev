"""Thin client for a Kev / System One endpoint (kev.serve, POST /v1/systemone).

Returns per-question probabilities. Response shapes verified against kev/api.py
on 2026-09-30: choice -> {"choice", "confidence", "probabilities"}, noul -> {"noul": p}.
"""
import json
import os
import time
import urllib.request

from data.questions import QUESTIONS


def ask(url, state, questions=QUESTIONS, model="kev-latest", timeout=30):
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/systemone", data=body,
                                 headers={"Content-Type": "application/json"})
    key = os.environ.get("KEV_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    t0 = time.perf_counter()
    for attempt in range(20):                       # the server may be restarting on a newer snapshot: wait for it
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.load(r)
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if attempt == 19:
                raise
            time.sleep(3)
    resp["_client_ms"] = (time.perf_counter() - t0) * 1000
    return resp


def p_true(answer):
    """Probability of 'yes' for a noul answer. kev.serve returns {"type": "noul", "noul": p}."""
    for k in ("noul", "probability", "p_true", "p"):
        if isinstance(answer.get(k), (int, float)):
            return float(answer[k])
    probs = answer.get("probabilities")
    if isinstance(probs, dict):
        for k in ("true", "True", "yes"):
            if k in probs:
                return float(probs[k])
    if isinstance(answer.get("answer"), bool):
        return 1.0 if answer["answer"] else 0.0
    raise KeyError(f"cannot read noul probability from {answer}")


def probs(resp):
    """-> {"move": {"left": p, "right": p, "stay": p}, "open": p_true}"""
    a = resp["answers"]
    return {"move": {k: float(v) for k, v in a["move"]["probabilities"].items()},
            "open": p_true(a["open"])}


def model_policy(url, show=False):
    def policy(w):
        pr = probs(ask(url, w.render()))
        move = max(pr["move"], key=pr["move"].get)
        open_ = pr["open"] >= 0.5
        if show:
            bars = "  ".join(f"{k}:{v:.2f}" for k, v in pr["move"].items())
            print(f"   P[{bars}]  P(open)={pr['open']:.2f}")
        return {"move": move, "open": open_}
    return policy
