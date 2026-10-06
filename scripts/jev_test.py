"""Score Jev (TypeSafe /v1/systemone) on Skyfall with the same metrics as Kev.

    python3 scripts/jev_test.py smoke                 # 3 calls, prints the raw response shape
    python3 scripts/jev_test.py eval  --n 2000        # accuracy / Brier / ECE on data/eval.jsonl
    python3 scripts/jev_test.py race  --episodes 20   # live jumps, seeds 6000..

Safety: key is read from skyfall-kev/.env (JEV_API_KEY), never printed. Spend is estimated from the
usage.input_tokens each response reports and the run STOPS at --max_usd. The run also aborts after
5 consecutive failed calls, so a bad key can never be scored as 'stay, don't open'.
Results go to results/jev-eval-2000.json, jev-eval-predictions.jsonl and jev-race-6000.json.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from data.questions import QUESTIONS  # noqa: E402
from game.oracle import oracle  # noqa: E402
from game.sim import World  # noqa: E402

URL = "https://api.typesafe.ai/v1/systemone"
PRICE_IN_PER_MTOK = 0.042  # TypeSafe blog: $0.042 / MTok input, output free (verify on their pricing page)
OUT = REPO / "results"


def load_key():
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("JEV_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("JEV_API_KEY not found in skyfall-kev/.env")


class Jev:
    def __init__(self, model, max_usd):
        self.key, self.model, self.max_usd = load_key(), model, max_usd
        self.tokens_in, self.calls, self.fail_streak = 0, 0, 0
        self.served = set()

    @property
    def usd(self):
        return self.tokens_in * PRICE_IN_PER_MTOK / 1e6

    def ask(self, state, questions=QUESTIONS, retries=4):
        if self.usd >= self.max_usd:
            raise SystemExit(f"STOP: estimated spend ${self.usd:.4f} reached the cap ${self.max_usd}")
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        for attempt in range(retries + 1):
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    resp = json.load(r)
                ms = (time.perf_counter() - t0) * 1000
                self.fail_streak = 0
                self.calls += 1
                self.tokens_in += resp.get("usage", {}).get("input_tokens", 0)
                if resp.get("model"):
                    self.served.add(resp["model"])
                resp["_ms"] = ms
                return resp
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}: {e.read()[:200]!r}"
                retryable = e.code in (429, 500, 529)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                err, retryable = f"{type(e).__name__}: {e}", True
            self.fail_streak += 1
            if self.fail_streak >= 5 or not retryable or attempt == retries:
                raise SystemExit(f"ABORT after repeated failures. Last error: {err}")
            time.sleep(min(2 ** attempt, 16))


def p_true(a):
    for k in ("noul", "probability", "p_true", "p"):
        if isinstance(a.get(k), (int, float)):
            return float(a[k])
    probs = a.get("probabilities") or {}
    for k in ("true", "yes", "True"):
        if k in probs:
            return float(probs[k])
    raise KeyError(f"cannot read noul probability from {a}")


def read(resp):
    a = resp["answers"]
    mv = {k: float(v) for k, v in a["move"]["probabilities"].items()}
    return mv, p_true(a["open"])


def cmd_smoke(jev, _):
    rows = [json.loads(l) for l in open(REPO / "data" / "eval.jsonl")][:3]
    for r in rows:
        qs = {k: {kk: vv for kk, vv in v.items() if kk != "label"} for k, v in r["questions"].items()}
        resp = jev.ask(r["state"], qs)
        mv, po = read(resp)
        print("served by:", resp.get("model"), "| usage:", resp.get("usage"), f"| {resp['_ms']:.0f} ms")
        print("  move probs:", {k: round(v, 3) for k, v in mv.items()}, "gold:", r["questions"]["move"]["label"], "| P(open):", round(po, 3), "gold:", r["questions"]["open"]["label"])
    print(f"estimated spend so far: ${jev.usd:.6f} over {jev.calls} calls")


def ece(conf_correct, bins=15):
    n, total = len(conf_correct), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in conf_correct if lo < c <= hi or (b == 0 and c == 0)]
        if sel:
            total += len(sel) / n * abs(sum(ok for _, ok in sel) / len(sel) - sum(c for c, _ in sel) / len(sel))
    return total


def cmd_eval(jev, a):
    OUT.mkdir(exist_ok=True)
    acc = {"move": 0, "open": 0}
    br = {"move": 0.0, "open": 0.0}
    cc, lat, n = [], [], 0
    conf = {g: {p: 0 for p in ('left', 'right', 'stay')} for g in ('left', 'right', 'stay')}
    open_conf = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    preds = []
    for line in open(REPO / "data" / "eval.jsonl"):
        if n >= a.n:
            break
        r = json.loads(line)
        qs = {k: {kk: vv for kk, vv in v.items() if kk != "label"} for k, v in r["questions"].items()}
        resp = jev.ask(r["state"], qs)
        mv, po = read(resp)
        gm, go = r["questions"]["move"]["label"], bool(r["questions"]["open"]["label"])
        pm = max(mv, key=mv.get)
        conf[gm][pm] += 1
        open_conf[('t' if (po >= 0.5) == go else 'f') + ('p' if po >= 0.5 else 'n')] += 1
        preds.append({'gold_move': gm, 'move_probs': mv, 'gold_open': go, 'p_open': po})
        acc["move"] += pm == gm
        acc["open"] += (po >= 0.5) == go
        br["move"] += sum((mv.get(k, 0.0) - (k == gm)) ** 2 for k in ("left", "right", "stay"))
        br["open"] += (po - go) ** 2
        cc += [(mv[pm], pm == gm), (max(po, 1 - po), (po >= 0.5) == go)]
        lat.append(resp["_ms"])
        n += 1
        if n % 250 == 0:
            print(f"  {n}/{a.n}  est ${jev.usd:.4f}", flush=True)
    out = {"n_states": n, "served_by": sorted(jev.served), "move_acc": acc["move"] / n, "open_acc": acc["open"] / n,
           "move_brier": br["move"] / n, "open_brier": br["open"] / n, "overall_brier": (br["move"] + br["open"]) / (2 * n),
           "ece": ece(cc), "latency_ms_median": statistics.median(lat), "latency_ms_p95": sorted(lat)[int(0.95 * len(lat))],
           "move_confusion_gold_by_pred": conf, "open_confusion": open_conf,
           "est_usd": jev.usd, "input_tokens": jev.tokens_in}
    (OUT / "jev-eval-predictions.jsonl").write_text("\n".join(json.dumps(x) for x in preds))
    (OUT / "jev-eval-2000.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_race(jev, a):
    OUT.mkdir(exist_ok=True)
    wins, lat, decisions, agree = 0, [], 0, {"move": 0, "open": 0}
    jumps = []
    for seed in range(a.seed, a.seed + a.episodes):
        w = World(seed=seed)
        while not w.done:
            truth = oracle(w)
            resp = jev.ask(w.render())
            mv, po = read(resp)
            move, op = max(mv, key=mv.get), po >= 0.5
            agree["move"] += move == truth["move"]
            agree["open"] += op == truth["open"]
            lat.append(resp["_ms"])
            decisions += 1
            w.step(move, op)
        wins += w.outcome[0] == "win"
        jumps.append({"seed": seed, "result": w.outcome[0], "reason": w.outcome[1], "ticks": w.ticks})
        print(f"  seed {seed}: {w.outcome[0]} ({w.outcome[1]}) in {w.ticks} decisions  est ${jev.usd:.4f}", flush=True)
    out = {"episodes": a.episodes, "seeds": f"{a.seed}-{a.seed + a.episodes - 1}", "wins": wins, "win_rate": wins / a.episodes,
           "decisions": decisions, "move_agreement": agree["move"] / decisions, "open_agreement": agree["open"] / decisions,
           "latency_ms_median": statistics.median(lat), "latency_ms_mean": statistics.mean(lat), "latency_ms_p95": sorted(lat)[int(0.95 * len(lat))],
           "served_by": sorted(jev.served), "est_usd": jev.usd, "jumps": jumps}
    (OUT / "jev-race-6000.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "jumps"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke", "eval", "race"])
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=6000)
    ap.add_argument("--max_usd", type=float, default=1.0)
    a = ap.parse_args()
    jev = Jev(a.model, a.max_usd)
    {"smoke": cmd_smoke, "eval": cmd_eval, "race": cmd_race}[a.mode](jev, a)
