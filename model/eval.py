"""Score a Kev endpoint against the oracle labels in a JSONL file.

python -m model.eval --url http://127.0.0.1:8009 --data data/eval.jsonl --n 500
Prints per-question accuracy, Brier score, and mean latency.
"""
import argparse
import json

from model.client import ask, probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8009")
    p.add_argument("--data", default="data/eval.jsonl")
    p.add_argument("--n", type=int, default=500)
    a = p.parse_args()

    n = 0
    acc = {"move": 0, "open": 0}
    brier = {"move": 0.0, "open": 0.0}
    ms = 0.0
    with open(a.data) as f:
        for line in f:
            if n >= a.n:
                break
            r = json.loads(line)
            qs = {k: {kk: vv for kk, vv in v.items() if kk != "label"} for k, v in r["questions"].items()}
            resp = ask(a.url, r["state"], qs)
            pr = probs(resp)
            ms += resp.get("latency_ms", resp["_client_ms"])
            n += 1
            lab_move = r["questions"]["move"]["label"]
            lab_open = bool(r["questions"]["open"]["label"])
            acc["move"] += max(pr["move"], key=pr["move"].get) == lab_move
            acc["open"] += (pr["open"] >= 0.5) == lab_open
            brier["move"] += sum((pr["move"].get(k, 0.0) - (1.0 if k == lab_move else 0.0)) ** 2
                                 for k in ("left", "right", "stay"))
            brier["open"] += (pr["open"] - (1.0 if lab_open else 0.0)) ** 2
    print(json.dumps({
        "n": n,
        "move_acc": round(acc["move"] / n, 3), "open_acc": round(acc["open"] / n, 3),
        "move_brier": round(brier["move"] / n, 3), "open_brier": round(brier["open"] / n, 3),
        "mean_latency_ms": round(ms / n, 1),
    }, indent=1))


if __name__ == "__main__":
    main()
