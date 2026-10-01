"""Move confusion matrix and open error types for a served checkpoint, on held-out states.
python3 -m model.confusion --url http://127.0.0.1:8010 --n 300"""
import argparse, json
from model.client import ask, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010"); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--data", default="data/eval.jsonl"); ap.add_argument("--show_errors", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for _, l in zip(range(a.n), open(a.data))]
    labels = ("left", "right", "stay"); conf = {}; errs = []; open_err = {"false_open": 0, "missed_open": 0}
    for r in rows:
        pr = probs(ask(a.url, r["state"]))
        pred = max(pr["move"], key=pr["move"].get); true = r["questions"]["move"]["label"]
        conf[(true, pred)] = conf.get((true, pred), 0) + 1
        if pred != true:
            errs.append((true, pred, pr["move"], r["state"]))
        po, to = pr["open"] >= 0.5, bool(r["questions"]["open"]["label"])
        open_err["false_open"] += po and not to; open_err["missed_open"] += to and not po
    print("rows = oracle, cols = model")
    print("           " + "  ".join(f"{l:>5}" for l in labels))
    for t in labels:
        print(f"true {t:>5}: " + "  ".join(f"{conf.get((t, p), 0):5d}" for p in labels))
    print("open errors:", open_err, "| move errors:", len(errs), "of", len(rows))
    for true, pred, pm, state in errs[: a.show_errors]:
        print(f"\n--- oracle {true}, model {pred} ({', '.join(f'{k} {v:.2f}' for k, v in pm.items())})\n{state}")


if __name__ == "__main__":
    main()
