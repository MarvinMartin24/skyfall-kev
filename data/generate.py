"""Generate training/eval JSONL from the oracle. No gameplay, no screenshots.

python -m data.generate --train 20000 --eval 2000
"""
import argparse, json, random
from game.sim import World, OPEN_ALT
from game.oracle import oracle
from data.questions import QUESTIONS


def key(text):
    """The decision-relevant content of a state: the grid, and which side of 1000 m it is on.
    Two states with equal keys have identical oracle labels, so cross-split exclusion is keyed on it
    (exact-text exclusion alone left 9% of eval grids in train at another altitude)."""
    alt_line, grid = text.split("\n", 1)
    return f"{int(alt_line.split(': ')[1]) <= OPEN_ALT}|{grid}"


def example(w, labels):
    qs = {}
    for name, q in QUESTIONS.items():
        qs[name] = dict(q, label=labels[name])
    return {"state": w.render(), "questions": qs}


def collect(seed_start, n_target, eps=0.3, mix=(0.45, 0.15, 0.40), exclude=frozenset()):
    """Play oracle episodes with eps-random moves so we also see bad positions.
    Labels always come from the oracle. Three buckets, mixed to fixed ratios:
      dodge: move != stay        open: move == stay and open == True
      idle:  everything else (stay, no open)
    States whose key() is in `exclude` (the eval set's keys) are never emitted."""
    rng = random.Random(seed_start)
    seen, buckets = set(), {"dodge": [], "open": [], "idle": []}
    need = {k: int(n_target * f) for k, f in zip(buckets, mix)}
    need["idle"] = n_target - need["dodge"] - need["open"]
    seed = seed_start
    while any(len(buckets[k]) < need[k] for k in buckets):
        w = World(seed=seed); seed += 1
        while not w.done:
            lab = oracle(w)
            text = w.render()
            if text not in seen and key(text) not in exclude:
                seen.add(text)
                k = "dodge" if lab["move"] != "stay" else ("open" if lab["open"] else "idle")
                buckets[k].append(example(w, lab))
            a = dict(lab)
            if rng.random() < eps:
                a["move"] = rng.choice(["left", "right", "stay"])
            if a["open"] and rng.random() < 0.6:
                a["open"] = False          # keep falling below 1000 m to collect more open states
            elif not a["open"] and rng.random() < 0.02:
                a["open"] = True
            w.step(a["move"], a["open"])
    rows = []
    for k in buckets:
        rng.shuffle(buckets[k]); rows += buckets[k][: need[k]]
    rng.shuffle(rows)
    assert len(rows) == n_target, (len(rows), n_target)
    return rows


def stats(rows):
    n = len(rows)
    mv = {k: 0 for k in ("left", "right", "stay")}
    op, low_not_open = 0, 0
    for r in rows:
        mv[r["questions"]["move"]["label"]] += 1
        op += r["questions"]["open"]["label"]
        low = int(r["state"].split("\n", 1)[0].split(": ")[1]) <= OPEN_ALT
        low_not_open += low and not r["questions"]["open"]["label"]
    return {"n": n, "move": {k: round(v / n, 3) for k, v in mv.items()}, "open_true": round(op / n, 3),
            "low_alt_but_blocked": round(low_not_open / n, 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=20000)
    p.add_argument("--eval", type=int, default=2000)
    p.add_argument("--out", default="data")
    a = p.parse_args()
    # eval first, then train with every eval state excluded: no leakage across splits
    eval_rows = collect(2_000_000, a.eval)
    train_rows = collect(1_000_000, a.train, exclude=frozenset(key(r["state"]) for r in eval_rows))
    for name, rows in (("train", train_rows), ("eval", eval_rows)):
        path = f"{a.out}/{name}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(name, path, stats(rows))


if __name__ == "__main__":
    main()
