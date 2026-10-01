"""Dataset integrity: every label must equal the oracle's decision for the
state as written, and no state may appear in both train and eval."""
import json, os
from game.parse import parse_state
from game.oracle import oracle
from game.sim import World

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rows(name):
    path = os.path.join(HERE, "data", f"{name}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f]


def test_parse_roundtrip():
    w = World(seed=5)
    for _ in range(4):
        w.step("stay", False)
    w2 = parse_state(w.render())
    assert w2.render() == w.render()
    assert oracle(w2) == oracle(w)


def test_labels_match_oracle():
    checked = 0
    for name in ("train", "eval"):
        for r in _rows(name):
            lab = oracle(parse_state(r["state"]))
            assert r["questions"]["move"]["label"] == lab["move"], r["state"]
            assert r["questions"]["open"]["label"] == lab["open"], r["state"]
            checked += 1
    assert checked > 0, "no data files, run python -m data.generate"


def test_no_overlap_between_splits():
    from data.generate import key
    train = {r["state"] for r in _rows("train")}
    ev = {r["state"] for r in _rows("eval")}
    assert train and ev
    assert not (train & ev), f"{len(train & ev)} states leak from eval into train"
    leak = {key(s) for s in train} & {key(s) for s in ev}
    assert not leak, f"{len(leak)} eval grids appear in train with the same decision inputs"
    assert len(train) == len(_rows("train")) and len(ev) == len(_rows("eval")), "duplicates within a split"


def test_open_clause_is_exercised():
    """The eval must contain low-altitude states where opening is wrong because the cell below is blocked."""
    from game.sim import OPEN_ALT
    n = sum(1 for r in _rows("eval")
            if int(r["state"].split("\n", 1)[0].split(": ")[1]) <= OPEN_ALT and not r["questions"]["open"]["label"])
    assert n >= 50, n
