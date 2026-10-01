from game.sim import World, EMPTY, PLAYER_ROW, WIDTH, OPEN_ALT
from game.oracle import oracle


def test_deterministic():
    a, b = World(seed=7), World(seed=7)
    assert a.render() == b.render()


def test_dodge_and_hit():
    w = World(seed=1, spawn_prob=0.0)
    w.grid[PLAYER_ROW + 1][w.col] = "B"
    assert oracle(w)["move"] == "right"   # centre column dodges right first
    w2 = World(seed=1, spawn_prob=0.0)
    w2.grid[PLAYER_ROW + 1][w2.col] = "B"
    assert w2.step("stay", False) == ("lose", "hit_B")
    assert w.step("right", False) is None


def test_falls_back_to_other_side():
    w = World(seed=1, spawn_prob=0.0)
    w.grid[PLAYER_ROW + 1][w.col] = "A"
    w.grid[PLAYER_ROW][w.col + 1] = "P"
    assert oracle(w)["move"] == "left"


def test_open_rules():
    w = World(seed=1, spawn_prob=0.0)
    assert oracle(w)["open"] is False
    assert w.step("stay", True) == ("lose", "opened_too_high")
    w = World(seed=1, spawn_prob=0.0)
    w.alt = OPEN_ALT
    assert oracle(w)["open"] is True
    assert w.step("stay", True) == ("win", "safe_landing")
    w = World(seed=1, spawn_prob=0.0)
    w.alt = OPEN_ALT
    w.grid[PLAYER_ROW + 1][w.col] = "P"
    assert oracle(w)["open"] is False or oracle(w)["move"] != "stay"


def test_ground():
    w = World(seed=1, spawn_prob=0.0)
    out = None
    while out is None:
        out = w.step("stay", False)
    assert out == ("lose", "ground")
