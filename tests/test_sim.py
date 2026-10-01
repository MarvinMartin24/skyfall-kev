from game.sim import World, EMPTY, PLAYER_ROW, WIDTH, OPEN_ALT, CLOUD
from game.oracle import oracle


def test_deterministic():
    a, b = World(seed=7), World(seed=7)
    assert a.render() == b.render()


def test_clouds_are_harmless():
    w = World(seed=1, spawn_prob=0.0, cloud_prob=0.0)
    w.grid[PLAYER_ROW + 1][w.col] = CLOUD
    assert oracle(w) == {"move": "stay", "open": False}          # a cloud below is not a reason to move
    assert w.step("stay", False) is None and "Y" in w.render()    # flew into the cloud, still alive
    w = World(seed=1, spawn_prob=0.0, cloud_prob=0.0)
    w.alt = OPEN_ALT
    w.grid[PLAYER_ROW + 1][w.col] = CLOUD
    assert oracle(w)["open"] is True
    assert w.step("stay", True) == ("win", "safe_landing")        # opening into a cloud is fine
    w = World(seed=1, spawn_prob=0.0, cloud_prob=0.0)
    w.grid[PLAYER_ROW + 1][w.col] = "B"
    w.grid[PLAYER_ROW][w.col + 1] = CLOUD                         # sidestep through a cloud is allowed
    assert oracle(w)["move"] == "right"
    assert w.step("right", False) is None


def test_dodge_and_hit():
    w = World(seed=1, spawn_prob=0.0, cloud_prob=0.0)
    w.grid[PLAYER_ROW + 1][w.col] = "B"
    assert oracle(w)["move"] == "right"   # centre column dodges right first
    w2 = World(seed=1, spawn_prob=0.0, cloud_prob=0.0)
    w2.grid[PLAYER_ROW + 1][w2.col] = "B"
    assert w2.step("stay", False) == ("lose", "hit_B")
    assert w.step("right", False) is None


def test_falls_back_to_other_side():
    w = World(seed=1, spawn_prob=0.0, cloud_prob=0.0)
    w.grid[PLAYER_ROW + 1][w.col] = "A"
    w.grid[PLAYER_ROW][w.col + 1] = "P"
    assert oracle(w)["move"] == "left"


def test_open_rules():
    w = World(seed=1, spawn_prob=0.0, cloud_prob=0.0)
    assert oracle(w)["open"] is False
    assert w.step("stay", True) == ("lose", "opened_too_high")
    w = World(seed=1, spawn_prob=0.0, cloud_prob=0.0)
    w.alt = OPEN_ALT
    assert oracle(w)["open"] is True
    assert w.step("stay", True) == ("win", "safe_landing")
    w = World(seed=1, spawn_prob=0.0, cloud_prob=0.0)
    w.alt = OPEN_ALT
    w.grid[PLAYER_ROW + 1][w.col] = "P"
    assert oracle(w) == {"move": "right", "open": False}    # sidestep first, open on a later tick
    assert w.step("right", False) is None


def test_ground():
    w = World(seed=1, spawn_prob=0.0, cloud_prob=0.0)
    out = None
    while out is None:
        out = w.step("stay", False)
    assert out == ("lose", "ground")


def test_sideways_into_row1_obstacle_and_canopy_hit():
    w = World(seed=1, spawn_prob=0.0, cloud_prob=0.0)
    w.grid[PLAYER_ROW][w.col + 1] = "A"
    assert w.step("right", False) == ("lose", "hit_A")
    assert "X" in w.render() and "Y" not in w.render()      # terminal frame marks the hit
    w = World(seed=1, spawn_prob=0.0, cloud_prob=0.0)
    w.alt = OPEN_ALT
    w.grid[PLAYER_ROW + 1][w.col] = "B"
    assert w.step("stay", True) == ("lose", "canopy_hit_B")


def test_edges_clamp_and_forced_stay():
    w = World(seed=1, spawn_prob=0.0, cloud_prob=0.0)
    w.col = 0
    assert w.step("left", False) is None and w.col == 0
    w.col = WIDTH - 1
    assert w.step("right", False) is None and w.col == WIDTH - 1
    w = World(seed=1, spawn_prob=0.0, cloud_prob=0.0)
    w.col = 0
    w.grid[PLAYER_ROW + 1][0] = "B"
    w.grid[PLAYER_ROW][1] = "P"
    assert oracle(w)["move"] == "stay"                       # nowhere safe: forced


def test_surprise_keeps_layouts():
    """Turning the balloon on must not move anything: same seed, same positions, only some glyphs become O."""
    import re
    for seed in range(50):
        a, b = World(seed=seed), World(seed=seed, surprise=True)
        for _ in range(12):
            ra, rb = a.render(), b.render()
            assert re.sub("[BPA]", "#", ra) == re.sub("[BPAO]", "#", rb), (ra, rb)   # clouds C must match exactly
            a.step("stay", False); b.step("stay", False)
            if a.done or b.done:
                break


def test_oracle_properties():
    """On random reachable states: the oracle never opens into a loss, and never picks a move that dies
    this tick when another move would survive it."""
    import copy, random
    from game.sim import MOVES
    rng = random.Random(0)
    checked = 0
    for seed in range(300):
        w = World(seed=seed)
        while not w.done:
            a = oracle(w)
            if a["open"]:
                assert copy.deepcopy(w).step(a["move"], True) == ("win", "safe_landing")
            survivors = {m for m in MOVES if copy.deepcopy(w).step(m, False) is None}
            if survivors:
                assert a["move"] in survivors, (w.render(), a, survivors)
            checked += 1
            m = a["move"] if rng.random() < 0.7 else rng.choice(MOVES)
            w.step(m, a["open"] and m == a["move"])
    assert checked > 3000
