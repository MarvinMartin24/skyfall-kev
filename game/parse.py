"""Rebuild a World from its rendered text. Used to verify dataset labels and for demos."""
from game.sim import World, WIDTH, HEIGHT, PLAYER, PLAYER_ROW, EMPTY


def parse_state(text):
    lines = text.strip("\n").split("\n")
    assert lines[0].startswith("altitude: "), lines[0]
    w = World(seed=0, spawn_prob=0.0, cloud_prob=0.0)
    w.alt = int(lines[0].split(": ")[1])
    rows = [line.split(" ") for line in lines[1:]]
    assert len(rows) == HEIGHT and all(len(r) == WIDTH for r in rows), rows
    col = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == PLAYER:
                assert r == PLAYER_ROW and col is None
                col = c
                row[c] = EMPTY
    assert col is not None, "no player"
    w.grid, w.col = rows, col
    return w
