"""The ground-truth policy. This is what the model has to learn from text.

Rule, in words:
  1. If the cell below you is empty, stay.
  2. Otherwise step sideways, toward the centre of the grid first, into a
     column whose current cell and cell below are both empty.
  3. Open the parachute only at or below 1000 m, and only when the cell below
     (after your move) is empty.
"""
from game.sim import WIDTH, EMPTY, PLAYER_ROW, OPEN_ALT


def _safe(w, c):
    return 0 <= c < WIDTH and w.cell(PLAYER_ROW, c) == EMPTY and w.cell(PLAYER_ROW + 1, c) == EMPTY


def oracle(w):
    col = w.col
    toward_centre = "left" if col > WIDTH // 2 else "right"
    other = "right" if toward_centre == "left" else "left"
    delta = {"left": -1, "right": 1}
    if w.cell(PLAYER_ROW + 1, col) == EMPTY:
        move = "stay"
    elif _safe(w, col + delta[toward_centre]):
        move = toward_centre
    elif _safe(w, col + delta[other]):
        move = other
    else:
        move = "stay"
    new_col = min(WIDTH - 1, max(0, col + delta.get(move, 0)))
    open_ = w.alt <= OPEN_ALT and w.cell(PLAYER_ROW + 1, new_col) == EMPTY
    return {"move": move, "open": open_}
