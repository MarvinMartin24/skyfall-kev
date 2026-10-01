"""The ground-truth policy. This is what the model has to learn from text.

Rule, in words ("clear" = empty or a cloud; clouds are harmless):
  1. If the cell below you is clear, stay.
  2. Otherwise step sideways, toward the centre of the grid first, into a
     column whose current cell and cell below are both clear.
  3. Open the parachute only at or below 1000 m, and only when the cell
     directly below you is already clear. If something deadly is below you,
     sidestep this tick and open on a later one.
Each answer is a pure function of the state, which is how Kev answers them:
the two questions never see each other.
"""
from game.sim import WIDTH, PASSABLE, PLAYER_ROW, OPEN_ALT


def _safe(w, c):
    return 0 <= c < WIDTH and w.cell(PLAYER_ROW, c) in PASSABLE and w.cell(PLAYER_ROW + 1, c) in PASSABLE


def oracle(w):
    col = w.col
    toward_centre = "left" if col > WIDTH // 2 else "right"
    other = "right" if toward_centre == "left" else "left"
    delta = {"left": -1, "right": 1}
    if w.cell(PLAYER_ROW + 1, col) in PASSABLE:
        move = "stay"
    elif _safe(w, col + delta[toward_centre]):
        move = toward_centre
    elif _safe(w, col + delta[other]):
        move = other
    else:
        move = "stay"
    open_ = w.alt <= OPEN_ALT and w.cell(PLAYER_ROW + 1, col) in PASSABLE
    return {"move": move, "open": open_}
