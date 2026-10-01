"""Skyfall: a tiny grid skydiving game. No I/O, deterministic per seed."""
import random

WIDTH, HEIGHT = 5, 6
PLAYER_ROW = 1          # row 0 is "already passed"; obstacles scroll up and out
EMPTY = "."
PLAYER = "Y"
OBSTACLES = ("B", "P", "A")   # deadly: bird, other skydiver under parachute, airplane
CLOUD, CLOUD_PROB = "C", 0.3  # harmless: you fly straight through clouds (spawned independently of obstacles)
PASSABLE = (EMPTY, CLOUD)
SURPRISE, SURPRISE_PROB = "O", 0.25   # hot-air balloon, deadly: never in training data; --surprise swaps ~25% of obstacles for it
DEAD = "X"                   # drawn where the skydiver was hit, on terminal frames only
START_ALT = 3000
ALT_STEP = 100
OPEN_ALT = 1000          # may only open at or below this altitude
MOVES = ("left", "right", "stay")


class World:
    def __init__(self, seed=0, spawn_prob=0.6, cloud_prob=CLOUD_PROB, surprise=False):
        self.rng = random.Random(seed)
        # a separate stream for the surprise swap, so turning it on never changes the layouts for a seed
        self.surprise_rng = random.Random(seed ^ 0x5EED)
        self.surprise = surprise
        self.spawn_prob = spawn_prob
        self.cloud_prob = cloud_prob
        self.col = WIDTH // 2
        self.alt = START_ALT
        self.ticks = 0
        self.done = False
        self.outcome = None
        self.grid = [[EMPTY] * WIDTH for _ in range(HEIGHT)]
        for r in range(PLAYER_ROW + 1, HEIGHT):
            self.grid[r] = self._new_row()

    def _new_row(self):
        row = [EMPTY] * WIDTH
        if self.rng.random() < self.spawn_prob:
            sym = self.rng.choice(OBSTACLES)      # type first, then column: this call order fixes the layouts
            col = self.rng.randrange(WIDTH)
            if self.surprise and self.surprise_rng.random() < SURPRISE_PROB:
                sym = SURPRISE
            row[col] = sym
        if self.rng.random() < self.cloud_prob:
            col = self.rng.randrange(WIDTH)
            if row[col] == EMPTY:
                row[col] = CLOUD
        return row

    def cell(self, row, col):
        return self.grid[row][col]

    def render(self):
        lines = [f"altitude: {self.alt}"]
        for r, row in enumerate(self.grid):
            cells = list(row)
            if r == PLAYER_ROW:
                cells[self.col] = PLAYER if row[self.col] in PASSABLE else DEAD
            lines.append(" ".join(cells))
        return "\n".join(lines)

    def _end(self, result, reason):
        self.done, self.outcome = True, (result, reason)
        return self.outcome

    def step(self, move, open_):
        """Apply one tick. Returns None while the game continues, else (result, reason)."""
        assert not self.done, "game is over"
        assert move in MOVES, move
        self.ticks += 1          # ticks = decisions taken, including the terminal one
        if move == "left":
            self.col = max(0, self.col - 1)
        elif move == "right":
            self.col = min(WIDTH - 1, self.col + 1)
        here = self.grid[PLAYER_ROW][self.col]
        if here not in PASSABLE:
            return self._end("lose", f"hit_{here}")
        if open_:
            if self.alt > OPEN_ALT:
                return self._end("lose", "opened_too_high")
            below = self.grid[PLAYER_ROW + 1][self.col]
            if below not in PASSABLE:
                return self._end("lose", f"canopy_hit_{below}")
            return self._end("win", "safe_landing")
        self.grid.pop(0)
        self.grid.append(self._new_row())
        self.alt -= ALT_STEP
        here = self.grid[PLAYER_ROW][self.col]
        if here not in PASSABLE:
            return self._end("lose", f"hit_{here}")
        if self.alt <= 0:
            return self._end("lose", "ground")
        return None
