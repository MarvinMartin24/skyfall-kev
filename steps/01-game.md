# Step 01: Game and oracle

## Goal
A deterministic, text-rendered skydiving grid game and a rule-based oracle
policy that is the ground truth for everything downstream.

## Plan
- `game/sim.py`: 5x6 grid, player `Y` fixed at row 1, obstacles `B`/`P`/`A`
  scroll up one row per tick, altitude 3000 -> 0 by 100 per tick.
- `game/oracle.py`: the rule (stay if clear below; else sidestep toward the
  centre into a column that is clear at rows 1 and 2; open at <= 1000 m when
  clear below).
- `game/play.py`: run N seeded episodes with `oracle`, `random`, or `model`;
  `--surprise` adds obstacle `O` never seen in training; `--show` prints ticks.
- `tests/`: determinism, collisions, dodge rule, open rules, ground.

## Acceptance
- Tests pass.
- Oracle win rate >= 0.90 over 500 seeded jumps, random <= 0.05.
- Oracle keeps winning with `--surprise` (it must treat any non-`.` as a wall).
- A state renders in under ~80 tokens.
- Code review by a second reader finds no logic bug in `step()` or `oracle()`.

## Result
_(filled after execution)_

## Review
_(filled after execution)_
