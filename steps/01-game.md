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
- 12 tests pass (`python3 -m tests.run`), including a property test over ~3,000
  reachable states: the oracle never opens into a loss and never takes a move
  that dies this tick when another move survives it.
- 500 seeded jumps: oracle 0.952 win rate (mean 20.6 decisions), random 0.002.
- `--surprise` (balloon `O` in ~25% of spawns) on identical layouts: oracle 0.952.
- A state is 43 Qwen tokens (measured with the Kev-0.8B tokenizer).

## Review
Independent reviewer (read-only, exhaustive enumeration of all 64,800 reachable
(col, rows 1–2, altitude) states plus replay of seeds 0–4999). Verdict: **ACCEPT**,
no logic bug in `step()` or `oracle()`. Findings and what we did:

| # | Severity | Finding | Action |
|---|---|---|---|
| 1 | risk | The oracle is 1-step optimal with no lookahead; ~9 of its 24 deaths in 500 jumps were avoidable from the visible grid. 0.95 is the rule's ceiling, not the game's (~0.97). | Kept on purpose (simplicity is the point). Relabelled "oracle (the rule)" instead of "ceiling" in PLAN and docs. |
| 2 | risk | `--surprise` changed the RNG consumption, so balloon jumps had different layouts than plain jumps for the same seed. | **Fixed**: the balloon swap uses a separate RNG stream; `test_surprise_keeps_layouts` asserts positions are identical with and without it. Training data verified byte-identical before and after the change. |
| 3 | nit | Terminal frame drew `Y` over the obstacle that killed it. | Fixed: terminal frame shows `X` at the hit cell. |
| 4 | nit | Random policy shared one RNG across episodes, so single episodes were not reproducible by seed. | Fixed: per-episode RNG from the episode seed. |
| 5 | nit | `ticks` counted the terminal tick inconsistently. | Fixed: `ticks` = decisions taken, including the terminal one. |
| 6 | nit | One test assertion was too weak. | Fixed, plus 4 new tests for sideways hits, canopy hits, edge clamping and forced states. |
