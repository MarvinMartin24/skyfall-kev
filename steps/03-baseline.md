# Step 03: Zero-shot baseline (released Kev-0.8B)

## Goal
Show that the released model does not already play the game. This is the
"before" row of the demo table.

## Plan
- `kev.benchmark --run jaredpalmer/kev-0.8b --data data/eval.jsonl` on `mps`.
- `kev.serve --run jaredpalmer/kev-0.8b`, then `model/eval.py` (our own
  accuracy / Brier / latency) and `game/play.py --policy model` for 200 jumps,
  with and without `--surprise`.
- Record a terminal clip of the base model dying (asciinema or plain log).

## Acceptance
- Numbers recorded in `results/baseline.json` and in the table in `PLAN.md`.
- If base win rate > 0.20 the game is too easy: raise spawn probability,
  regenerate data, rerun this step.

## Result
_(filled after execution)_
