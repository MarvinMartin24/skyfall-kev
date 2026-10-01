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
`kev.benchmark --run jaredpalmer/kev-0.8b --data data/eval.jsonl --device mps` (v4 data,
2,000 states, 4,000 questions, ~40 min while sharing the GPU with training). Full
report in `runs/base-eval/report.json`, summary in `results/baseline.json`.

| question | accuracy | Brier | NLL | note |
|---|---|---|---|---|
| move (3-way) | 0.554 | 0.626 | 1.04 | equals the "always stay" rate (55% of eval is stay); uniform would be 0.667 Brier |
| open (yes/no) | 0.467 | 0.506 | 0.70 | worse than "always no" (0.85); mean P(open) is 0.53 at every altitude |
| both | 0.511 | 0.566 | 0.87 | ECE 0.12, mean confidence 0.46 |

Interpretation: the released model has no idea what the grid means. Its move
probabilities sit near uniform (typical answer 0.27 / 0.35 / 0.38) and it opens
the parachute about half the time regardless of altitude, so it loses almost
every jump by opening above 1000 m. This is the "before" row.

Live play (`game.play --policy model --episodes 200` against `kev.serve` on the
released checkpoint): see `results/base-play.txt`.
