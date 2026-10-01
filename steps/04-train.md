# Step 04: Fine-tune Kev-0.8B on Skyfall

## Goal
A checkpoint that clears the bar: `move` accuracy >= 0.90, `open` >= 0.95 on
`eval.jsonl`, trained on one device.

## Plan
- `kev.train --data data/train.jsonl --init_from jaredpalmer/kev-0.8b`,
  rank-16 LoRA, bf16, `--device mps` here (or `cuda` on Colab).
- Start with 1 epoch on 20k rows and measure; add epochs only if needed.
- Save to `runs/skyfall-0.8b/` (gitignored). Record wall-clock, loss curve
  from `training_metrics.json`, hardware.

## Ladder if the bar is missed
1. 2 epochs. 2. 40k rows. 3. Kev-4B (`--batch 1 --accum 8 --checkpointing 1`).
4. Shrink the grid to 3 columns.

## Acceptance
- `kev.benchmark` and `model/eval.py` agree within 1 point.
- Loss decreases monotonically per epoch; no NaNs.
- Review: training config and metrics committed under `results/`.

## Result
_(filled after execution)_
