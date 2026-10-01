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

## Result (in progress)
- Started 2026-09-30 23:04 on the M2 Max, `scripts/train_mac.sh`: 20,000 records,
  `--init_from jaredpalmer/kev-0.8b`, LoRA rank 16, lr 2e-5, batch 4 x accum 2 =
  2,500 optimizer steps, fp32 on `mps`. 11.3M trainable parameters.
- Live metrics mirrored to Weights & Biases by `scripts/wandb_tail.py`
  (Kev has no W&B hook; the tailer parses the log):
  https://wandb.ai/marvtin/skyfall-kev/runs/0vk3l20p
- Early loss: step 10 0.866, step 30 0.688, step 50 0.584 (per-record mean
  cross-entropy over the two questions). ~0.8 s/record while sharing the GPU
  with the baseline benchmark.
