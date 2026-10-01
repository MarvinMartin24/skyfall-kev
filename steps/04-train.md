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

## Result (in progress; attempts 1–4 were restarted after data fixes and the clouds change, see steps/02-data.md)
- Attempt 5 (final recipe) started 2026-09-30 23:50 with `--lora_snapshot_every 100`
  (our 12-line patch to Kev, `patches/kev-lora-snapshots.diff`), so a loadable
  checkpoint exists every 100 steps for the learning montage and for playing
  early models. Kev itself only writes LoRA runs at the end.
- Started 2026-09-30 23:04 on the M2 Max, `scripts/train_mac.sh`: 20,000 records,
  `--init_from jaredpalmer/kev-0.8b`, LoRA rank 16, lr 2e-5, batch 4 x accum 2 =
  2,500 optimizer steps, fp32 on `mps`. 11.3M trainable parameters.
- Live metrics mirrored to Weights & Biases by `scripts/wandb_tail.py`
  (Kev has no W&B hook; the tailer parses the log):
  https://wandb.ai/marvtin/skyfall-kev/runs/0vk3l20p
- Loss: step 50 0.58, step 250 0.57 (end of OneCycle warm-up), step 300 0.50,
  step 400 0.37. About 1.0–1.2 s/record on `mps` while sharing the GPU with
  servers and (for a while) a probe run.
- Learning curve on 300 held-out states (`results/learning_curve.jsonl`,
  W&B run `skyfall-0.8b-eval-curve`): move 0.51 / 0.50 / 0.57 / 0.69 and open
  0.89 / 0.93 / 0.93 / 0.94 at steps 100 / 200 / 300 / 400. Altitude first,
  grid later.
- Probe `runs/probe-lr1e-4` (lr 1e-4, 300 steps, batch 4, accum 1): move 0.50
  at steps 100, 200 and 300. Higher LR did not help; retired.
