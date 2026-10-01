#!/usr/bin/env bash
# Fine-tune Kev-0.8B on Skyfall on an Apple Silicon Mac (Metal via torch mps).
# Usage: KEV=/path/to/kev bash scripts/train_mac.sh [run-name] [extra kev.train flags...]
# Flags verified against kev/train.py at commit 1d77363:
#   --dtype bf16 is CUDA-only, so fp32 here. --out must not exist. --init_from = delta mode.
#   --base and --base_revision match the released checkpoint's training_config.json.
set -euo pipefail
SKY="$(cd "$(dirname "$0")/.." && pwd)"
KEV="${KEV:-$SKY/../kev}"
RUN="${1:-skyfall-0.8b}"; shift || true
OUT="$SKY/runs/$RUN"
[ -e "$OUT" ] && { echo "refusing: $OUT exists"; exit 1; }
cd "$KEV"
echo "train -> $OUT (log: $OUT.log)"
{ time uv run python -m kev.train \
  --data "$SKY/data/train.jsonl" \
  --init_from jaredpalmer/kev-0.8b \
  --base Qwen/Qwen3.5-0.8B-Base --base_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --epochs 1 --lr 2e-5 --batch 4 --accum 2 \
  --device mps --dtype fp32 \
  --lora_snapshot_every 100 \
  --out "$OUT" "$@" ; } 2>&1 | tee "$OUT.log"
