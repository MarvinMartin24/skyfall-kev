# Copy these cells into Colab (L4 or A100 runtime). Prefix shell lines with "!" in a cell.
# Flags verified against kev/train.py, kev/benchmark.py and kev/serve.py on 2026-09-30.

# 0. GPU check: must print True, otherwise change the runtime (T4 has no bf16)
python -c "import torch; print(torch.cuda.is_bf16_supported(), torch.cuda.get_device_name())"

# 1. Install Kev (+ our 8-line patch for periodic LoRA snapshots)
pip install -q uv
git clone https://github.com/jaredpalmer/kev.git
cd kev && git apply /content/drive/MyDrive/skyfall/patches/kev-lora-snapshots.diff && uv sync --extra serve

# 2. Mount Drive, upload the skyfall repo there, point at it
#    from google.colab import drive; drive.mount('/content/drive')
export SKY=/content/drive/MyDrive/skyfall
export RUNS=/content/drive/MyDrive/skyfall-runs

# 3. Zero-shot baseline: released Kev-0.8B on our eval set
uv run python -m kev.benchmark --run jaredpalmer/kev-0.8b --data $SKY/data/eval.jsonl --out $RUNS/base-eval --device cuda

# 4. Fine-tune (0.8B, ~20 min on an H100; longer on an L4). No validation split: evaluate afterwards.
#    "delta mode": --init_from warm-starts the LoRA and pointer head from the released checkpoint.
#    --base must be the backbone the released checkpoint was trained on: Qwen/Qwen3.5-0.8B-Base
#    (from https://huggingface.co/jaredpalmer/kev-0.8b/raw/main/training_config.json; it used lora 16, lr 2e-5, batch 4, accum 2).
#    --dtype bf16 is CUDA-only in kev.train; on a Mac use --device mps --dtype fp32 (see REPRODUCE.md).
uv run python -m kev.train --data $SKY/data/train.jsonl --init_from jaredpalmer/kev-0.8b \
  --base Qwen/Qwen3.5-0.8B-Base --epochs 1 --lr 2e-5 --batch 4 --accum 2 \
  --dtype bf16 --device cuda --checkpointing 1 --lora_snapshot_every 100 --out $RUNS/skyfall-0.8b
#    If memory is tight on an L4: --batch 1 --accum 8

# 5. Evaluate the fine-tune the same way
uv run python -m kev.benchmark --run $RUNS/skyfall-0.8b --data $SKY/data/eval.jsonl --out $RUNS/ft-eval --device cuda

# 6. Serve, then play and score from the skyfall repo (run in a second cell / background)
uv run --extra serve python -m kev.serve --run $RUNS/skyfall-0.8b --port 8009 &
cd $SKY
python -m model.eval --url http://127.0.0.1:8009 --n 500
python -m game.play --policy model --url http://127.0.0.1:8009 --episodes 200
python -m game.play --policy model --url http://127.0.0.1:8009 --episodes 200 --surprise
python -m game.play --policy model --url http://127.0.0.1:8009 --episodes 1 --show
