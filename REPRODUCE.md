# Reproduce everything

Two paths: **Mac (Apple Silicon, Metal)**, which is what every number in
`results/` was produced on (M2 Max, 64 GB), and **Colab (CUDA)** in
`colab/train.sh`. Commands are identical except for `--device` and `--dtype`.

## 0. Prerequisites

- Python 3.9+ for the game and data (standard library only).
- [`uv`](https://docs.astral.sh/uv/) for the Kev repo.
- About 6 GB of disk for the Qwen3.5-0.8B base, the released Kev adapter and
  the Qwen3.5-0.8B chat model used by the latency baseline.

```bash
git clone https://github.com/MarvinMartin24/skyfall-kev.git
cd skyfall-kev
export SKY="$PWD"
git clone https://github.com/jaredpalmer/kev.git ../kev   # tested at commit 1d77363
cd ../kev && git apply "$SKY/patches/kev-lora-snapshots.diff"   # adds --lora_snapshot_every (periodic loadable LoRA checkpoints)
uv sync --extra serve && export KEV="$PWD" && cd "$SKY"
```

The patch is 12 lines: it reuses Kev's own final-save code (adapter, `head.pt`,
tokenizer) every N optimizer steps into `runs/<run>-snapshots/step-<N>/`, each
of which `kev.serve` and `kev.benchmark` load like any run. We use it for the
"model at step 100 / 500 / 2500" montage in the demo.

## 1. Game and oracle (seconds)

```bash
python3 -m tests.run
python3 -m game.play --policy oracle --episodes 500
python3 -m game.play --policy random --episodes 500
python3 -m game.play --policy oracle --episodes 500 --surprise
python3 -m game.play --policy oracle --episodes 1 --show      # watch one jump
```

## 2. Data (seconds)

```bash
python3 -m data.generate --train 20000 --eval 2000
python3 -m tests.run          # includes label and leakage checks on the generated files
```

Confirm Kev loads it:

```bash
cd "$KEV" && uv run python -c "from kev.data import load_records; print(sum(1 for _ in load_records('$SKY/data/train.jsonl')))"; cd "$SKY"
```

## 3. Zero-shot baseline, released Kev-0.8B

```bash
cd "$KEV" && uv run python -m kev.benchmark --run jaredpalmer/kev-0.8b --data "$SKY/data/eval.jsonl" --out "$SKY/runs/base-eval" --device mps; cd "$SKY"
```
`--out` must not exist yet. Report lands in `runs/base-eval/report.json`.

Serve it and let it play:

```bash
cd "$KEV" && uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009 &   # wait for "listening"
cd "$SKY"
python3 -m model.eval --n 500
python3 -m game.play --policy model --episodes 200
python3 -m game.play --policy model --episodes 200 --surprise
```

## 4. Fine-tune (Mac)

```bash
KEV=../kev bash scripts/train_mac.sh      # wraps the kev.train command below, logs to runs/skyfall-0.8b.log
```

Measured: 5.8 h wall-clock on an M2 Max (64 GB), 20,000 records, 2,500 steps,
3.2 GB peak device memory. Or skip training: the final checkpoint is attached
to GitHub release `v0.1-final` (62 MB); unpack it to `runs/skyfall-0.8b/`.

While it trains, three optional watchers keep the demo material building itself:

```bash
KEV=../kev nohup bash scripts/serve_latest.sh skyfall-0.8b 8010 &        # serves the newest snapshot on :8010
nohup .venv-wandb/bin/python scripts/eval_snapshots.py --port 8010 &      # scores each served snapshot (results/learning_curve.jsonl, W&B)
nohup bash scripts/snapshot_extras.sh 8010 &                              # GIF + 50-jump win rate per snapshot (results/winrate_curve.jsonl)
```
(`.venv-wandb` is a small venv with `wandb`, `pillow`, `matplotlib`: `uv venv .venv-wandb && uv pip install --python .venv-wandb/bin/python wandb pillow matplotlib`.)

which runs

```bash
cd "$KEV" && uv run python -m kev.train \
  --data "$SKY/data/train.jsonl" --init_from jaredpalmer/kev-0.8b --base Qwen/Qwen3.5-0.8B-Base \
  --epochs 1 --lr 2e-5 --batch 4 --accum 2 --device mps --dtype fp32 \
  --lora_snapshot_every 100 --out "$SKY/runs/skyfall-0.8b"
```

Play any snapshot while training continues (one server per snapshot, pick a port):

```bash
cd "$KEV" && uv run --no-sync python -m kev.serve --run "$SKY/runs/skyfall-0.8b-snapshots/step-100" --port 8010 &
cd "$SKY" && python3 skyfall.py --policy model --url http://127.0.0.1:8010 --fps 2
```

`--dtype bf16` is CUDA-only in `kev.train`; see `colab/train.sh` for the GPU form.
Runtime on M2 Max: see `results/train.md`.

## 5. Evaluate the fine-tune

```bash
cd "$KEV" && uv run python -m kev.benchmark --run "$SKY/runs/skyfall-0.8b" --data "$SKY/data/eval.jsonl" --out "$SKY/runs/ft-eval" --device mps; cd "$SKY"
cd "$KEV" && uv run --extra serve python -m kev.serve --run "$SKY/runs/skyfall-0.8b" --port 8009 &
cd "$SKY"
python3 -m model.eval --n 2000
python3 -m game.play --policy model --episodes 200
python3 -m game.play --policy model --episodes 200 --surprise
python3 -m game.play --policy model --episodes 1 --show
```

## 6. Latency vs generation

```bash
cd "$KEV" && uv run python "$SKY/model/generative_baseline.py" --data "$SKY/data/eval.jsonl" --n 100 --device mps --out "$SKY/results/generative_baseline.json"; cd "$SKY"
python3 -m model.eval --n 100          # Kev side, through the server started in step 5
```

## 7. Figures, tables, GIFs

```bash
.venv-wandb/bin/python scripts/plot_curve.py                  # results/learning_curve.png
python3 scripts/make_table.py                                  # results/table.md from every saved result
.venv-wandb/bin/python scripts/record_gif.py --player kev:http://127.0.0.1:8010 --label "final" --seeds 7000,7001,7002 --out results/gifs/kev-final.gif
.venv-wandb/bin/python scripts/montage.py                      # seed 7000 across training, from the per-snapshot GIFs
```

## Expected outputs

Every number we got is in `results/table.md`, `results/race-*.md`, `STORY.md`
and the `steps/*.md` files, with the command that produced it.
