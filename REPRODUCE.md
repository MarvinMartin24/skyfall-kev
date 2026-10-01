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
cd ../kev && uv sync --extra serve && export KEV="$PWD" && cd "$SKY"
```

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
bash scripts/train_mac.sh         # wraps the kev.train command below, logs to runs/skyfall-0.8b.log
```

which runs

```bash
cd "$KEV" && uv run python -m kev.train \
  --data "$SKY/data/train.jsonl" --init_from jaredpalmer/kev-0.8b --base Qwen/Qwen3.5-0.8B-Base \
  --epochs 1 --lr 2e-5 --batch 4 --accum 2 --device mps --dtype fp32 \
  --out "$SKY/runs/skyfall-0.8b"
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

## Expected outputs

Every number we got is in `results/table.md`, `results/latency.md` and the
`steps/*.md` files, with the command that produced it.
