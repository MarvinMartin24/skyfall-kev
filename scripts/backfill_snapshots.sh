#!/usr/bin/env bash
# Evaluate, play and record snapshots that the live watcher skipped. Usage: KEV=... bash scripts/backfill_snapshots.sh 8012 step-1000 step-1100 ...
SKY="$(cd "$(dirname "$0")/.." && pwd)"; cd "$SKY"; KEV="${KEV:-$SKY/../kev}"; PORT="$1"; shift
for name in "$@"; do
  dir="$SKY/runs/skyfall-0.8b-snapshots/$name"; step=${name#step-}
  pkill -f "kev.serve.*--port $PORT" 2>/dev/null; sleep 2
  (cd "$KEV" && nohup uv run --no-sync python -m kev.serve --run "$dir" --port "$PORT" > "$SKY/runs/serve-backfill.log" 2>&1 &)
  for i in $(seq 1 60); do curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" && break; sleep 3; done
  m=$(python3 -m model.eval --url "http://127.0.0.1:$PORT" --n 300 | python3 -c "import json,sys; d=json.load(sys.stdin); d.update(step=$step, run_dir='$dir', time='$(date '+%Y-%m-%d %H:%M:%S')', backfilled=True); print(json.dumps(d))")
  echo "$m" >> results/learning_curve.jsonl
  out=$(python3 -m game.play --policy model --url "http://127.0.0.1:$PORT" --episodes 50 --seed 8000)
  wr=$(echo "$out" | grep -o "win_rate=[0-9.]*" | cut -d= -f2); mt=$(echo "$out" | grep -o "mean_ticks=[0-9.]*" | cut -d= -f2)
  echo "{\"step\": $step, \"snapshot\": \"$name\", \"win_rate_50\": $wr, \"mean_ticks\": $mt, \"time\": \"$(date '+%Y-%m-%d %H:%M')\", \"backfilled\": true}" >> results/winrate_curve.jsonl
  .venv-wandb/bin/python scripts/record_gif.py --player "kev:http://127.0.0.1:$PORT" --label "Kev fine-tuned, snapshot $name" --seeds 7000,7001,7002 --out "results/gifs/kev-$name.gif" > /dev/null 2>&1
  echo "$(date +%H:%M) $name: $(echo "$m" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"move {d['move_acc']:.3f} open {d['open_acc']:.3f} brier {d['move_brier']:.3f}/{d['open_brier']:.3f}\")") win_rate_50=$wr mean_ticks=$mt"
done
pkill -f "kev.serve.*--port $PORT" 2>/dev/null; echo "backfill done"
