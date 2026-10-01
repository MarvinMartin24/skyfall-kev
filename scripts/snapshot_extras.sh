#!/usr/bin/env bash
# For every snapshot the :8010 server serves: record the montage GIF (seeds 7000-7002) and a 50-jump
# win rate (seeds 8000-8049), appending to results/winrate_curve.jsonl. Stops after the final checkpoint.
SKY="$(cd "$(dirname "$0")/.." && pwd)"; cd "$SKY"
PORT="${1:-8010}"; done_list=""
while true; do
  cur=$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['models'][0]['run'])" 2>/dev/null)
  name=$(basename "$cur"); [ "$name" = "skyfall-0.8b" ] && name="final"
  if [ -n "$cur" ] && ! echo "$done_list" | grep -q "|$cur|"; then
    label="Kev fine-tuned, snapshot $name"; [ "$name" = "final" ] && label="Kev fine-tuned, final (step 2500)"
    .venv-wandb/bin/python scripts/record_gif.py --player "kev:http://127.0.0.1:$PORT" --label "$label" --seeds 7000,7001,7002 --out "results/gifs/kev-$name.gif" > /dev/null 2>&1
    out=$(python3 -m game.play --policy model --url "http://127.0.0.1:$PORT" --episodes 50 --seed 8000 2>/dev/null)
    wr=$(echo "$out" | grep -o "win_rate=[0-9.]*" | cut -d= -f2); mt=$(echo "$out" | grep -o "mean_ticks=[0-9.]*" | cut -d= -f2)
    step=$(echo "$name" | grep -o "[0-9]*$"); [ "$name" = "final" ] && step=2500
    # only record if the server did not switch during the run
    now=$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['models'][0]['run'])" 2>/dev/null)
    if [ "$now" = "$cur" ]; then
      echo "{\"step\": $step, \"snapshot\": \"$name\", \"win_rate_50\": $wr, \"mean_ticks\": $mt, \"time\": \"$(date '+%Y-%m-%d %H:%M')\"}" >> results/winrate_curve.jsonl
      echo "$(date +%H:%M) $name: win_rate_50=$wr mean_ticks=$mt (gif + 50 jumps done)"
      done_list="$done_list|$cur|"
      [ "$name" = "final" ] && exit 0
    fi
  fi
  sleep 60
done
