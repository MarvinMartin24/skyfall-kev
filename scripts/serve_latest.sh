#!/usr/bin/env bash
# Keep kev.serve on PORT pointed at the newest LoRA snapshot of RUN (or the final run dir once it exists).
# Restarts the server when a newer snapshot appears; model/client.py waits through the restart.
# Usage: KEV=/path/to/kev bash scripts/serve_latest.sh [run-name] [port]
set -u
SKY="$(cd "$(dirname "$0")/.." && pwd)"
KEV="${KEV:-$SKY/../kev}"
RUN="${1:-skyfall-0.8b}"; PORT="${2:-8010}"
current=""
while true; do
  if [ -f "$SKY/runs/$RUN/head.pt" ]; then latest="$SKY/runs/$RUN"
  else latest="$(ls -d "$SKY/runs/$RUN-snapshots"/step-* 2>/dev/null | awk -F'step-' '{print $NF" "$0}' | sort -n | tail -1 | cut -d' ' -f2-)"; fi   # numeric by step (text sort put step-900 after step-1300)
  if [ -n "$latest" ] && [ "$latest" != "$current" ] && [ -f "$latest/head.pt" ]; then
    echo "$(date +%H:%M:%S) serving $latest on :$PORT"
    pkill -f "kev.serve.*--port $PORT" 2>/dev/null; sleep 2
    (cd "$KEV" && nohup uv run --no-sync python -m kev.serve --run "$latest" --port "$PORT" >> "$SKY/runs/serve-latest.log" 2>&1 &)
    current="$latest"
    [ -f "$SKY/runs/$RUN/head.pt" ] && { echo "final checkpoint served; exiting watcher"; exit 0; }
  fi
  sleep 30
done
