#!/usr/bin/env bash
# Render the two explainer videos at 1080p / 30 fps into video/out/.
#
#   ./video/render.sh                  # both videos, PACE=1.5 (default)
#   PACE=2 ./video/render.sh skyfall   # only the Skyfall video, slower
#   QUALITY=low ./video/render.sh      # 480p preview of everything
#   QUALITY=4k  ./video/render.sh      # 3840x2160 @ 30 fps (about 4x the render time of 1080p)
#   FPS=60 QUALITY=4k ./video/render.sh   # smoother animation; the GIFs themselves are ~3 fps
#
# PACE multiplies every animation and wait; 1 = the original fast cut. Needs manim, ffmpeg and the
# GIFs in results/gifs/ (see REPRODUCE.md). Run from anywhere.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PACE="${PACE:-1.5}"
MANIM="${MANIM:-manim}"      # e.g. MANIM=/path/to/venv/bin/manim if it is not on PATH
command -v "$MANIM" >/dev/null || { echo "manim not found; activate its venv or set MANIM=/path/to/manim (see video/README.md)"; exit 1; }
WHAT="${1:-all}"
FPS="${FPS:-30}"
case "${QUALITY:-high}" in
  low)  RES="-ql"; SUB=480p15 ;;
  4k)   RES="-r 3840,2160 --fps $FPS"; SUB=2160p$FPS ;;
  *)    RES="-r 1920,1080 --fps $FPS"; SUB=1080p$FPS ;;
esac
export CRF="${CRF:-16}"
MEDIA="$HERE/out/media"

if [ "$WHAT" = "all" ] || [ "$WHAT" = "skyfall" ]; then
  "$MANIM" $RES --media_dir "$MEDIA" skyfall_demo.py SkyfallDemo
  python3 compose_gifs.py "$MEDIA/videos/skyfall_demo/$SUB/SkyfallDemo.mp4" "$HERE/out/SkyfallDemo_final.mp4"
fi
if [ "$WHAT" = "all" ] || [ "$WHAT" = "kev" ]; then
  "$MANIM" $RES --media_dir "$MEDIA" kev_architecture.py KevDeepDive
  cp "$MEDIA/videos/kev_architecture/$SUB/KevDeepDive.mp4" "$HERE/out/KevDeepDive.mp4"
fi
echo "done -> $HERE/out/"
