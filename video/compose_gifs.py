"""Overlay the real GIFs (full speed, looping) onto the Manim render, using out/gif_slots.json.

    python3 video/compose_gifs.py <manim_video.mp4> <output.mp4>

Manim cannot play GIFs (they would render as still images), so skyfall_demo.py leaves a dark placeholder
and records when each slot is on screen; this script then overlays results/gifs/<name>.gif with ffmpeg.
Needs ffmpeg on PATH.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GIFS = HERE.parent / "results" / "gifs"
SLOTS = HERE / "out" / "gif_slots.json"


def probe(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    d = json.loads(out)
    return d["streams"][0]["width"], d["streams"][0]["height"], float(d["format"]["duration"])


def main(src, dst):
    meta = json.loads(SLOTS.read_text())
    W, H, dur = probe(src)
    ppu = W / meta["frame_width"]                      # pixels per Manim unit
    slots = [s for s in meta["slots"] if s["t1"] is not None]
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    chains, last = [], "0:v"
    for k, s in enumerate(slots, start=1):
        cmd += ["-stream_loop", "-1", "-i", str(GIFS / s["gif"])]
        wpx = int(round(s["w"] * ppu / 2) * 2)
        hpx = int(round(wpx * 420 / 980 / 2) * 2)
        x = int(round(W / 2 + s["cx"] * ppu - wpx / 2))
        y = int(round(H / 2 - s["cy"] * ppu - hpx / 2))
        chains.append(f"[{k}:v]setpts=(PTS-STARTPTS)/{s['speed']}+{s['t0']:.3f}/TB,scale={wpx}:{hpx}:flags=neighbor,format=yuv420p[g{k}]")
        chains.append(f"[{last}][g{k}]overlay={x}:{y}:enable='between(t,{s['t0']:.3f},{s['t1']:.3f})':eof_action=pass[v{k}]")
        last = f"v{k}"
    cmd += ["-filter_complex", ";".join(chains), "-map", f"[{last}]", "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-profile:v", "high", "-crf", os.environ.get("CRF", "16"), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)
    print("wrote", dst, f"({len(slots)} gifs overlaid)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
