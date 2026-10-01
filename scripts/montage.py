"""One GIF: the same jump (seed 7000) played by the released model and by snapshots at increasing steps.
Takes the first jump out of each existing per-snapshot GIF (cut at its outcome frame, which is the 1.8 s one).
.venv-wandb/bin/python scripts/montage.py --parts kev-released,kev-step-300,kev-step-700,kev-step-1000,kev-step-1300,kev-final --out results/gifs/montage.gif"""
import argparse, os
from PIL import Image, ImageSequence

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def first_jump(path):
    im = Image.open(path); frames, durs = [], []
    for fr in ImageSequence.Iterator(im):
        d = fr.info.get("duration", 333); frames.append(fr.convert("RGB").copy()); durs.append(d)
        if d >= 1700:
            break
    return frames, durs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="kev-released,kev-step-300,kev-step-700,kev-step-1000,kev-step-1300,kev-final")
    ap.add_argument("--out", default="results/gifs/montage.gif"); a = ap.parse_args()
    frames, durs = [], []
    for name in a.parts.split(","):
        f, d = first_jump(os.path.join(HERE, "results", "gifs", name + ".gif")); frames += f; durs += d
        print(name, len(f), "frames")
    frames[0].save(os.path.join(HERE, a.out), save_all=True, append_images=frames[1:], duration=durs, loop=0)
    print("wrote", a.out, len(frames), "frames")


if __name__ == "__main__":
    main()
