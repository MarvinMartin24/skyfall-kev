# Explainer videos (Manim Community)

Two videos, both built from this repo's own results (nothing is typed in by hand except the illustrative logits
on the architecture slides, which the slides label as made up).

| file | video | length at PACE=1.5 |
|---|---|---|
| `kev_architecture.py` | **KevDeepDive**: how a Kev decision model works, text to loss and calibration (token layout, hidden states, the pointer head with its shared `W_q`/`W_k`, the matrix multiplications, softmax, cross-entropy, how T is fitted) | about 4 min |
| `skyfall_demo.py` | **SkyfallDemo**: the game, oracle, dataset, metrics, Kev on Skyfall, before training, launching `kev.train`, learning curve, released vs fine-tuned vs Jev vs Gemini (table, side-by-side GIFs, latency, cost), the balloon test, takeaways | about 5 min |
| `common.py` | shared helpers and the `Slides` scene base class (title, caption, `PACE`) | |
| `compose_gifs.py` | overlays `results/gifs/*.gif` onto the render so they actually play | |
| `render.sh` | renders both at 1080p and composes the GIFs into `out/` | |

## Setup (macOS)

```bash
brew install pkg-config cairo pango ffmpeg      # pycairo needs cairo and pkg-config to build
python3.11 -m venv .venv-video && source .venv-video/bin/activate
pip install manim numpy                         # use a venv of its own: manim pulls numpy 2.x
```

## Render

```bash
./video/render.sh                    # both, 1080p, PACE=1.5   -> video/out/KevDeepDive.mp4, SkyfallDemo_final.mp4
PACE=2 ./video/render.sh skyfall     # only Skyfall, slower (more room to talk)
QUALITY=low ./video/render.sh        # 480p preview
```

`PACE` multiplies every animation and every pause (default 1.5; 1 is the original fast cut). GIF playback speeds
are divided by PACE so each clip still fills its longer slide, and the on-screen captions state the real speed.

## Why GIFs are composed afterwards

Manim has no animated-image mobject: a GIF added to a scene shows as a single still frame. `skyfall_demo.py` leaves a
dark placeholder at each GIF position and writes when it is on screen to `out/gif_slots.json`;
`compose_gifs.py` then overlays the real GIF with ffmpeg. If you change a slide's timing, re-render and
re-compose; don't edit `gif_slots.json` by hand.

## Where each number comes from

The Skyfall slides read `results/learning_curve.jsonl` and `results/winrate_curve.jsonl` directly. The other
figures are typed into the slide code from these files, so update both together when results change:

| on the slides | source |
|---|---|
| released Kev 0.554 / 0.467, Brier 0.57 | `results/baseline.json` |
| fine-tuned 0.9975 / 0.999, ECE 0.0017 | `results/final_benchmark.json` |
| 56 ms, 20/20 | `results/race-final-local.json` |
| Jev 0.343 / 0.949, 3/20, 176 ms | `results/jev-eval-2000.json`, `results/jev-race-6000.json` |
| Gemini bare 0/10, 745 ms | `results/race-base-vs-gemini-lite.json` |
| Gemini + CoT 15/20, 0.86 / 0.99, 809 ms | `results/race-gemini-cot.json` |
| 11.3M of ~800M, 2,500 steps, 5.8 h, 3.2 GB | `results/training_metrics.json`, `STORY.md` |
| balloon 83/100, probe | `results/final-play.txt`, `results/final-symbol-probe.txt` |
