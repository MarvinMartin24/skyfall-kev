# Step 05: Evaluate, play, surprise obstacle

## Goal
Fill the demo table: before vs after, on the eval file and on 200 live jumps,
with and without the never-seen balloon `O`.

## Plan
- Serve the fine-tuned run, run `model/eval.py --n 2000`.
- `game/play.py --policy model --episodes 200` and `--surprise`.
- Calibration: reliability table for `open` (binned P(open) vs empirical).
- `game/play.py --policy model --show` with probability bars for the clip.
- Error analysis: list the states the model gets wrong, look for a pattern
  (edges? obstacle type? altitude boundary at exactly 1000?).

## Acceptance
- `results/table.md` complete; numbers also in `PLAN.md` and `DEMO.md`.
- Written one-paragraph interpretation of the `O` result either way.


## Result (snapshot step-1300, interim; final model pending)
- 300 held-out states: move 0.977, open 0.987, Brier 0.044 / 0.011. Confusion:
  left 52/54, right 92/96, stay 149/150; 2 false opens, 2 missed opens.
- 50 live jumps (seeds 8000–8049): 48 wins. With balloons on the same seeds:
  39 wins (6 balloon hits, 5 other). Oracle on the balloon seeds: 49.
- Symbol probe (one symbol directly below Y, averaged over the 5 columns):
  P(stay) = 1.00 for `.`/`C`, 0.00 for `B`/`P`/`A`/`Z`, 0.11 for `O`.
- GIFs: `results/gifs/kev-step-1300.gif`, `results/gifs/kev-step-1300-balloons.gif`.


## Result (final model, step 2500)
- 100 live jumps (seeds 9000–9099): **94 wins**; oracle 97 on the same seeds.
- Same 100 seeds with balloons: **83 wins**, 14 balloon hits; oracle 97.
- 300 held-out states (`model/confusion.py`): **2 move errors, 0 open errors**.
  Both errors are at exactly 1000 m with a deadly cell below: one "stay"
  where the oracle sidesteps right (the left neighbour is blocked by a P on
  row 1), one "right" at the right edge where the oracle is forced to stay
  (same outcome either way). Full text in `results/final-confusion.txt`.
- Symbol probe (`results/final-symbol-probe.txt`): P(stay) with a single
  symbol under Y: `.` 1.00, `C` 1.00, `B`/`P`/`A` 0.00, `Z` 0.00, `X` 0.00,
  `O` 0.33, `0` 0.61. It applies "any other letter is deadly" to letters it
  never saw, except round glyphs, which it half-reads as empty. The balloon
  was the one unseen symbol that happens to look like a hole.
- Training: 5.8 h wall-clock on the M2 Max (fp32, `mps`, shared GPU), 20,000
  records, 2,500 steps, 3.6M forward tokens, 3.2 GB peak device memory.
  Checkpoint: 62 MB (LoRA adapter + pointer head + tokenizer).
