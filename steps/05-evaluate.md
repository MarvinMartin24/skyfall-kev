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
