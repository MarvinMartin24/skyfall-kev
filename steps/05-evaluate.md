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
