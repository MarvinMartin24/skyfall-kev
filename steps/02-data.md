# Step 02: Dataset and schema check

## Goal
Training and evaluation JSONL generated entirely by code, labelled by the
oracle, balanced, and proven to load in Kev's own loader without edits.

## Plan
- `data/questions.py`: the two typed questions (`move` choice, `open` noul).
- `data/generate.py`: eps-random rollouts of the oracle (so bad positions are
  covered), oracle labels, dedupe on state text, fixed mix 45% dodge / 15% open
  / 40% idle. Train seed and eval seed are disjoint.
- Install the Kev repo locally and call `kev.data.load_records` on both files.
- Token-count a state with the Qwen tokenizer to confirm it is far under
  Kev's `--max_state`.

## Acceptance
- 20,000 train / 2,000 eval rows, no duplicate state text across the two files.
- `load_records` accepts every row, zero rejections.
- Label distribution printed and recorded here.
- Review: a second reader checks the generator cannot leak eval states into
  train and that labels are always the oracle's (never the eps-random action).

## Result
_(filled after execution)_

## Review
_(filled after execution)_
