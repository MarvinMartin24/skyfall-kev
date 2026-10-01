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
- `python -m data.generate --train 20000 --eval 2000` (seconds). Mix in both files:
  move = left 0.16 / right 0.29 / stay 0.55, open_true = 0.17.
- **Bug found by `tests/test_data.py`:** 42 eval states also appeared in train.
  Dedupe was per split only. Fixed: eval is generated first and its states are
  passed to the train generator as `exclude`. Test now passes.
- `tests/test_data.py` also re-parses every rendered state back into a `World`
  (`game/parse.py`) and recomputes the oracle: all 22,000 labels match.
- Kev's own loader (`kev.data.load_records`) accepts 20,000 + 2,000 records,
  zero rejected. `_meta` auto-filled with `source=custom`. Kev commit `1d77363`.
- Kev installed locally with `uv sync --extra serve`: torch 2.8.0, transformers 5.17.0, `mps` available.

## Review
_(filled after execution)_
