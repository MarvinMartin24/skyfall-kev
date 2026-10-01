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
Independent reviewer (read-only, regenerated the files under several hash seeds,
measured overlaps with scripts). Verdict: **ACCEPT**, with two findings we acted on:

| # | Severity | Finding | Action |
|---|---|---|---|
| 1 | risk | Seed disjointness alone leaks 2.1% of eval states into train (initial grids are not uniform); only the explicit `exclude` prevents it. | Already fixed by the `exclude` path; test keeps it fixed. |
| 2 | risk | Exclusion was exact-text only: 9.1% of eval rows had the same grid in train at another altitude, 8% on the same side of 1000 m, i.e. identical decision inputs. | **Fixed**: exclusion keyed on (grid, altitude <= 1000). New test asserts zero key overlap. |
| 3 | ok | Labels are always the oracle's for the recorded state (22,000/22,000 re-derived). | None. |
| 4 | nit | Dodge rows at <= 1000 m had `open=True`, so the open-true share was 16.8%, not 15%. | Resolved by the rule change below (open now implies stay). |
| 5 | design | `open` was equivalent to `altitude <= 1000` in the data (the oracle sidestepped and opened in the same tick), so the "cell below is empty" clause was never tested. | **Rule changed**: open only when the cell directly below is already empty; sidestep first otherwise. Each question is now a pure function of the state. New test requires >= 50 eval rows at low altitude with `open=False`. |
| 6 | nit | An idle-bucket shortfall would silently truncate the file. | Fixed: loop waits on every bucket and asserts the row count. |
| 7 | ok | Byte-for-byte deterministic across hash seeds. | None. |
| 8 | nit | Eval slightly sparser than train (3.59 vs 3.68 obstacles per state) because dedupe pressure scales with split size. | Noted; not material. |
| 9 | nit | `noul` questions carry no `criteria`; Kev's loader does not require it. PLAN wording tightened. | Fixed. |

Follow-up: the first regeneration had only 37 eval rows for the "low altitude,
blocked below" case. Added a fourth bucket `low_blocked` with a 5% quota
(1,000 train / 100 eval rows) so the clause is measurable. Final mix: 40% dodge,
15% open, 5% low_blocked, 40% idle.

Consequence: data regenerated (v3), and the baseline benchmark and fine-tune were
restarted on the new files (about 12 minutes of training lost in total).
