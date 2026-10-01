# Step 08: Dry run

## Goal
Prove `REPRODUCE.md` works from a clean clone and the demo fits 30 minutes.

## Plan
- Fresh clone to a temp dir, follow `REPRODUCE.md` literally (skip training,
  use the saved checkpoint), confirm the table reproduces.
- Time the demo commands end to end.

## Acceptance
- No undocumented step was needed. Timing recorded.


## Result (2026-10-01 06:10)
Fresh `git clone` of the GitHub repo into a temporary directory, following REPRODUCE.md:
- `data.generate` reproduces `train.jsonl` and `eval.jsonl` **byte-identical** to the files the model was trained on.
- 14/14 tests pass; oracle 0.94 over 500 jumps (v4 layouts), as recorded.
- Kev's loader accepts the clone's eval file (2,000 records).
- The release asset `skyfall-0.8b-checkpoint.tgz` downloads, unpacks and serves; scored from the clone: move 0.99+, open 1.00 on 100 states, 20/20 jumps.
- Nothing outside REPRODUCE.md was needed except the Kev clone with the patch (documented in §0).
