# Steps

One file per step. Each is written **before** the step runs (Goal, Plan,
Acceptance) and completed **after** (Result, Review, Fixes). If a review
fails acceptance, the step is redone and the file says so.

| # | Step | Status |
|---|---|---|
| 01 | [Game and oracle](01-game.md) | **done**, reviewed (accept), fixes applied |
| 02 | [Dataset and schema check](02-data.md) | **done**, reviewed (accept), v4 with clouds |
| 03 | [Zero-shot baseline](03-baseline.md) | **done**: move 0.55, open 0.47, Brier 0.57 |
| 04 | [Fine-tune](04-train.md) | **done**: 5.8 h on M2 Max, final move 0.993 / open 1.000 |
| 05 | [Evaluate, play, surprise obstacle](05-evaluate.md) | **done**: 94/100 jumps, 83/100 with balloons |
| 06 | [The race: latency, tokens, cost](06-latency.md) | **done** locally (60 ms/decision); Gemini part paused by request |
| 07 | [Docs: reproduce, demo, explain](07-docs.md) | drafted with final numbers; review pending |
| 08 | [Dry run](08-dry-run.md) | **done**: fresh clone reproduces data byte-for-byte; release checkpoint serves |

Execution environment for every step: this Mac (Apple M2 Max, 64 GB, Metal via
PyTorch `mps`). Colab commands are kept in `colab/train.sh` as the alternative.
