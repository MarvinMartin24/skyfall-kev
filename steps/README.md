# Steps

One file per step. Each is written **before** the step runs (Goal, Plan,
Acceptance) and completed **after** (Result, Review, Fixes). If a review
fails acceptance, the step is redone and the file says so.

| # | Step | Status |
|---|---|---|
| 01 | [Game and oracle](01-game.md) | built, pending review |
| 02 | [Dataset and schema check](02-data.md) | built, pending schema check |
| 03 | [Zero-shot baseline](03-baseline.md) | todo |
| 04 | [Fine-tune](04-train.md) | todo |
| 05 | [Evaluate, play, surprise obstacle](05-evaluate.md) | todo |
| 06 | [Latency: decision model vs generation](06-latency.md) | todo |
| 07 | [Docs: reproduce, demo, explain](07-docs.md) | todo |
| 08 | [Dry run](08-dry-run.md) | todo |

Execution environment for every step: this Mac (Apple M2 Max, 64 GB, Metal via
PyTorch `mps`). Colab commands are kept in `colab/train.sh` as the alternative.
