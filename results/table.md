# Results

## Before and after (held-out states; live jumps on seeds 8000–8049)

| model | move acc | open acc | move Brier | open Brier | win rate |
|---|---|---|---|---|---|
| released Kev-0.8B, zero-shot | 0.554 | 0.467 | 0.626 | 0.506 | 0.00 (0/200) |
| fine-tuned, step 2500 | 0.993 | 1.000 | 0.013 | 0.000 | 0.96 (50 jumps) |
| oracle (the rule) | 1.000 | 1.000 | 0 | 0 | 0.94 (500 jumps) |

Zero-shot numbers: full `kev.benchmark` on 2,000 states. Fine-tuned: 300 states via `model/eval.py` (final: see `runs/ft-eval/report.json` if present).

## Learning curve

| step | records | move acc | open acc | move Brier | win rate (50 jumps) |
|---|---|---|---|---|---|
| 100 | 800 | 0.505 | 0.885 | 0.613 | |
| 200 | 1,600 | 0.497 | 0.927 | 0.612 | |
| 300 | 2,400 | 0.567 | 0.927 | 0.531 | 0.18 |
| 400 | 3,200 | 0.690 | 0.943 | 0.369 | |
| 500 | 4,000 | 0.773 | 0.943 | 0.279 | 0.42 |
| 600 | 4,800 | 0.860 | 0.953 | 0.209 | 0.38 |
| 700 | 5,600 | 0.873 | 0.980 | 0.214 | 0.24 |
| 800 | 6,400 | 0.923 | 0.977 | 0.129 | 0.58 |
| 900 | 7,200 | 0.897 | 0.970 | 0.177 | 0.74 |
| 1000 | 8,000 | 0.960 | 0.983 | 0.065 | 0.76 |
| 1100 | 8,800 | 0.973 | 0.977 | 0.053 | 0.82 |
| 1200 | 9,600 | 0.970 | 0.990 | 0.056 | 0.86 |
| 1300 | 10,400 | 0.977 | 0.987 | 0.044 | 0.96 |
| 1400 | 11,200 | 0.987 | 0.987 | 0.026 | 0.96 |
| 1500 | 12,000 | 0.987 | 0.987 | 0.016 | 0.98 |
| 1600 | 12,800 | 0.990 | 0.997 | 0.016 | 0.94 |
| 1700 | 13,600 | 0.990 | 0.997 | 0.012 | 0.96 |
| 1800 | 14,400 | 0.990 | 0.993 | 0.021 | 0.96 |
| 1900 | 15,200 | 0.987 | 0.997 | 0.019 | 0.96 |
| 2000 | 16,000 | 0.993 | 0.997 | 0.014 | 0.96 |
| 2100 | 16,800 | 0.993 | 0.997 | 0.012 | 0.96 |
| 2200 | 17,600 | 0.993 | 1.000 | 0.013 | 0.96 |
| 2300 | 18,400 | 0.993 | 1.000 | 0.013 | 0.96 |
| 2400 | 19,200 | 0.993 | 1.000 | 0.012 | 0.96 |
| 2500 | 20,000 | 0.993 | 1.000 | 0.013 | 0.96 |

![learning curve](learning_curve.png)

## Final model, full benchmark (`kev.benchmark`, 2,000 states / 4,000 questions)

| | move | open | overall |
|---|---|---|---|
| accuracy | 0.9975 | 0.9990 | 0.9982 |
| Brier | 0.0049 | 0.0020 | 0.0034 |
| NLL | 0.0091 | 0.0072 | |
| ECE (overall) | | | 0.0017 |

Released model on the same 2,000 states: move 0.554 / Brier 0.626, open 0.467 / Brier 0.506, ECE 0.117.

## Latency per decision (both questions), Kev served by MLX on the M2 Max, GPU otherwise idle, 500 states

| model | mean ms | move acc | open acc |
|---|---|---|---|
| final | 60 | 0.996 | 1.000 |
| released | 62 | 0.524 | 0.454 |

Gemini numbers (free tier, bare and chain-of-thought prompts) are in the race files below and in STORY.md §7.

## race-base-vs-gemini-lite.md

Latency is per decision (both questions). `seconds_per_jump` is the time the skydiver spends in the air if the sky waits for each decision. `rows_fallen_while_thinking` assumes a row every 100 ms. Kev tokens are priced at Jev list price ($0.042/M input, output free); Gemini at its list price. Accuracy is agreement with the oracle on the states actually visited.

| player | win_rate | move_acc | open_acc | latency_ms_p50 | latency_ms_p95 | seconds_per_jump | rows_fallen_while_thinking | tokens_in_per_decision | tokens_out_per_decision | usd_per_1000_decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 1.00 | 1.00 | 1.00 | 0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0000 |
| kev-base | 0.00 | 0.80 | 0.00 | 280 | 474 | 0.3 | 3.0 | 179 | 84 | 0.0075 |
| gemini:gemini-flash-lite-latest | 0.00 | 0.37 | 1.00 | 745 | 914 | 3.1 | 7.5 | 198 | 14 | 0.0704 |


## race-final-local.md

Latency is per decision (both questions). `seconds_per_jump` is the time the skydiver spends in the air if the sky waits for each decision. `rows_fallen_while_thinking` assumes a row every 100 ms. Kev tokens are priced at Jev list price ($0.042/M input, output free); Gemini at its list price. Accuracy is agreement with the oracle on the states actually visited.

| player | win_rate | move_acc | open_acc | latency_ms_p50 | latency_ms_p95 | seconds_per_jump | rows_fallen_while_thinking | tokens_in_per_decision | tokens_out_per_decision | usd_per_1000_decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 1.00 | 1.00 | 1.00 | 0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0000 |
| kev-released | 0.00 | 0.90 | 0.00 | 56 | 92 | 0.1 | 0.6 | 179 | 84 | 0.0075 |
| kev-final | 1.00 | 1.00 | 1.00 | 56 | 84 | 1.3 | 0.6 | 179 | 70 | 0.0075 |


## Gemini prompt variants (gemini-flash-lite-latest, thinking none, 60 held-out states)

| prompt | move acc | open acc | p50 ms |
|---|---|---|---|
| minimal | 0.33 | 0.82 | 719 |
| rules | 0.43 | 0.93 | 694 |
| procedure | 0.45 | 0.92 | 741 |
| procedure (gemini-3.1-flash-lite, thinking minimal) | 0.37 | 0.97 | 1029 |
| cot (gemini-flash-lite-latest, thinking none) | 0.70 | 1.00 | 896 |
