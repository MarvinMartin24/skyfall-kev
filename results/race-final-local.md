# The race: 20 seeded jumps per player (seeds 6000–6019)

Latency is per decision (both questions). `seconds_per_jump` is the time the skydiver spends in the air if the sky waits for each decision. `rows_fallen_while_thinking` assumes a row every 100 ms. Kev tokens are priced at Jev list price ($0.042/M input, output free); Gemini at its list price. Accuracy is agreement with the oracle on the states actually visited.

| player | win_rate | move_acc | open_acc | latency_ms_p50 | latency_ms_p95 | seconds_per_jump | rows_fallen_while_thinking | tokens_in_per_decision | tokens_out_per_decision | usd_per_1000_decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 1.00 | 1.00 | 1.00 | 0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0000 |
| kev-released | 0.00 | 0.90 | 0.00 | 56 | 92 | 0.1 | 0.6 | 179 | 84 | 0.0075 |
| kev-final | 1.00 | 1.00 | 1.00 | 56 | 84 | 1.3 | 0.6 | 179 | 70 | 0.0075 |
