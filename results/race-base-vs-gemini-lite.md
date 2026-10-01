# The race: 10 seeded jumps per player (seeds 5000–5009)

Latency is per decision (both questions). `seconds_per_jump` is the time the skydiver spends in the air if the sky waits for each decision. `rows_fallen_while_thinking` assumes a row every 100 ms. Kev tokens are priced at Jev list price ($0.042/M input, output free); Gemini at its list price. Accuracy is agreement with the oracle on the states actually visited.

| player | win_rate | move_acc | open_acc | latency_ms_p50 | latency_ms_p95 | seconds_per_jump | rows_fallen_while_thinking | tokens_in_per_decision | tokens_out_per_decision | usd_per_1000_decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 1.00 | 1.00 | 1.00 | 0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0000 |
| kev-base | 0.00 | 0.80 | 0.00 | 280 | 474 | 0.3 | 3.0 | 179 | 84 | 0.0075 |
| gemini:gemini-flash-lite-latest | 0.00 | 0.37 | 1.00 | 745 | 914 | 3.1 | 7.5 | 198 | 14 | 0.0704 |
