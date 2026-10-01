# Gemini probes (2 calls each, seed-1000 state, free tier, 2026-10-01 ~00:00)

Recorded from console output of a throwaway probe; not a benchmark. Latency per call; "think" = thoughtsTokenCount.

| model | thinkingConfig | call 1 | call 2 |
|---|---|---|---|
| gemini-3.8-flash | thinkingLevel minimal | HTTP 400 (not supported) | HTTP 400 |
| gemini-3.8-flash | thinkingLevel low | 28,984 ms, 2 retries (503), failed | 26,974 ms, think=385, ok |
| gemini-3.5-flash | thinkingLevel minimal | 8,572 ms, think=0 | 8,182 ms, think=0 |
| gemini-3.1-flash-lite | thinkingLevel minimal | 2,284 ms | 1,573 ms |
| gemini-3.1-flash-lite | thinkingBudget 0 | 2,136 ms | 2,902 ms |
| gemini-flash-lite-latest (served as 3.5-flash-lite) | none | 774 ms | 782 ms |

Earlier single calls on gemini-3.8-flash with thinkingBudget 0 (ignored by the model): 4.3 s / 5.7 s / 2.9 s per call, 290–470 thinking tokens, up to 6 retries on 503.
