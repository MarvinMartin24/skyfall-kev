# Step 06: The race. Latency, tokens and cost: decision model vs LLM calls

## Goal
The closing slide: the same seeded jumps played by the oracle, released Kev,
fine-tuned Kev and a hosted LLM (Gemini), with per decision latency, tokens,
cost, agreement with the oracle, and "time as gravity": how long the skydiver
is in the air if the sky waits for each decision.

Tool: `model/bench.py` (`results/race.md`). Players: `oracle`, `kev-<label>:<url>`,
`gemini[:<model>]`. Kev tokens priced at Jev's list price ($0.042/M input, output
free) or $0 local; Gemini at list price. The Gemini key is read from the
environment or `.env`, never from chat.

Metrics per player: win rate, move/open agreement with the oracle, latency p50/p95
per decision, seconds per jump, rows fallen while thinking (at 100 ms per row),
tokens in/out per decision, USD per 1,000 decisions and per jump, parse failures.

## Plan
- `model/generative_baseline.py`: load `Qwen/Qwen3.5-0.8B` (instruct) with
  transformers on `mps`, prompt with the same state and questions, generate a
  short JSON answer, parse. Time 100 decisions. Also record its accuracy so
  the slide can say "slower *and* not better".
- Time 100 Kev decisions through `kev.serve` (reuse `model/eval.py`).

## Acceptance
- `results/latency.md` with mean and p95 for both, same hardware, same states. _(superseded: the numbers live in `results/table.md` and `results/race-*.md`, see Result below)_
- Note any caveat (serving overhead, batch size 1, warm-up excluded).


## Result
- Kev through its MLX server on the M2 Max, GPU idle, 500 held-out states:
  final model 60 ms mean per decision, released 62 ms.
- Local race (`results/race-final-local.md`, 20 jumps): final p50 56 ms, p95 84 ms,
  1.3 s per jump, 20/20 wins; released 0/20.
- Gemini (free tier, paused by request before the chain-of-thought race finished):
  bare prompt 745 ms p50 and 0/10 jumps (`results/race-base-vs-gemini-lite.md`);
  prompt study on 60 states (`results/llm_prompt_eval.json`, `results/llm_eval_lite_cot.json`):
  minimal 0.33, rules 0.43, procedure 0.45, chain-of-thought 0.70 move accuracy at ~900 ms;
  gemini-3.8-flash with thinking "low" ~27 s per decision (probe only).
- Open: the chain-of-thought race and the 3.5/3.8 thinking variants, if Gemini is re-enabled.
