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
- Gemini, first round (free tier): bare prompt 745 ms p50 and 0/10 jumps
  (`results/race-base-vs-gemini-lite.md`; the race args record no prompt field, bench.py's default
  then was "procedure"). Prompt study on 60 states (`results/llm_prompt_eval.json`,
  `results/llm_eval_lite_cot.json`): minimal 0.33, rules 0.43, procedure 0.45, chain-of-thought
  0.70 move accuracy at ~900 ms; gemini-3.8-flash with thinking "low" ~27 s per decision (probe only,
  one successful call).
- Gemini, chain-of-thought race, now run (Oct 6, `results/race-gemini-cot.md`, 20 jumps, seeds
  6000-6019, `gemini-flash-lite-latest` served as `gemini-3.5-flash-lite`, thinking none):
  **15 of 20 wins (0.75)**, move agreement 0.86, open 0.99, p50 809 ms, p95 1,138 ms, 627 tokens in
  and 73 out per decision, $0.266 per 1,000 decisions at list price, 0.51 retries per decision
  (so wall time per decision was 3.7 s). The earlier 0/10 was the bare prompt and does not
  describe chain of thought. Kev final on the same seeds: 20/20 at 56 ms.
- Jev, hosted (`scripts/jev_test.py`, `jev-1.13.0`, same questions as Kev, zero-shot, no prompt
  tuning): 2,000 held-out states: move 0.343, open 0.9485, Brier 0.353 overall, ECE 0.062; median
  176 ms per decision including the network; 3 of 20 jumps won on seeds 6000-6019.
  See `results/jev-eval-2000.json`, `results/jev-race-6000.json`.
- Not measured: the 3.5/3.8 thinking variants as full races; any Gemini or Jev prompt tuning.
