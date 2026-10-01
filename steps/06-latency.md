# Step 06: Latency, decision model vs text generation

## Goal
One honest number pair for the closing slide: milliseconds per decision for
the fine-tuned Kev (one forward pass, both questions) vs the same backbone
asked to *generate* the answer as text and parse it.

## Plan
- `model/generative_baseline.py`: load `Qwen/Qwen3.5-0.8B` (instruct) with
  transformers on `mps`, prompt with the same state and questions, generate a
  short JSON answer, parse. Time 100 decisions. Also record its accuracy so
  the slide can say "slower *and* not better".
- Time 100 Kev decisions through `kev.serve` (reuse `model/eval.py`).

## Acceptance
- `results/latency.md` with mean and p95 for both, same hardware, same states.
- Note any caveat (serving overhead, batch size 1, warm-up excluded).
