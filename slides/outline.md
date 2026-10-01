# Slides: outline with speaker notes (numbers from results/)

### 1. Title: Teaching a decision model to skydive, on one laptop
"Thirty minutes. One game, one rule, one fine-tune, one table. Everything is in a repo you can clone, with the checkpoint as a release."

### 2. Generator vs decision model
Left: prompt → decode tokens → parse. Right: state + typed questions → one forward pass → probabilities.
"Same transformer, different readout. A generator gives you a string to trust and parse after N passes. A decision model gives you a probability per option after one. TypeSafe sells this as Jev; Kev is the open clone we train."

### 3. Three question types
`choice` → probability per option; `noul` → P(true); `score` → distribution over ordered levels.
"Options are text you write at call time. Ten questions in one call are answered independently."

### 4. The game
5x6 grid, Y on row 1, B/P/A deadly, C harmless, altitude 3000 → 0, open at ≤ 1000 m with a clear cell below.
Live: `skyfall.py --policy oracle --fps 4`. "A state is 43 tokens."

### 5. The rule
`game/oracle.py`, 15 lines. "The teacher. 94% of jumps. The model never sees the code."

### 6. The data
One JSONL line; 20,000 / 2,000; mix 40% dodge, 15% open, 5% low-and-blocked, 40% idle.
"Rollouts with 30% random moves. Two tests: every label re-derived from the grid; no eval grid in train. The first generator leaked 42 eval states and had an open label that was altitude alone. Reviews caught both."

### 7. Before: released Kev-0.8B, zero-shot
2,000 held-out states: move 0.554 (= always stay), open 0.467 (worse than always no), Brier 0.57. Live: 0 of 200 jumps.
"It has never seen a grid. Three bars at a third each; P(open) ≈ 0.5 everywhere; opens on tick one."

### 8. Inside Kev
(a) `<state> … <q> instr <opt> o </opt> … <decide>`; (b) block-causal mask; (c) `logits = (K·h_opts)·(Q·h_decide)/√256`.
"Five reused Qwen special tokens. The head is one attention query over the options, so the option set can change per call. LoRA rank 16 on every linear layer: 11.3M of ~800M parameters, warm-started from the released checkpoint."

### 9. The training command
```
kev.train --data train.jsonl --init_from jaredpalmer/kev-0.8b --base Qwen/Qwen3.5-0.8B-Base \
  --epochs 1 --lr 2e-5 --batch 4 --accum 2 --device mps --dtype fp32 --lora_snapshot_every 100
```
5.8 h on an M2 Max (fp32, two kernels in fallback mode); the Kev authors report ~20 min for a comparable run on an H100 (not measured here). 3.2 GB peak device memory. Checkpoint 62 MB.
"Cross-entropy against the oracle's label. Kev's default augmentation on: options shuffled, a third of move questions get a none-or-distractor option. Imitation of a rule, not reinforcement learning."

### 10. Watching it learn
`results/learning_curve.png` and `results/gifs/montage.gif`.
Steps 100 / 300 / 700 / 1000 / 1300 / 2500: move 0.51 / 0.57 / 0.87 / 0.96 / 0.98 / 0.99; open 0.89 / 0.93 / 0.98 / 0.98 / 0.99 / 1.00; live wins on 50 seeds: – / 18% / 24% / 76% / 96% / 96% (and 94 of 100 on a second seed set).
"Altitude in 100 steps. Nothing on the grid until warm-up ends at step 250. At 700 it dodges but picks the wrong side: 16 of 96 'right' cases answered 'left'. By 1,300, parity with the rule."

### 11. After
Live: `skyfall.py --policy model --fps 2`. Table: released vs final vs oracle (move 0.55 / 0.993 / 1; open 0.47 / 1.00 / 1; wins 0 / 94 / 97 per 100).
"Two move errors in 300 states, both at exactly 1,000 m, the one altitude where the two rules interact."

### 12. The balloon
`--surprise`: 83 of 100 (oracle 97). Probe: P(stay) with one symbol below = 0.00 for B, P, A, Z, X; 0.33 for O; 0.61 for the digit 0.
"It applies 'any other letter is deadly' to letters it never saw. It also learned that round hollow glyphs look like empty cells. We picked the one surprise symbol that looks like a hole."

### 13. Why this architecture exists, and what it is not
Latency per decision (same Mac, same states): fine-tuned Kev vs Gemini flash-lite vs Gemini with thinking; tokens and $ per 1,000 decisions; "rows fallen while thinking". Numbers: `results/table.md`.
Caveats: a tiny CNN solves this grid; imitation not RL; Kev ≠ Jev; option-order sensitivity exists.
"One forward pass versus a decoding loop. The honest caveats are on the slide so nobody has to raise them."

### Backup
FAQ.md; EXPLAINER.md §2–4; `results/final-confusion.txt` (the two remaining errors); the free-tier story (44% of Gemini decisions needed a retry).
