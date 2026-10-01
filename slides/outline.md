# Slides: outline with speaker notes

Placeholders in `{{ }}` are filled from `results/table.md` and `results/latency.md`.
Target: 12 slides, 30 minutes including questions.

---

### 1. Title
**Teaching a decision model to skydive, on one laptop**
Notes: "Thirty minutes. One game, one rule, one fine-tune, one table. Everything is in a repo you can clone."

### 2. Generator vs decision model
Left: prompt → decode token by token → parse string. Right: state + typed questions → one forward pass → probabilities.
Notes: "Same transformer. The difference is what you read out. A generator gives you a string you have to trust and parse, after N forward passes. A decision model gives you a probability per option after one. TypeSafe calls this System One and sells it as Jev. Kev is the open reimplementation; that is what we train today."

### 3. The three question types
`choice` → probability per option. `noul` → P(true). `score` → distribution over ordered levels.
Notes: "Options are text you write at call time. Ten questions in one call are answered independently, in one pass."

### 4. The game
The 5x6 grid, Y at row 1, obstacles B P A scrolling up, altitude 3000 → 0, open at ≤ 1000 m.
Notes: live `game.play --policy oracle --show`. "A state is 43 tokens. Obstacles come at you one row per tick."

### 5. The rule (the oracle)
The 15 lines of `game/oracle.py`, verbatim.
Notes: "This is the teacher. It wins 95% of jumps. The model never sees the code, only states and the oracle's answers."

### 6. The data
One JSONL line, pretty-printed: state, `move` choice with three options and label, `open` noul with label.
Numbers: 20,000 train / 2,000 eval, 45% dodge, 15% open, 40% idle; eval states excluded from train; every label re-derived by a test.
Notes: "No gameplay was recorded. Rollouts with 30% random moves so the model also sees bad positions. The first generator leaked 42 eval states into train; a test caught it. Say that: it is the kind of thing an audience trusts you more for."

### 7. Before: released Kev-0.8B, zero-shot
Table row: move acc {{base_move_acc}}, open acc {{base_open_acc}}, Brier {{base_brier}}, win rate {{base_win}}.
Clip: base model dying.
Notes: "Same questions, same states, no training. It has never seen a grid. Chance on `move` is 0.33; on `open` always-false gets 0.83, so read the Brier, not the accuracy."

### 8. Inside Kev (one slide, three pictures)
(a) Packed sequence: `<state> … <q> instr <opt> o </opt> … <decide>` per question.
(b) Block-causal mask: branches see the state and themselves, never each other.
(c) Pointer head: `logits = (K·h_opts) · (Q·h_decide) / √256`, softmax.
Notes: "Five rarely-used Qwen tokens reused as delimiters. The head is one attention query, the decision, over K keys, the options. That is why the option set can change per call: the head scores whatever option text you wrote. LoRA rank 16 on every linear layer, 11.3M trainable parameters out of 800M."

### 9. The training command
```
kev.train --data train.jsonl --init_from jaredpalmer/kev-0.8b --base Qwen/Qwen3.5-0.8B-Base \
  --epochs 1 --lr 2e-5 --batch 4 --accum 2 --device mps --dtype fp32
```
{{train_wallclock}} on an M2 Max; {{colab_wallclock}} on a Colab L4 in bf16.
Notes: "Delta mode: warm-start the LoRA and head from the released checkpoint, then add our domain. Cross-entropy against the oracle's label. No reinforcement learning anywhere; this is imitation of a rule."

### 10. After: live
Live `game.play --policy model --show` with probability bars, then `--episodes 200`.
Table: before vs after vs oracle. {{table}}
Notes: "Watch P(stay) collapse when something is under Y. Watch P(open) jump below 1000 m with a clear cell. Then the numbers over 2,000 held-out states and 200 jumps."

### 11. The balloon
`--surprise` adds `O`, never in training. Win rate: oracle 0.93, fine-tuned {{ft_win_surprise}}.
Notes: "Did it learn 'non-empty cell below' or the three letters? {{balloon_interpretation}}"

### 12. Why this architecture exists, and what it is not
Latency per decision on the same machine and states: Kev {{kev_ms}} ms vs Qwen3.5-0.8B generating JSON {{gen_ms}} ms (and {{gen_move_acc}} accuracy).
Caveats: a tiny CNN solves this grid; imitation not RL; Kev ≠ Jev; option-order sensitivity exists.
Notes: "One forward pass versus a decoding loop plus parsing. The honest caveats are on the slide so nobody has to raise them for you."

### Backup
- FAQ.md answers.
- EXPLAINER.md §2–4 diagrams.
- Error analysis: the states the fine-tune still gets wrong ({{error_pattern}}).
