# Hard questions you will get, with short answers

**Isn't this just a classifier with extra steps?**
A classifier has a fixed output layer: N labels, decided at training time.
Here the options are text you write at call time, and the model scores them
through a pointer head (dot product between the decision vector and each
option's representation). Same request shape works for 2 options or 20, and
for questions it has never seen. Fine-tuning taught it the domain, not a
label set.

**Why not a CNN on the grid?**
For this grid, a CNN would win on compute. The point of the demo is the
request shape and the training loop, which transfer unchanged to tickets,
logs, agent traces and policies. The grid was chosen because the rule is
checkable and the base model cannot already do it.

**How is this different from reading next-token logprobs for "A", "B", "C"?**
Options are arbitrary spans with descriptions, each option's hidden state has
read its own description, questions are isolated by a block-causal mask, the
readout was trained for this, and there is a fitted temperature for
calibration. Logprob-on-a-letter gets you none of that and is notoriously
sensitive to the letter.

**Why does one forward pass answer several questions?**
The state is encoded once; each question is a branch that attends to the
state and to itself only. Branches never see each other, so their answers are
independent and can be computed in the same pass, or as rows on a cached
state prefix when serving.

**What exactly is trained?**
LoRA adapters (rank 16) on every linear layer of the frozen 0.8B backbone and
the pointer head (two 1024 to 256 linear maps). We warm-start both from the
released checkpoint ("delta mode"). Loss is cross-entropy per question.

**Did it learn the rule or memorise states?**
Eval grids are excluded from training by construction and it makes 2 move
errors in 300 of them. The balloon test adds a symbol never seen in training:
83 of 100 jumps, and a probe shows it treats `Z` and `X`, also never seen, as
deadly with probability 1.00. It learned the rule; it also learned that a
round hollow glyph looks like an empty cell.

**Is this reinforcement learning?**
No. Supervised imitation of a 15-line oracle. The model never plays during
training. A "let it play, collect its deaths, relabel, retrain" loop (DAgger)
would be the next step and is still supervised.

**How much data did it need?**
20,000 records, one epoch. The altitude rule was learned by record 800, the
grid by record 10,400 (step 1,300), and the last 1,200 steps only sharpened
confidence (Brier 0.04 → 0.01). A smaller run would likely do; we did not
ablate it. The Kev repo's guidance: under about 400 records gains are noise.

**Why are probabilities better than an answer?**
You can threshold them, log them, and measure calibration. We report the
Brier score next to accuracy for that reason.

**What is the difference between Jev and Kev?**
Jev: TypeSafe's closed, hosted model, input-token pricing, stronger on
knowledge-heavy questions. Kev: Apache-2.0 reimplementation of the published
architecture on Qwen3.5 bases (0.8B to 27B), same `/v1/systemone` contract,
so the official SDK works against it.

**Why fp32 on the Mac?**
`kev.train` only enables bf16 autocast on CUDA. On Metal we train in fp32; at
0.8B this fits easily in 64 GB and is just slower.

**What would break on real data?**
Label noise (the oracle is noiseless), long states (we are at 43 tokens),
class imbalance you did not fix, and questions that need world knowledge the
0.8B base does not have. Kev's repo reports a wide gap to Jev on MMLU-style
questions; grid rules are not that.

**Could the base model have done it with a better prompt?**
The question text is the only prompt there is, and it is identical before
and after fine-tuning. The before/after rows differ only in the LoRA and head
weights.
