# Skyfall: the story so far

A journal of how this demo was built, in the order it happened, with the
numbers as they came in. Written for the talk: every twist here is a slide.
(`PLAN.md` is what we intended; this is what happened.)

---

## 1. The question (Tue Sep 30, evening)

TypeSafe shipped **Jev**, a "decision model": a transformer that takes a state
and typed questions and returns probabilities in one forward pass, instead of
generating text. Jared Palmer cloned the architecture as **Kev**, open source,
on Qwen3.5 bases. The claim behind both is speed and cost: one pass, no
decoding, no parsing.

We wanted a demo where (a) the released model is at chance, (b) a few thousand
synthetic examples make it good, (c) it runs on one Mac, and (d) the audience
can see it play. A game does all four, and a skydiving game fits the owner.

## 2. Designing a game the model cannot guess

Four candidate tasks were on the table: prompt-injection detection, toxicity,
a skydiving sensor debrief, and a grid game. The first two fail (a): public
datasets mean the base model already does them. The sensor debrief fails (c)
on data volume. The grid game is the "XOR" of decision models: a rule that is
trivial to state, impossible to guess from text priors, and checkable.

Final rules took three iterations:

1. 5 x 6 grid, `Y` on row 1, obstacles scroll up, altitude 3000 -> 0.
2. **Open rule changed** after review: "open only when the cell below is
   already clear, sidestep first otherwise". Each question had to be a pure
   function of the state, because Kev answers the two questions without
   either seeing the other (block-causal mask).
3. **Clouds added** (`C`, harmless) so the model must learn which letters kill,
   not "non-empty means wall". The balloon `O`, never in training, became the
   test of whether it applies the sentence "any other letter is deadly".

A 15-line oracle plays this at 95% and labels every state. No gameplay was
ever recorded; `data/generate.py` makes 20,000 labelled states in 3 seconds.

## 3. Two code reviews that earned their keep

Independent reviewers (read-only, exhaustive enumeration of 64,800 reachable
states) accepted the game and the generator, but found:

- **Eval leakage.** Different random seeds for train and eval were not enough:
  2.1% of eval states also appeared in train, and after an exact-text fix, 9%
  of eval *grids* were still in train at a different altitude, which is the
  same decision input. Exclusion is now keyed on (grid, altitude <= 1000 m).
- **A label that was secretly trivial.** In the first dataset `open` equalled
  `altitude <= 1000` on all but one row, because the oracle sidestepped and
  opened in the same tick. The "cell below must be empty" clause was never
  tested. That drove the rule change above and a dedicated 5% data quota for
  "low altitude but blocked".
- **The balloon changed the layouts.** Adding a fourth obstacle type consumed
  the random generator differently, so "with balloon" jumps were different
  jumps. Fixed with a separate random stream; a test now proves the layouts
  are identical with and without balloons.

Talking point: a test that re-derives every label from the grid text caught
the first leak within minutes. Say that out loud; audiences trust it.

## 4. The zero-shot baseline (released Kev-0.8B)

On 2,000 held-out states: move accuracy 0.554, which is exactly the share of
"stay" labels; open accuracy 0.467, worse than always answering no; Brier
0.57. Its move probabilities sit at about 0.27 / 0.35 / 0.38 on every state and
P(open) is about 0.5 at every altitude, so it opens on the first tick and
loses every jump. Condition (a) holds.

## 5. Training on a laptop

- Apple M2 Max, 64 GB. Training runs on the GPU through PyTorch's Metal
  backend (`device=mps`). Serving uses MLX in bfloat16. Two Qwen3.5 kernels
  only have CUDA builds, so on the Mac they fall back to plain PyTorch: about
  0.75 to 1.1 s per record, versus the 20-minute epoch the Kev authors get on
  an H100.
- LoRA rank 16 on every linear layer plus the pointer head: 11.3M trainable
  parameters out of about 800M, warm-started from the released checkpoint
  ("delta mode"). fp32, because Kev enables bf16 autocast on CUDA only.
- The run was restarted four times in the first hour, each time for a good
  reason (leak fix, rule change, clouds, snapshots). Early restarts are cheap;
  the lesson is to have the reviews *before* the long run.
- Kev only writes LoRA checkpoints at the end, so we added a 12-line patch
  (`patches/kev-lora-snapshots.diff`) that saves a loadable checkpoint every
  100 steps. Everything below depends on it.

## 6. Watching it learn (Oct 1, after midnight)

Snapshots are served automatically on port 8010 as they land, so the same
jump can be watched with a progressively better brain.

| step | records seen | open accuracy | move accuracy |
|---|---|---|---|
| 0 (released) | 0 | 0.47 | 0.55 (= always stay) |
| 100 | 800 | 0.89 | 0.51 |
| 200 | 1,600 | 0.93 | 0.50 |
| 300 | 2,400 | 0.93 | 0.57 (Brier 0.53, first real movement) |
| 400 | 3,200 | 0.94 | 0.69 (Brier 0.37) |
| 500 | 4,000 | 0.94 | 0.77 (Brier 0.28) |

The model learned the altitude rule almost immediately and, for the first
250 steps, did not read the grid at all. It always says "stay" and flies into the first obstacle.
That is what Marvin noticed watching it. Why it is plausible: the altitude is
a number the backbone already knows how to read; "the cell six tokens after
Y" is a positional pattern the adapters have to invent. Also, Kev's schedule
(OneCycle, 10% warm-up) means the learning rate only reached its peak at step
250, so judging at step 200 was early.

What we did about it: a probe run in parallel with a 5x higher learning rate
(1e-4, 300 steps, 1,200 records) to see whether the plateau was a
learning-rate problem, and a watcher that evaluates every snapshot and logs
the curve to W&B. Verdict: the probe never left 0.50 on move; the main run
broke through on its own once warm-up ended (0.57 at step 300, 0.69 at step
400). The original recipe stays. Lesson for the talk: a plateau right after
warm-up is not a reason to touch the learning rate; look at the curve one or
two snapshots later.

Win rates on live jumps at this stage: released Kev 0/200 (opens on tick 1),
step-300 snapshot 9/50, oracle 95%.

## 7. The race against an LLM

The point of the architecture is latency and cost, so we built "the race":
the same seeded jumps played by the oracle, released Kev, fine-tuned Kev and
Gemini, measuring per decision latency, tokens, cost, and agreement with the
oracle, plus "time as gravity": seconds the skydiver spends in the air if the
sky waits for each decision.

First round, 10 jumps, released Kev vs the fastest Gemini (flash-lite, no
thinking):

| player | wins | move agreement | latency p50 | seconds per jump | $ per 1,000 decisions |
|---|---|---|---|---|---|
| released Kev (local) | 0/10 | 0.80 | 280 ms | 0.3 | 0.0075 at Jev list price |
| Gemini flash-lite | 0/10 | 0.37 | 745 ms | 3.1 | 0.07 |

Two different failures: Kev opened too high every time; Gemini never opened
early but steered into things. Then the fair question: was the prompt too
thin? We wrote three richer prompts and scored them on 60 held-out states:

| prompt | move acc | open acc |
|---|---|---|
| two bare questions | 0.33 | 0.82 |
| full rules + labeled grid | 0.43 | 0.93 |
| rules + step-by-step procedure | 0.45 | 0.92 |

Even with the procedure spelled out, a no-thinking lite model cannot execute
it on the grid in one shot (always-stay would score 0.55). Letting it write a
one-sentence analysis before the answer (visible chain of thought, ~70 output
tokens) changed everything: **move 0.70, open 1.00, at 896 ms per decision**.
"Minimal" thinking on the 3.1 lite model did nothing (0.37). The flagship
3.8-flash with thinking took 27 s per decision in probing, which at one row
per 100 ms is 270 rows of free fall. Larger variants are being measured.

The free tier also matters for the story: 44% of lite decisions needed a
retry, so the wall time per decision was 2.5 s.

## 8. What the audience should take away (draft)

- A decision model is the same transformer with a different readout. One
  pass, probabilities out, options described in text.
- Synthetic data from a rule is legitimate when the rule *is* the task, and
  tests that re-derive labels and forbid leakage are what make it honest.
- Fine-tuning learns the easy signal first. Watch the curve, not the loss.
- An LLM can play this, but only by thinking, and thinking is what you pay
  for in time and tokens. The decision model's job is to not need it.

_(continued as results land: learning curve, final model, balloon test, the
full race)_
