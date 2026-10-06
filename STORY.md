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

A 15-line oracle plays this at 94% and labels every state. No gameplay was
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
  0.75 to 1.1 s per record. The Kev authors report about 20 minutes for a
  comparably sized run on an H100; we did not measure that.
- Kev's default augmentation stayed on: option order shuffled every time,
  and about a third of the `move` questions got a "none of the above" or a
  distractor option. The model trained on four-option questions it is never
  asked at demo time.
- LoRA rank 16 on every linear layer plus the pointer head: 11.3M trainable
  parameters out of about 800M, warm-started from the released checkpoint
  ("delta mode"). fp32, because Kev enables bf16 autocast on CUDA only.
- The run was restarted four times in the first hour, each time for a good
  reason (leak fix, rule change, clouds, snapshots). Early restarts are cheap;
  the lesson is to have the reviews *before* the long run.
- Kev only writes LoRA checkpoints at the end, so we added an 8-line patch
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
| 600 | 4,800 | 0.95 | 0.86 (Brier 0.21) |
| 700 | 5,600 | 0.98 | 0.87 (Brier 0.21) |
| 800 | 6,400 | 0.98 | 0.92 (Brier 0.13) |
| 900 | 7,200 | 0.97 | 0.90 (Brier 0.18) |
| 1000 | 8,000 | 0.98 | 0.96 (Brier 0.07) |
| 1300 | 10,400 | 0.99 | 0.98 (Brier 0.04) |

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

Win rates on live jumps (50 seeded jumps per snapshot): released Kev 0/200,
step 300: 18%, step 500: 42%, step 600: 38%, step 700: 24%, step 800: 58%,
step 900: 74%, step 1000: 76%, **step 1300: 96%**, final: 96%; the oracle
scores 94% on these layouts (it has no lookahead; the model imitates it, so
parity is the ceiling). All 50-jump numbers use seeds 8000–8049; the 100-jump
numbers in §10 use seeds 9000–9099.

The dip at step 700 while held-out accuracy kept rising is instructive. The
confusion matrix on 300 states (`results/step700-confusion.txt`) says why: every "left" case is right, but in
16 of 96 "right" cases the model picks **left**, and in 18 of 150 "stay" cases
it dodges for no reason. All 38 deaths in those 50 jumps were collisions,
none were parachute mistakes. So the model has learned "something is below
me, move" before it has learned *which way*. The oracle's side choice depends
on the neighbouring columns and on the tie-break toward the centre, which
needs the column index, a harder positional fact than "the cell under Y".
A jump needs about 20 consecutive correct decisions, so a few wrong-side
dodges cost more wins than the accuracy number suggests.

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
| released Kev (local) | 0/10 | 0.80 (8 of its 10 first-tick moves) | 280 ms | 0.3 | 0.0075 at Jev's list price as cited by Simon Willison |
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
3.8-flash with thinking "low" took 27 s per decision in a two-call probe
(`results/gemini-probes.md`), which at one row per 100 ms is 270 rows of free
fall. The larger variants were not measured further.

**Update, Oct 6.** The chain-of-thought race was finally run (`results/race-gemini-cot.md`,
20 jumps, seeds 6000-6019): Gemini flash-lite with a visible analysis field wins **15 of 20**
(move agreement 0.86, open 0.99) at 809 ms p50 and 1.1 s p95. So the 0/10 above is a
statement about the bare prompt, not about Gemini: thinking out loud is what makes it
competitive. It is still 14x slower than Kev (809 vs 56 ms), uses 627 tokens in and 73 out per
decision, and had 0.51 retries per decision on the free tier.

The free tier also matters for the story: 44% of lite decisions needed a
retry, so the wall time per decision was 2.5 s.

## 8. Takeaways, first draft (superseded by §12)

- A decision model is the same transformer with a different readout. One
  pass, probabilities out, options described in text.
- Synthetic data from a rule is legitimate when the rule *is* the task, and
  tests that re-derive labels and forbid leakage are what make it honest.
- Fine-tuning learns the easy signal first. Watch the curve, not the loss.
- An LLM can play this, but only by thinking, and thinking is what you pay
  for in time and tokens. The decision model's job is to not need it.

## 9. Step 1,300: it flies (Oct 1, 02:53)

Three quarters of an hour after the plateau broke, the snapshot at step 1,300
lands 48 of 50 jumps. On 300 held-out states it makes 7 move errors and 4
open errors, and the wrong-side dodges are gone (zero left/right confusions,
`results/step1300-confusion.txt`).
The success bar from `PLAN.md` (move >= 0.90, open >= 0.95, win rate >= 0.60)
is cleared with 1,200 steps of training still to go.

**The balloon test, first pass.** Same 50 seeds with hot-air balloons `O`
swapped into a quarter of the obstacles: 39 wins. Six deaths were balloons,
five were ordinary obstacles. The oracle wins 49 of the same 50. So the
model mostly applies "any other letter is deadly" to a letter it never saw,
but not always. A direct probe makes it concrete: with only one symbol placed
directly below `Y`, mean P(stay) is 1.00 for `.` and `C`, 0.00 for `B`, `P`,
`A` and for `Z` (another never-seen letter), and 0.11 for `O`. It generalises
the rule; `O` specifically is slightly ambiguous to it, perhaps because `O`
reads as a round, empty-looking glyph. That is a better slide than a clean
pass would have been.

One infrastructure note for honesty: the snapshot server stuck on step 900
for an hour because a text sort put `step-900` after `step-1300`. The
skipped snapshots were backfilled on a side port; the curve is complete.

## 10. The final model (Oct 1, 05:30)

One epoch, 2,500 steps, 5.8 hours on the laptop, loss at zero for the last
third. The final checkpoint is 62 MB: a rank-16 LoRA adapter, the pointer
head and a tokenizer.

| | released Kev | final fine-tune | oracle |
|---|---|---|---|
| move accuracy, held-out | 0.55 | 0.993 (2 errors in 300) | 1 |
| open accuracy, held-out | 0.47 | 1.000 | 1 |
| live jumps won, 100 seeds (9000–9099) | 0 | 94 | 97 |
| live jumps won, 50 seeds (8000–8049, the curve's seeds) | 0 | 48 | 47 |
| same seeds with balloons | 0 | 83 | 97 |

The two remaining move errors are both at exactly 1,000 m with something
deadly below, the one altitude where the open and move rules interact.

**The balloon, final answer.** 83 of 100 with a symbol the model never saw,
14 deaths by balloon. The probe explains it: P(stay) with one symbol under
`Y` is 0.00 for `B`, `P`, `A`, and also 0.00 for `Z` and `X`, letters it never
saw; but 0.33 for `O` and 0.61 for the digit `0`. It learned "any other
letter is deadly" and applies it to new letters, and it also learned that
round, hollow glyphs look like empty cells. Of all the symbols we could have
picked for the surprise, the balloon was the one that looks like a hole.
Had we chosen `Z`, the test would have passed cleanly. That is the slide.

## 11. The full benchmark, and the clock (Oct 1, 06:00)

`kev.benchmark` on all 2,000 held-out states: move accuracy 0.9975 (5 errors),
open 0.999, Brier 0.005 and 0.002, expected calibration error 0.002. The
released model on the same states: 0.554 and 0.467.

With the GPU finally idle, one decision (both questions, 43-token state)
takes **56 ms median** through Kev's MLX server, p95 84 ms,
1.3 s for a whole jump. The released model is the same speed
(56 ms): fine-tuning changes the answers, not the cost. At Jev's
list price that is $0.0075 per 1,000 decisions; locally it is
electricity. The Gemini numbers are in §7 and §13:
745 ms per decision and 0 of 10 jumps with the bare prompt; with a visible
chain of thought 0.70 move accuracy on 60 states and 15 of 20 live jumps at
809 ms; 27 s with the flagship's thinking on (one successful call).

Local race, 20 fresh seeds: oracle 20/20, released Kev 0/20, final 20/20.

## 12. What the audience should take away

- A decision model is the same transformer with a different readout. One
  pass, probabilities out, options described in text.
- Synthetic data from a rule is legitimate when the rule *is* the task, and
  tests that re-derive labels and forbid leakage are what make it honest.
  Two reviews found real bugs before training; both would have been
  invisible in the final accuracy.
- Fine-tuning learns the easy signal first. The altitude rule took 100
  steps, reading the grid took 1,300. Watch the curve, not the loss, and do
  not touch the learning rate during warm-up.
- The surprise test is only as good as the surprise. `Z` passes, `O` half
  fails, because `O` looks like a hole. Say which one you tested and why.
- An LLM can play this, but only by thinking, and thinking is what you pay
  for in time and tokens. The decision model's job is to not need it:
  56 ms versus 750-900 ms (Gemini flash-lite) with 15 of 20 jumps won
  by chain of thought. List price per 1,000 decisions: Kev $0.0075 (priced at
  Jev's rate; free locally), Gemini $0.07 with the bare prompt and $0.27 with
  chain of thought, i.e. 9x to 35x more. (Earlier drafts said "a hundredth of
  the price"; the measured ratio is not 100x.)
- Do not read "decision model" as "good at everything": Jev, the hosted
  original, scores 0.34 on `move` zero-shot (§13).

## 13. Round two: Jev and Gemini chain of thought, run live (Oct 6)

Two gaps were left open: Gemini's chain-of-thought race, and Jev itself. Both were
run on the same questions and the same states as Kev, with no prompt tuning.

**Jev (`jev-1.13.0`, hosted, zero-shot)**, `scripts/jev_test.py`, 2,000 held-out
states and 20 live jumps on seeds 6000-6019, about $0.04 in total at list price:

| | released Kev | **Jev** | fine-tuned Kev |
|---|---|---|---|
| move accuracy | 0.554 | **0.343** | 0.9975 |
| open accuracy | 0.467 | **0.9485** | 0.999 |
| Brier (overall) | 0.566 | 0.353 | 0.003 |
| ECE | 0.117 | 0.062 | 0.0017 |
| jumps won (seeds 6000-6019) | 0/20 | 3/20 | 20/20 |
| latency per decision, median | 56 ms (local) | 176 ms (remote) | 56 ms (local) |

Jev gets `open` almost right and `move` below the always-stay rate (0.554).
Its `open` mistakes are all the same kind: all 94 of its false opens are at 1,000 m
or lower with something deadly directly below, and it said open on 94 of those
100 states, which is exactly the case the oracle's "cell below must be clear"
clause covers. Its `move` errors are mostly over-dodging: it predicts `stay` on only 7% of
states (true share 55%) and moves on 90% of the 1,108 states where the right answer is to
stay (992 of them). When a dodge really is needed (892 states) it picks the wrong side 33% of
the time and stays 3% of the time (confusion matrix in `results/jev-eval-2000.json`). The run was repeated once: 0.3375 and 0.343 move
accuracy, so this is not noise. Read it as "zero-shot, with our wording, on a
character grid", not as a verdict on Jev: TypeSafe's own docs list literal reading,
large irrelevant state and numbers among its failure modes, and Red Hat found Jev's
accuracy moves with policy wording. A prompt-tuned run was not done.

Latency is not like for like: Kev ran on the same Mac with an idle GPU, Jev through the
network. Jev billed 474 input tokens per request (its own tokenizer; Kev's is 179), so
at its list price a decision costs $0.020 per 1,000 decisions, not the $0.0075 we use for
Kev.

**Gemini flash-lite with visible chain of thought**, 20 live jumps on seeds 6000-6019:
15 wins, move agreement 0.86, open 0.99, 809 ms p50. Details in §7 and
`results/race-gemini-cot.md`.

GIFs on seeds 7000-7002: `results/gifs/jev.gif` (loses all three) and
`results/gifs/gemini-cot.gif` (wins two of three).
