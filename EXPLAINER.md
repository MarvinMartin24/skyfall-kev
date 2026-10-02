# How this works, at the level you can defend in front of ML engineers

Everything in this document about Kev's internals was read from the Kev source
(`kev/model.py`, `kev/train.py`, `kev/api.py`, commit `1d77363`) and from the
released checkpoint's `training_config.json`. Where a claim is ours, it is
backed by a number in `results/`.

## 1. Two kinds of language model

A generative LLM is trained to predict the next token. To get a decision out
of it you write a prompt, let it decode tokens one at a time, and parse the
text. Every decoded token is a full forward pass through the network, and the
answer arrives as a string you must trust and parse.

A **decision model** keeps the same transformer backbone but changes what you
read out of it. You give it a *state* (any text) and one or more *typed
questions*, and it returns a probability distribution per question from **one
forward pass**. No decoding loop, no parsing, no sampling. TypeSafe calls this
"System One" and ships it as Jev (closed). Kev is the open reimplementation
we train here.

Three question types, all answered the same way underneath:

| Type | You give | You get |
|---|---|---|
| `choice` | named options, each with a description | a probability per option |
| `noul` | a yes/no statement | P(true) |
| `score` | ordered levels | a distribution over levels, plus the expected level |

## 2. What the backbone actually sees

Kev packs one request into a single token sequence:

```
<state> altitude: 1200 ⏎ . . . B . ⏎ B . . . Y ⏎ ...          ← the state, once
<q> Which way should you move... <opt> left: move one column... </opt> <opt> right: ... </opt> <opt> stay: ... </opt> <decide>
<q> Should the skydiver open... <opt> no </opt> <opt> yes </opt> <decide>
```

The delimiters `<state> <q> <opt> </opt> <decide>` are five rarely used Qwen
special tokens that Kev repurposes, so no new embeddings are needed; the LoRA
learns what they mean. User text is sanitised so it can never produce these
tokens, which makes option boundaries unforgeable.

Attention is **block-causal**: state tokens attend causally among themselves;
each question's branch attends to the state and to its own earlier tokens, and
**never to another question's branch**. Branch position ids restart right
after the state.

One honest detail about *our* backbone. That packed, single-sequence form is
what Kev uses on attention-only models. Qwen3.5 is a hybrid: most of its
layers are Gated DeltaNet, which is recurrent and cannot honour a block-causal
mask. So on Qwen3.5 Kev encodes the state once and runs **each question as its
own causal row continuing from the state's cache**. Kev tests that the two
forms give the same probabilities to fp32 noise, so everything below holds
either way, and there is still no token generation: one pass over the state
plus one short pass per question. Two consequences:

- Questions in one call are independent by construction. Asking ten
  questions does not change the answer to any one of them.
- The state's hidden states and key/value cache are identical with or without
  the branches, so a server can encode the state once and run the question
  branches as cheap rows on top of the cached prefix. That is where the
  latency win comes from.

Our Skyfall state is 43 tokens. Kev trains on states up to 384 tokens by
default and serves up to 64k.

## 3. The pointer head (the whole trick, in 8 lines)

After the backbone, Kev reads two kinds of hidden vectors per question:

- `h_decide`: the hidden state at the `<decide>` token. It has attended to the
  state, the instructions and every option. Think of it as "the question,
  having read everything".
- `h_opt[k]`: the hidden state at the `</opt>` token closing option *k*. It
  has attended to the state, the instructions, option *k*'s text, and, in the
  default configuration we used, the options listed before it. (Kev has an
  `option_isolation` mode that removes that last dependency; it needs the
  packed mask and is not available on hybrid backbones. This is where the
  option-order sensitivity in §9 comes from.)

The head is two linear maps into a 256-dim space and a dot product:

```python
class PointerHead(nn.Module):
    def __init__(self, d, dp=256):
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
    def forward(self, h_decide, h_opts):        # [d], [K, d] -> logits [K]
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale
```

Softmax over the K logits is the answer. It is literally one attention
query ("the decision") over K keys ("the options"), which is why it is called
a *pointer*: the model points at an option. A `noul` question is a 2-option
pointer over the options "no" and "yes" (in that order); a `score` is a pointer over the ordered levels.

Why this beats a classifier head: a linear classifier has a fixed label set
baked into its weights. The pointer head scores **whatever option text you
wrote**, so K can change per question and options are described in natural
language. The model learned "how to compare a decision to an option", not
"which of these 3 labels". That is what makes new questions at inference time
possible without retraining.

Why this beats next-token logprobs on "A/B/C": options are arbitrary spans
with descriptions, not single tokens; each option's representation has read
its own description; the objective was trained for this readout; and released Kev checkpoints
carry a calibration temperature fitted on held-out rows (`head.pt["temperature"]`,
2.35 for the released 0.8B) that divides logits at inference without changing
the argmax. Our fine-tune never ran Kev's `scripts/calibrate_checkpoint.py`, so it
serves raw logits (temperature 1.0); its expected calibration error on the
held-out set is 0.002 regardless.

## 4. What fine-tuning changes

Kev fine-tunes with **LoRA** (rank 16, alpha 32, dropout 0.05, on every linear
layer of the backbone) plus the pointer head. The 0.8B backbone's original
weights are frozen. We use Kev's "delta mode": `--init_from jaredpalmer/kev-0.8b`
warm-starts the LoRA and head from the released checkpoint rather than from
zero. The repo reports one user's test where training from scratch scored 0.33 on
Kev's own evaluation set against 0.84 when warm-started, with 0.88 on that
user's new domain. It means we are *adding* Skyfall to a model that already
knows how to be a decision model, not teaching the readout from scratch.

Kev's trainer also augments every `choice` question by default, and we left
the defaults on: option order is shuffled on every record; with probability
0.10 the true option is removed and a "none of the above" option becomes the
answer; with 0.12 a "none" option is added as a wrong alternative; with 0.15
an irrelevant distractor option is added. So about a third of the `move`
questions the model trained on had four options, not three. That is Kev's
built-in defence against learning option positions, and it is also why the
model is comfortable with option sets it did not see at training time.

What the LoRA has to learn for Skyfall, concretely:

1. Locate `Y` in the second grid row and infer its column from the token
   sequence (the grid is one token per cell, so "column" is a position
   pattern, not a number the model is told).
2. Read the cell directly under `Y` in the next row.
3. If it is neither `.` nor `C`, compare the two neighbouring columns at rows 1 and 2.
4. For `open`: compare the altitude number to 1000 **and** check that the
   cell directly below is empty. Both conditions matter: 5% of the data is
   "low altitude but something is under you", where the right answer is to
   sidestep and not open.

Note what question isolation implies here. Kev answers `move` and `open`
without either seeing the other, so each label must be a pure function of the
state. That is why the rule says "open only when the cell below is *already*
empty" rather than "sidestep and open in the same tick": the second version
would make `open` depend on the `move` answer, which the model cannot see.

None of this is in the released model's training data, which is why it starts
at the majority class (move 0.55 = "always stay") and below chance on open
(0.47). What the curve then shows, in order: the altitude rule by step 100
(open 0.89), nothing on the grid until the learning-rate warm-up ends at step
250, "something is below me" by step 400 (move 0.69), "which way" by step
800 (0.92), parity with the rule by step 1,300 (0.98, and 48 of 50 live
jumps). Easy signals first, positional ones later. It is also why the task is a clean test:
the rule is simple to state, impossible to guess from text priors, and
checkable against a 15-line oracle.

The loss is cross-entropy against the oracle's label for each question. There
is no reinforcement learning anywhere: this is **behaviour cloning of a rule**.
Say that out loud in the demo.

## 5. Why synthetic data is legitimate here

The oracle is the definition of correct play, so labels are ground truth by
construction, the dataset is as large as we like, and evaluation is exact.
Four safeguards make it
honest:

- Rollouts mix 30% random moves so the data covers bad positions the oracle
  alone would never reach (otherwise the model never sees "you are next to a
  wall with a bird under you").
- Train and eval are generated from disjoint seeds **and** eval states are
  explicitly excluded from train. A test re-parses every rendered state and
  recomputes the oracle label; a second test asserts zero overlap. The first
  version of the generator leaked 42 eval states into train; the test caught
  it.
- The class mix is fixed (40% dodge, 15% open, 5% low-altitude-but-blocked,
  40% idle) so "always say stay" and "open iff altitude <= 1000" cannot score
  well. The first generator had no quota for the blocked-at-low-altitude case,
  and in that data `open` was perfectly predictable from the altitude alone;
  the review caught it.
- Cross-split exclusion is keyed on the decision inputs (the grid and which
  side of 1000 m the state is on), not on the exact text, after the review
  found 9% of eval grids in train at a different altitude.

## 6. Calibration, and why probabilities matter

Accuracy only checks the argmax. A decision model's value is the probability,
because that is what you threshold in production ("open only if P(open) >
0.9"). We report the **Brier score**: the mean squared error between the
predicted probability vector and the one-hot truth. 0 is perfect; for a
3-way question, always predicting uniform gives 0.667. Lower is better. The
repo reports Kev-27B at 0.225 vs Jev at 0.211 on their eval; our numbers are
in `results/table.md`.

## 7. The balloon test

Training data contains obstacles `B`, `P`, `A`. At demo time we add `O`, a
hot-air balloon, which the model has never seen. The oracle does not care
(anything that is not `.` or `C` is a wall) and keeps winning 94%.

- If the fine-tuned model keeps dodging `O`, it learned "any letter except C
  below means move", a rule over the grid, not a lookup over three letters.
- If it flies into balloons, it learned the letters. That is also a real
  result: it tells you the LoRA latched onto token identity, and the fix is
  obvious (more obstacle symbols in training).

What happened: 83 of 100 jumps with balloons, 14 deaths by balloon, against
97 for the oracle. A probe with a single symbol under `Y` gives P(stay) = 0.00
for `B`, `P`, `A` and also for `Z` and `X`, letters it never saw, but 0.33 for
`O` and 0.61 for the digit `0`. The rule generalised to new letters; round,
hollow glyphs are half-read as empty cells. The surprise symbol we chose was
the one that looks like a hole.

## 8. Why it is fast (the closing slide)

For our two questions, one Kev decision is: encode 43 state tokens, run two
short question rows on that cache, two softmaxes. Measured: 56 ms median per
decision on the M2 Max through Kev's MLX server. A generative model answering
the same thing must run a forward pass per generated token (a 20-token JSON
answer is 20 passes plus the prompt) and then parse: 745 ms for the smallest
Gemini with a bare prompt, about 900 ms with a visible chain of thought, 27 s
for the flagship with thinking on. `results/table.md` and `results/race-*.md`
have the measurements. The architectural reason is in section 2: there is no
decoding loop to run.

## 9. Limits, said plainly

- A tiny CNN or even logistic regression on the 30 cells would solve Skyfall.
  The demo is about the API shape and the one-device fine-tuning loop, which
  carry over to real text decisions; it is not a claim that transformers are
  the right tool for 5x6 grids.
- Imitation of a rule, not learning by playing.
- Kev-0.8B is not Jev. Jev is closed and stronger on knowledge-heavy
  questions. Kev reproduces the architecture and the API contract.
- Option order can still influence answers; Kev's benchmark can average over
  option rotations (`--rotations`) and training has an `option_isolation`
  mode. We use neither, for simplicity.
- On a Mac, Kev trains in fp32 (`--dtype bf16` is CUDA-only in `kev.train`),
  which is slower than the Colab path but fine at 0.8B.
- Our checkpoint is uncalibrated (temperature 1.0); released Kev checkpoints
  are calibrated. The before/after Brier rows compare a calibrated released
  model with raw logits from ours; the gap is far too large for that to matter.

## 10. Glossary

- **State**: the text the questions are about. Here, the rendered grid plus altitude.
- **Branch**: one question's tokens (instructions, options, decide), attending to the state only.
- **Pointer head**: the tiny module that scores options by dot product with the decision vector.
- **LoRA**: low-rank adapters added to frozen weights; the only backbone parameters we train.
- **Delta mode**: initialise LoRA and head from a released Kev checkpoint, then train further.
- **Brier score**: mean squared error of probabilities; measures calibration and accuracy together.
- **Oracle**: the rule-based policy that defines correct play and labels the data.

flowchart TB

    %% ───────────────────────────────────────────────
    %% 1. SYSTEM PIPELINE: offline vs online
    %% ───────────────────────────────────────────────
    subgraph PIPE["1. SYSTEM PIPELINE"]
        direction TB
        subgraph OFFLINE["Offline / index time (LLM budget lives here)"]
            A[New chunk + provider embedding] --> B[LLM: generate 2-3 questions this chunk answers]
            B --> C[Embed questions as separate rows, metadata -> chunk_id]
            A --> D[Mini-batch k-LLMmeans, soft assignment to angle clusters]
            D -->|ambiguous or outlier| E[LLM slow path: assign / spawn / merge]
            D --> F[LLM rewrites angle descriptions periodically, temporal smoothness penalty]
            F --> G[Embed angle descriptions -> routing index]
            H[Query logs + reranker verdicts] --> I[Store query embedding as row -> chunk_id]
            H --> J[Distil LLM reranker into small cross-encoder]
        end
        subgraph ONLINE["Online / query time (no LLM on hot path)"]
            Q[User query] --> QE[Embed query, one call]
            QE --> R[Match against routing index, pick top 3-5 angles]
            R --> S[Per angle: vector search over content + question + logged-query rows, collapse by chunk_id]
            S --> T[Cross-encoder rerank within angle]
            T --> U[MMR, pick 2-3 chunks per angle]
            U --> V[Answer structured by angle]
            R -.->|no angle matches well| W[Fallback: live LLM facet generation]
        end
        G --> R
        C --> S
        I --> S
        J --> T
        V --> H
    end

    %% ───────────────────────────────────────────────
    %% 2. GRAPH SCHEMA: nodes and edges
    %% ───────────────────────────────────────────────
    subgraph SCHEMA["2. GRAPH SCHEMA"]
        direction TB
        ART["ARTICLE
        id, title, source, published_at
        (provenance, no vector)"]
        CH["CHUNK
        id, text, position
        vec_content (from provider API)
        summary (optional)"]
        GQ["GENERATED_QUESTION
        text, vec (own row in vector index)
        created_by: llm, at ingest"]
        LQ["LOGGED_QUERY
        text, vec (own row in vector index)
        confirmed_by: reranker | click
        count, last_seen"]
        ANG["ANGLE_CLUSTER
        id, description_vN, vec_description
        version, created_at, parent_id
        status: active | merged | split"]
        ENT["ENTITY (optional)
        name, type
        e.g. Paris, Haussmann"]

        ART -->|HAS_CHUNK, position| CH
        CH -.->|NEXT, prev| CH
        CH -->|ANSWERS, generated| GQ
        LQ -->|ANSWERED_BY, confidence| CH
        CH -->|BELONGS_TO, weight 0..1, since_iter| ANG
        ANG -.->|CANNOT_LINK, editorial| ANG
        ANG -->|SUPERSEDES, merge/split| ANG
        CH -->|MENTIONS| ENT
        ANG -->|ABOUT| ENT
        GQ -.->|SAME_INTENT| LQ
    end

    %% ───────────────────────────────────────────────
    %% 3. ONE ITERATION: how an angle cluster evolves
    %% ───────────────────────────────────────────────
    subgraph EVOLVE["3. ONE ITERATION OF EVOLUTION"]
        direction LR
        subgraph T0["Iteration N"]
            A1["ANGLE 17 v3
            'Practical visitor guidance for Paris'
            members: 412"]
            A2["ANGLE 23 v1
            'Urban history of Paris'
            members: 88"]
            C1["chunk_9001 (Haussmann boulevards)
            BELONGS_TO 17 w=0.55
            BELONGS_TO 23 w=0.52"]
            C1 --> A1
            C1 --> A2
        end
        subgraph T1["Iteration N+1 (new batch of 5000 chunks)"]
            B1["ANGLE 17 v4
            'Practical visitor guidance for Paris'
            members: 430, description unchanged"]
            B2["ANGLE 23 v2
            'How 19th-century planning shaped modern Paris'
            members: 160, description rewritten"]
            B3["ANGLE 51 v1 (NEW)
            'Paris housing policy and affordability'
            spawned by slow path from 40 outliers"]
            C2["chunk_9001
            BELONGS_TO 23 w=0.71 (strengthened)
            BELONGS_TO 17 edge dropped, below 0.3"]
            C2 --> B2
            B1 -.->|CANNOT_LINK| B2
            B2 -->|SUPERSEDES| A2
        end
        T0 --> T1
    end

    PIPE ~~~ SCHEMA
    SCHEMA ~~~ EVOLVE

Summary

The system is built on one principle: every LLM call runs offline, and the query path is embeddings plus small models only.

Section 1, the pipeline. At ingest, each chunk gets its provider embedding, an LLM writes two or three questions it answers, and those questions are stored as separate vector rows pointing back to the chunk. Mini-batch k-LLMmeans softly assigns chunks to angle clusters, with an LLM slow path handling only outliers and ambiguous cases via assign, spawn, or merge. Periodically the LLM rewrites each angle's description under a temporal smoothness penalty, and those descriptions are embedded into a small routing index. Query logs feed back as additional vector rows and as training data for a distilled cross-encoder. At query time, the query is embedded once, matched to a few angles via the routing index, chunks are retrieved per angle across content, generated-question and logged-query rows, reranked by the cross-encoder, and diversified with MMR. A live LLM is only a fallback when no angle fits.

Section 2, the graph. Articles are provenance-only nodes. Chunks carry the content vector. Generated questions and logged queries are first-class nodes with their own vectors. Angle clusters are versioned nodes with a description vector and lineage. The three edges that carry the design are BELONGS_TO with a weight, which makes membership soft and auditable, CANNOT_LINK between angles, which stores editorial constraints explicitly, and SUPERSEDES, which preserves history across merges and splits.

Section 3, evolution. Between iterations, chunk vectors never change. What changes is edge weights, angle descriptions, and occasionally the set of angles. The example shows one angle holding steady, one being rewritten and pulling a borderline chunk decisively toward it, and a new angle spawned from unassigned outliers.
