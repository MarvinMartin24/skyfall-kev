# Demo: 30 minutes, what to run and what to say

Before the talk: Kev server running on the **fine-tuned** checkpoint on port
8009 (step 5 of `REPRODUCE.md`), a second terminal in the repo, the clips in
`results/clips/` open as backup, `results/table.md` on a slide.

| Min | Do | Say |
|---|---|---|
| 0–4 | Slide: generator vs decision model | "Same transformer, different readout. State in, typed questions in, probabilities out, one forward pass. No decoding, no parsing." Point at the three question types. |
| 4–8 | `python3 -m game.play --policy oracle --episodes 1 --show` then show `game/oracle.py` | "Five columns, six rows, you are Y, things scroll up at you. The rule is fifteen lines. That rule is the teacher." |
| 8–11 | `python3 -m data.generate --train 20000 --eval 2000` then `head -1 data/eval.jsonl` | "Twenty thousand labelled states in a few seconds. Two typed questions per state. No gameplay was recorded; the oracle labels everything. Eval states are excluded from train and a test re-derives every label." |
| 11–14 | Slide: the "before" row; play the base-model clip | "This is the released Kev-0.8B, zero-shot, same questions. It has never seen a grid. It dies in a few ticks." |
| 14–18 | Slide: architecture (EXPLAINER §2–4), then the training command | "LoRA rank 16 on the frozen backbone plus a 2-layer pointer head, warm-started from the released checkpoint. Trained once on this laptop; the command is on screen." |
| 18–24 | `python3 -m game.play --policy model --episodes 1 --show` live, then `--episodes 200` | "Watch the probabilities. When nothing is below it, P(stay) is near one. When a bird appears, mass moves to a side. Below 1000 m with a clear cell, P(open) jumps." Then the table. |
| 24–28 | `python3 -m game.play --policy model --episodes 200 --surprise` | "A hot-air balloon it has never seen." Read the number. Then the latency slide and the caveats in EXPLAINER §9. |
| 28–30 | Questions | FAQ.md has the answers. |

## If something breaks

- Server down: `results/clips/` has the base and fine-tuned runs recorded.
- Numbers questioned: every row in `results/table.md` names its command.

## The table (filled from results/)

See `results/table.md`.
