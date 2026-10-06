# Skyfall

A tiny grid skydiving game used to fine-tune a Kev decision model
(open reimplementation of TypeSafe's Jev) on one GPU, for a 30-minute demo.
Plan and rules: [PLAN.md](PLAN.md). The story of how it was built, for the talk: [STORY.md](STORY.md).

Play it yourself (from the repo folder, or by path from anywhere). In-place
terminal UI, single keys: `a`/`←` left, `d`/`→` right, enter stay, space open,
`h` shows what the rule would do, `r` next jump, `q` quit.

```bash
python3 skyfall.py                                  # you play
python3 skyfall.py --policy oracle --fps 4          # watch the rule play
python3 skyfall.py --policy model --fps 2           # watch the fine-tuned model, with live probability bars (needs kev.serve on :8009)
python3 skyfall.py --policy model --fps 2 --surprise   # ...with the balloon it never saw
```

```bash
python3 -m tests.run                                  # tests
python3 -m game.play --policy oracle --episodes 3 --show   # watch the rule-based oracle play
python3 -m game.play --policy oracle --episodes 500       # win rate
python3 -m game.play --policy oracle --episodes 500 --surprise   # adds obstacle "O" never seen in training
python3 -m data.generate --train 20000 --eval 2000        # writes data/train.jsonl, data/eval.jsonl
```

Compare against hosted models (keys go in a gitignored `.env`, see `.env.example`; never in chat):

```bash
python3 scripts/jev_test.py smoke                       # 3 calls, prints the response shape and estimated spend
python3 scripts/jev_test.py eval --n 2000 --max_usd 1   # accuracy / Brier / ECE of hosted Jev on data/eval.jsonl
python3 scripts/jev_test.py race --episodes 20          # live jumps, seeds 6000-6019
python3 -m model.bench --players oracle,gemini:gemini-flash-lite-latest --gemini_prompt cot --episodes 20 --seed 6000 --out results/race-gemini-cot
```

Layout: `game/sim.py` world, `game/oracle.py` the rule, `game/play.py` episodes,
`data/questions.py` the two typed questions, `data/generate.py` the dataset,
`model/` client + eval, `scripts/jev_test.py` hosted-Jev eval and race (stops at a spend cap, aborts on repeated API errors).
