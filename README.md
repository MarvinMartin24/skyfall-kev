# Skyfall

A tiny grid skydiving game used to fine-tune a Kev decision model
(open reimplementation of TypeSafe's Jev) on one GPU, for a 30-minute demo.
Plan and rules: [PLAN.md](PLAN.md).

Play it yourself (from the repo folder, or by path from anywhere):

```bash
python3 skyfall.py            # a = left, d = right, enter = stay, add o to open (e.g. "ao")
```

```bash
python3 -m tests.run                                  # tests
python3 -m game.play --policy oracle --episodes 3 --show   # watch the rule-based oracle play
python3 -m game.play --policy oracle --episodes 500       # win rate
python3 -m game.play --policy oracle --episodes 500 --surprise   # adds obstacle "O" never seen in training
python3 -m data.generate --train 20000 --eval 2000        # writes data/train.jsonl, data/eval.jsonl
```

Layout: `game/sim.py` world, `game/oracle.py` the rule, `game/play.py` episodes,
`data/questions.py` the two typed questions, `data/generate.py` the dataset,
`model/` client + eval (Day 4).
