"""Learning curve figure for the slides, from results/learning_curve.jsonl and results/winrate_curve.jsonl.
.venv-wandb/bin/python scripts/plot_curve.py --out results/learning_curve.png"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    path = os.path.join(HERE, "results", name)
    rows = {}
    for l in open(path):
        r = json.loads(l); rows[r["step"]] = r          # last write wins per step
    return [rows[k] for k in sorted(rows)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/learning_curve.png"); a = ap.parse_args()
    lc, wr = load("learning_curve.jsonl"), load("winrate_curve.jsonl")
    base = json.load(open(os.path.join(HERE, "results", "baseline.json")))
    steps = [0] + [r["step"] for r in lc]
    move = [base["move_acc"]] + [r["move_acc"] for r in lc]
    opn = [base["open_acc"]] + [r["open_acc"] for r in lc]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    ax[0].plot(steps, move, "o-", label="move accuracy", color="#2b8cff")
    ax[0].plot(steps, opn, "s-", label="open accuracy", color="#38c172")
    ax[0].axhline(0.55, ls=":", color="gray"); ax[0].text(steps[-1], 0.555, "always 'stay'", ha="right", va="bottom", fontsize=8, color="gray")
    ax[0].set_xlabel("optimizer step (8 records each)"); ax[0].set_ylabel("accuracy on 300 held-out states"); ax[0].set_ylim(0.4, 1.02); ax[0].legend(loc="lower right"); ax[0].set_title("What it learns, and when")
    ax[1].plot([0] + [r["step"] for r in wr], [0.0] + [r["win_rate_50"] for r in wr], "o-", color="#e3342f", label="fine-tuned Kev")
    ax[1].axhline(0.95, ls="--", color="gray"); ax[1].text(0, 0.955, "oracle (the rule): 0.95", va="bottom", fontsize=8, color="gray")
    ax[1].set_xlabel("optimizer step"); ax[1].set_ylabel("win rate, 50 seeded jumps"); ax[1].set_ylim(0, 1.02); ax[1].legend(loc="lower right"); ax[1].set_title("Can it land?")
    for x in ax: x.grid(alpha=0.3)
    fig.suptitle("Skyfall: Kev-0.8B + LoRA, one M2 Max, 20k synthetic states", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, a.out)); print("wrote", a.out)


if __name__ == "__main__":
    main()
