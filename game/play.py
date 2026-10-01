"""Run episodes with a policy. Usage: python -m game.play --policy oracle --episodes 5 --show"""
import argparse
import random

from game.sim import World, OBSTACLES, MOVES
from game.oracle import oracle


def random_policy(rng):
    return lambda w: {"move": rng.choice(MOVES), "open": rng.random() < 0.1}


def run_episode(policy, seed, obstacles=OBSTACLES, show=False):
    w = World(seed=seed, obstacles=obstacles)
    while not w.done:
        a = policy(w)
        if show:
            print(w.render())
            print(f"-> move={a['move']} open={a['open']}\n")
        w.step(a["move"], a["open"])
    if show:
        print(w.render())
        print("RESULT:", w.outcome, "ticks:", w.ticks, "\n" + "=" * 20)
    return w.outcome, w.ticks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["oracle", "random", "model"], default="oracle")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--surprise", action="store_true", help="add an obstacle type never seen in training")
    p.add_argument("--show", action="store_true")
    p.add_argument("--url", default="http://127.0.0.1:8009")
    args = p.parse_args()

    if args.policy == "oracle":
        policy = oracle
    elif args.policy == "random":
        policy = random_policy(random.Random(args.seed))
    else:
        from model.client import model_policy
        policy = model_policy(args.url)

    obstacles = OBSTACLES + ("O",) if args.surprise else OBSTACLES
    wins, ticks, reasons = 0, 0, {}
    for i in range(args.episodes):
        (result, reason), t = run_episode(policy, args.seed + i, obstacles, args.show)
        wins += result == "win"
        ticks += t
        reasons[reason] = reasons.get(reason, 0) + 1
    print(f"policy={args.policy} surprise={args.surprise} episodes={args.episodes}")
    print(f"win_rate={wins / args.episodes:.3f} mean_ticks={ticks / args.episodes:.1f}")
    print("outcomes:", dict(sorted(reasons.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
