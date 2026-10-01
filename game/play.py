"""Run episodes with a policy.

  python -m game.play --policy oracle --episodes 5 --show     # watch the rule play
  python -m game.play --policy human                          # play yourself: a = left, d = right, enter = stay, add o to open
  python -m game.play --policy model --url http://127.0.0.1:8009 --episodes 200 [--surprise]
"""
import argparse
import random

from game.sim import World, MOVES
from game.oracle import oracle


def random_policy(seed):
    rng = random.Random(seed)
    return lambda w: {"move": rng.choice(MOVES), "open": rng.random() < 0.1}


def human_policy(seed):
    def policy(w):
        while True:
            s = input("move [a=left  d=right  enter=stay]  add o to open the parachute > ").strip().lower()
            if s in ("", "a", "d", "s", "o", "ao", "do", "so", "oa", "od", "os"):
                move = "left" if "a" in s else "right" if "d" in s else "stay"
                return {"move": move, "open": "o" in s}
            print("keys: a, d, enter, o   (e.g. 'ao' = move left and open)")
    return policy


def run_episode(make_policy, seed, surprise=False, show=False):
    """make_policy(seed) -> policy(world) -> {"move", "open"}; a fresh policy per episode keeps episodes reproducible."""
    policy = make_policy(seed)
    w = World(seed=seed, surprise=surprise)
    while not w.done:
        if show:
            print(w.render())          # show the state first, then decide (matters for --policy human)
        a = policy(w)
        if show:
            print(f"-> move={a['move']} open={a['open']}\n")
        w.step(a["move"], a["open"])
    if show:
        print(w.render())
        print("RESULT:", w.outcome, "ticks:", w.ticks, "\n" + "=" * 20)
    return w.outcome, w.ticks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["oracle", "random", "model", "human"], default="oracle")
    p.add_argument("--episodes", type=int, default=None, help="default 200, or 1 for --policy human")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--surprise", action="store_true", help="add an obstacle type never seen in training")
    p.add_argument("--show", action="store_true")
    p.add_argument("--url", default="http://127.0.0.1:8009")
    args = p.parse_args()

    if args.episodes is None:
        args.episodes = 1 if args.policy == "human" else 200
    if args.policy == "human":
        args.show = True
        make_policy = human_policy
        print("You are Y. Obstacles scroll UP one row per tick. Reach 1000 m or lower, then press o to open the parachute\n"
              "while the cell directly below you is empty. Hitting the ground (altitude 0) without opening loses.\n")
    elif args.policy == "oracle":
        make_policy = lambda seed: oracle
    elif args.policy == "random":
        make_policy = random_policy
    else:
        from model.client import model_policy
        shared = model_policy(args.url, show=args.show)
        make_policy = lambda seed: shared

    wins, ticks, reasons = 0, 0, {}
    for i in range(args.episodes):
        (result, reason), t = run_episode(make_policy, args.seed + i, args.surprise, args.show)
        wins += result == "win"
        ticks += t
        reasons[reason] = reasons.get(reason, 0) + 1
    print(f"policy={args.policy} surprise={args.surprise} episodes={args.episodes}")
    print(f"win_rate={wins / args.episodes:.3f} mean_ticks={ticks / args.episodes:.1f}")
    print("outcomes:", dict(sorted(reasons.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
