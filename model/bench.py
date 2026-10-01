"""The race: the same seeded jumps played by several decision makers, with per-decision
latency, tokens, cost, and agreement with the oracle. Produces results/race.json and results/race.md.

    python3 -m model.bench --players oracle,kev-base:http://127.0.0.1:8009,kev-ft:http://127.0.0.1:8011,gemini --episodes 20

Player specs:  oracle | kev-<label>:<url> | gemini[:<model>] | qwen-gen (local generative baseline, slow)
"Time as gravity": seconds per jump = sum of the decision latencies in that jump, i.e. how long the
skydiver is actually in the air if the sky waits for the decision maker.
"""
import argparse
import json
import os
import statistics
import time

from game.sim import World
from game.oracle import oracle

TICK_MS = 100   # a row scrolls every 100 ms in "real time": rows fallen while thinking = latency / TICK_MS


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0


class OraclePlayer:
    name, price = "oracle", (0.0, 0.0)

    def decide(self, state_world):
        t0 = time.perf_counter()
        a = oracle(state_world)
        return {"action": a, "latency_ms": (time.perf_counter() - t0) * 1000, "input_tokens": 0, "output_tokens": 0, "parsed": True}


class KevPlayer:
    def __init__(self, label, url, price_in=0.0, price_out=0.0):
        from model.client import ask, probs
        self.ask, self.probs, self.url = ask, probs, url
        self.name, self.price = f"kev-{label}", (price_in, price_out)

    def decide(self, w):
        r = self.ask(self.url, w.render())
        pr = self.probs(r)
        a = {"move": max(pr["move"], key=pr["move"].get), "open": pr["open"] >= 0.5}
        u = r.get("usage", {})
        return {"action": a, "latency_ms": r.get("latency_ms", r["_client_ms"]), "client_ms": r["_client_ms"],
                "input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0), "parsed": True, "probs": pr}


class GeminiPlayer:
    def __init__(self, model, price_in, price_out, thinking=None, prompt="procedure"):
        from model.llm_policy import GeminiPolicy
        self.p = GeminiPolicy(model, thinking=thinking, prompt=prompt)
        self.name, self.price = self.p.name, (price_in, price_out)

    def decide(self, w):
        return self.p.decide(w.render())


def make_player(spec, args):
    from model.llm_policy import PRICES
    if spec == "oracle":
        return OraclePlayer()
    if spec.startswith("kev-"):
        label, url = spec[4:].split(":", 1)
        return KevPlayer(label, url, *PRICES["jev"]) if args.kev_price == "jev" else KevPlayer(label, url)
    if spec.startswith("gemini"):
        model = spec.split(":", 1)[1] if ":" in spec else "gemini-3.8-flash"
        pin, pout = (args.price_in, args.price_out) if args.price_in is not None else PRICES.get(model, (0.0, 0.0))
        thinking = {"none": None, "budget0": {"thinkingBudget": 0}, "minimal": {"thinkingLevel": "minimal"}, "low": {"thinkingLevel": "low"}}[args.gemini_thinking]
        return GeminiPlayer(model, pin, pout, thinking, args.gemini_prompt)
    raise SystemExit(f"unknown player {spec}")


def run(player, seeds, surprise, log):
    rows, jumps = [], []
    for seed in seeds:
        w = World(seed=seed, surprise=surprise)
        t_air = 0.0
        while not w.done:
            truth = oracle(w)
            d = player.decide(w)
            a = d["action"]
            if d.get("model_version") and not getattr(player, "model_version", None):
                player.model_version = d["model_version"]
            rows.append({"seed": seed, "tick": w.ticks, "alt": w.alt, "latency_ms": d["latency_ms"],
                         "in": d.get("input_tokens", 0), "out": d.get("output_tokens", 0), "think": d.get("thinking_tokens", 0),
                         "parsed": d.get("parsed", True), "retries": d.get("retries", 0), "wall_ms": d.get("wall_ms", d["latency_ms"]),
                         "move_ok": a["move"] == truth["move"], "open_ok": a["open"] == truth["open"]})
            t_air += d["latency_ms"]
            w.step(a["move"], a["open"])
        jumps.append({"seed": seed, "result": w.outcome[0], "reason": w.outcome[1], "ticks": w.ticks, "air_s": t_air / 1000})
        log(f"  {player.name} seed {seed}: {w.outcome[0]} ({w.outcome[1]}) in {w.ticks} decisions, {t_air/1000:.1f}s in the air")
    lat = [r["latency_ms"] for r in rows]
    n = len(rows)
    pin, pout = player.price
    cost_per_decision = (statistics.mean(r["in"] for r in rows) * pin + statistics.mean(r["out"] + r["think"] for r in rows) * pout) / 1e6
    return {"player": player.name, "model_version": getattr(player, "model_version", None), "decisions": n, "jumps": len(jumps),
            "win_rate": sum(j["result"] == "win" for j in jumps) / len(jumps),
            "move_acc": sum(r["move_ok"] for r in rows) / n, "open_acc": sum(r["open_ok"] for r in rows) / n,
            "parsed_rate": sum(r["parsed"] for r in rows) / n, "retries_per_decision": sum(r["retries"] for r in rows) / n,
            "wall_ms_mean": statistics.mean(r["wall_ms"] for r in rows),
            "latency_ms_mean": statistics.mean(lat), "latency_ms_p50": statistics.median(lat), "latency_ms_p95": pct(lat, 0.95),
            "seconds_per_jump": statistics.mean(j["air_s"] for j in jumps),
            "rows_fallen_while_thinking": statistics.mean(lat) / TICK_MS,
            "tokens_in_per_decision": statistics.mean(r["in"] for r in rows), "tokens_out_per_decision": statistics.mean(r["out"] + r["think"] for r in rows),
            "usd_per_1000_decisions": cost_per_decision * 1000, "usd_per_jump": cost_per_decision * n / len(jumps),
            "price_per_mtok": {"in": pin, "out": pout}, "jumps_detail": jumps}


def md_table(results):
    cols = [("player", "{}"), ("win_rate", "{:.2f}"), ("move_acc", "{:.2f}"), ("open_acc", "{:.2f}"),
            ("latency_ms_p50", "{:.0f}"), ("latency_ms_p95", "{:.0f}"), ("seconds_per_jump", "{:.1f}"),
            ("rows_fallen_while_thinking", "{:.1f}"), ("tokens_in_per_decision", "{:.0f}"), ("tokens_out_per_decision", "{:.0f}"),
            ("usd_per_1000_decisions", "{:.4f}")]
    head = "| " + " | ".join(c for c, _ in cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(f.format(r[c]) for c, f in cols) + " |\n" for r in results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", default="oracle,kev-base:http://127.0.0.1:8009")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--surprise", action="store_true")
    ap.add_argument("--kev_price", choices=["local", "jev"], default="jev", help="price Kev tokens at 0 (local) or at Jev's list price")
    ap.add_argument("--price_in", type=float, default=None); ap.add_argument("--price_out", type=float, default=None)
    ap.add_argument("--gemini_thinking", choices=["none", "budget0", "minimal", "low"], default="none", help="thinkingConfig sent to Gemini")
    ap.add_argument("--gemini_prompt", choices=["minimal", "rules", "procedure", "cot"], default="procedure")
    ap.add_argument("--out", default="results/race")
    args = ap.parse_args()
    seeds = list(range(args.seed, args.seed + args.episodes))
    results = []
    for spec in args.players.split(","):
        player = make_player(spec, args)
        print(f"== {player.name}: {len(seeds)} jumps, seeds {seeds[0]}..{seeds[-1]}" + (" +balloons" if args.surprise else ""))
        results.append(run(player, seeds, args.surprise, print))
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in results[-1].items() if k != "jumps_detail"}, indent=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"args": vars(args), "tick_ms": TICK_MS, "results": results}, open(args.out + ".json", "w"), indent=1)
    with open(args.out + ".md", "w") as f:
        f.write(f"# The race: {args.episodes} seeded jumps per player (seeds {seeds[0]}–{seeds[-1]}" + (", with balloons" if args.surprise else "") + ")\n\n")
        f.write("Latency is per decision (both questions). `seconds_per_jump` is the time the skydiver spends in the air if the sky waits for each decision. "
                f"`rows_fallen_while_thinking` assumes a row every {TICK_MS} ms. Kev tokens are priced at {'Jev list price ($0.042/M input, output free, as cited in Simon Willison\'s Sep 2026 write-up)' if args.kev_price == 'jev' else '$0 (local)'}; "
                "Gemini at its list price. Accuracy is agreement with the oracle on the states actually visited.\n\n")
        f.write(md_table(results))
    print("wrote", args.out + ".md")


if __name__ == "__main__":
    main()
