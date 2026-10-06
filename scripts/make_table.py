"""Assemble results/table.md from the saved results (baseline, curves, races, latency)."""
import glob, json, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(HERE, "results", *p)


def last(name, key="step"):
    rows = {}
    for l in open(R(name)):
        r = json.loads(l); rows[r[key]] = r
    return [rows[k] for k in sorted(rows)]


def main():
    base = json.load(open(R("baseline.json")))
    lc, wr = last("learning_curve.jsonl"), last("winrate_curve.jsonl")
    final_lc, final_wr = lc[-1], wr[-1]
    out = ["# Results", "", "## Before and after (held-out states; live jumps on seeds 8000–8049)", "",
           "| model | move acc | open acc | move Brier | open Brier | win rate |", "|---|---|---|---|---|---|",
           f"| released Kev-0.8B, zero-shot | {base['move_acc']:.3f} | {base['open_acc']:.3f} | {base['move_brier']:.3f} | {base['open_brier']:.3f} | 0.00 (0/200) |",
           f"| fine-tuned, step {final_lc['step']} | {final_lc['move_acc']:.3f} | {final_lc['open_acc']:.3f} | {final_lc['move_brier']:.3f} | {final_lc['open_brier']:.3f} | {final_wr['win_rate_50']:.2f} (50 jumps) |",
           "| oracle (the rule) | 1.000 | 1.000 | 0 | 0 | 0.94 (500 jumps) |", "",
           "Zero-shot numbers: full `kev.benchmark` on 2,000 states. Fine-tuned: 300 states via `model/eval.py` (final: see `runs/ft-eval/report.json` if present).", "",
           "## Learning curve", "", "| step | records | move acc | open acc | move Brier | win rate (50 jumps) |", "|---|---|---|---|---|---|"]
    wr_by = {r["step"]: r for r in wr}
    for r in lc:
        w = wr_by.get(r["step"])
        out.append(f"| {r['step']} | {r['step'] * 8:,} | {r['move_acc']:.3f} | {r['open_acc']:.3f} | {r['move_brier']:.3f} | {w['win_rate_50']:.2f} |" if w else
                   f"| {r['step']} | {r['step'] * 8:,} | {r['move_acc']:.3f} | {r['open_acc']:.3f} | {r['move_brier']:.3f} | |")
    out += ["", "![learning curve](learning_curve.png)", ""]
    if os.path.exists(R("final_benchmark.json")):
        fb = json.load(open(R("final_benchmark.json")))
        out += ["## Final model, full benchmark (`kev.benchmark`, 2,000 states / 4,000 questions)", "",
                "| | move | open | overall |", "|---|---|---|---|",
                f"| accuracy | {fb['move_acc']:.4f} | {fb['open_acc']:.4f} | {fb['overall_acc']:.4f} |",
                f"| Brier | {fb['move_brier']:.4f} | {fb['open_brier']:.4f} | {fb['overall_brier']:.4f} |",
                f"| NLL | {fb['move_nll']:.4f} | {fb['open_nll']:.4f} | |",
                f"| ECE (overall) | | | {fb['ece']:.4f} |", "",
                f"Released model on the same 2,000 states: move {base['move_acc']:.3f} / Brier {base['move_brier']:.3f}, open {base['open_acc']:.3f} / Brier {base['open_brier']:.3f}, ECE {base['ece']:.3f}.", ""]
    lat = {}
    for name, path in (("final", "final-eval-500.json"), ("released", "released-eval-500.json")):
        if os.path.exists(R(path)):
            lat[name] = json.load(open(R(path)))
    if lat:
        out += ["## Latency per decision (both questions), Kev served by MLX on the M2 Max, GPU otherwise idle, 500 states", "",
                "| model | mean ms | move acc | open acc |", "|---|---|---|---|"]
        out += [f"| {k} | {v['mean_latency_ms']:.0f} | {v['move_acc']:.3f} | {v['open_acc']:.3f} |" for k, v in lat.items()]
        out += ["", "Gemini numbers (free tier, bare and chain-of-thought prompts) are in the race files below and in STORY.md §7.", ""]
    if os.path.exists(R("jev-eval-2000.json")):
        j = json.load(open(R("jev-eval-2000.json")))
        out += [f"## Jev (hosted, {', '.join(j['served_by'])}), zero-shot, same {j['n_states']:,} held-out states", "",
                "| | move | open | overall |", "|---|---|---|---|",
                f"| accuracy | {j['move_acc']:.4f} | {j['open_acc']:.4f} | |",
                f"| Brier | {j['move_brier']:.4f} | {j['open_brier']:.4f} | {j['overall_brier']:.4f} |",
                f"| ECE | | | {j['ece']:.4f} |", "",
                f"Latency per decision: median {j['latency_ms_median']:.0f} ms, p95 {j['latency_ms_p95']:.0f} ms (remote API, network included). "
                f"Input tokens per request: {j['input_tokens'] / j['n_states']:.0f} (Jev's tokenizer). Run by `scripts/jev_test.py eval`; "
                f"estimated spend ${j['est_usd']:.3f}.", ""]
    if os.path.exists(R("jev-race-6000.json")):
        j = json.load(open(R("jev-race-6000.json")))
        out += [f"## Jev live race ({j['episodes']} jumps, seeds {j['seeds']})", "",
                "| wins | win rate | decisions | move agreement | open agreement | latency p50 / p95 ms |", "|---|---|---|---|---|---|",
                f"| {j['wins']} / {j['episodes']} | {j['win_rate']:.2f} | {j['decisions']} | {j['move_agreement']:.2f} | {j['open_agreement']:.2f} | {j['latency_ms_median']:.0f} / {j['latency_ms_p95']:.0f} |", ""]
    for path in sorted(glob.glob(R("race-*.md"))):
        out += [f"## {os.path.basename(path)}", "", open(path).read().split("\n", 2)[2], ""]
    if os.path.exists(R("llm_prompt_eval.json")):
        d = json.load(open(R("llm_prompt_eval.json")))
        out += [f"## Gemini prompt variants ({d['model']}, thinking {d['thinking']}, 60 held-out states)", "", "| prompt | move acc | open acc | p50 ms |", "|---|---|---|---|"]
        out += [f"| {k} | {v['move_acc']:.2f} | {v['open_acc']:.2f} | {v['latency_ms_p50']:.0f} |" for k, v in d["results"].items()]
        for extra in sorted(glob.glob(R("llm_eval_*.json"))):
            e = json.load(open(extra))
            for k, v in e["results"].items():
                out.append(f"| {k} ({e['model']}, thinking {e['thinking']}) | {v['move_acc']:.2f} | {v['open_acc']:.2f} | {v['latency_ms_p50']:.0f} |")
        out.append("")
    open(R("table.md"), "w").write("\n".join(out)); print("wrote results/table.md")


if __name__ == "__main__":
    main()
