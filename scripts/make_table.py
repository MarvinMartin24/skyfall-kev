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
           "| oracle (the rule) | 1.000 | 1.000 | 0 | 0 | 0.95 (500 jumps) |", "",
           "Zero-shot numbers: full `kev.benchmark` on 2,000 states. Fine-tuned: 300 states via `model/eval.py` (final: see `runs/ft-eval/report.json` if present).", "",
           "## Learning curve", "", "| step | records | move acc | open acc | move Brier | win rate (50 jumps) |", "|---|---|---|---|---|---|"]
    wr_by = {r["step"]: r for r in wr}
    for r in lc:
        w = wr_by.get(r["step"])
        out.append(f"| {r['step']} | {r['step'] * 8:,} | {r['move_acc']:.3f} | {r['open_acc']:.3f} | {r['move_brier']:.3f} | {w['win_rate_50']:.2f} |" if w else
                   f"| {r['step']} | {r['step'] * 8:,} | {r['move_acc']:.3f} | {r['open_acc']:.3f} | {r['move_brier']:.3f} | |")
    out += ["", "![learning curve](learning_curve.png)", ""]
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
