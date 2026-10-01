"""Score Gemini prompt variants directly against the oracle labels on held-out states.

python3 -m model.llm_eval --n 60 --prompts minimal,rules,procedure --model gemini-flash-lite-latest
"""
import argparse, json, statistics
from model.llm_policy import GeminiPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eval.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--model", default="gemini-flash-lite-latest")
    ap.add_argument("--prompts", default="minimal,rules,procedure,cot")
    ap.add_argument("--thinking", choices=["none", "minimal", "low"], default="none")
    ap.add_argument("--out", default="results/llm_prompt_eval.json")
    a = ap.parse_args()
    thinking = {"none": None, "minimal": {"thinkingLevel": "minimal"}, "low": {"thinkingLevel": "low"}}[a.thinking]
    rows = [json.loads(l) for l in open(a.data)][a.offset:a.offset + a.n]
    results = {}
    for name in a.prompts.split(","):
        p = GeminiPolicy(a.model, thinking=thinking, prompt=name)
        ok_m = ok_o = parsed = 0; lat = []; errs = 0
        for r in rows:
            d = p.decide(r["state"])
            lat.append(d["latency_ms"]); parsed += d["parsed"]; errs += bool(d["error"])
            ok_m += d["action"]["move"] == r["questions"]["move"]["label"]
            ok_o += d["action"]["open"] == bool(r["questions"]["open"]["label"])
        n = len(rows)
        results[name] = {"n": n, "move_acc": ok_m / n, "open_acc": ok_o / n, "parsed": parsed / n, "errors": errs,
                         "latency_ms_p50": statistics.median(lat), "served_by": d.get("model_version")}
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in results[name].items()}))
    json.dump({"model": a.model, "thinking": a.thinking, "results": results}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
