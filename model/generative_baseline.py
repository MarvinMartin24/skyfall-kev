"""Latency and accuracy of the *generative* way to answer the same questions.

Same family of backbone (Qwen3.5-0.8B, the chat-tuned sibling of the base Kev
adapts), but asked to generate the answer as text, which is then parsed.
This is the "what you would do with an ordinary LLM" baseline for the latency slide.

python -m model.generative_baseline --data data/eval.jsonl --n 100 --device mps
"""
import argparse, json, re, time, statistics

from data.questions import QUESTIONS

PROMPT = """You control the skydiver Y in this grid. Obstacles (any letter other than Y) scroll UP one row per tick.
{state}

Answer two questions and reply with ONLY a JSON object like {{"move": "left|right|stay", "open": true|false}}.
move: {move_q}
open: {open_q}
"""


def parse(text):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    mv = obj.get("move")
    op = obj.get("open")
    if mv not in ("left", "right", "stay") or not isinstance(op, bool):
        return None
    return {"move": mv, "open": op}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--data", default="data/eval.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--out", default="results/generative_baseline.json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if a.device != "cpu" else torch.float32).to(a.device).eval()

    rows = [json.loads(l) for _, l in zip(range(a.n + 1), open(a.data))]
    lat, ok_move, ok_open, parsed = [], 0, 0, 0
    for i, r in enumerate(rows):
        msgs = [{"role": "user", "content": PROMPT.format(
            state=r["state"], move_q=QUESTIONS["move"]["instructions"], open_q=QUESTIONS["open"]["instructions"])}]
        kw = {"add_generation_prompt": True, "return_tensors": "pt", "return_dict": True}
        try:
            enc = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            enc = tok.apply_chat_template(msgs, **kw)
        enc = {k: v.to(a.device) for k, v in enc.items()}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False)
        if a.device == "mps":
            torch.mps.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if i == 0:   # warm-up, not timed; show what it says
            print("warm-up output:", repr(text[:200]))
            continue
        lat.append(dt)
        ans = parse(text)
        if ans:
            parsed += 1
            ok_move += ans["move"] == r["questions"]["move"]["label"]
            ok_open += ans["open"] == bool(r["questions"]["open"]["label"])
    n = len(lat)
    res = {"model": a.model, "device": a.device, "n": n,
           "parsed_rate": round(parsed / n, 3),
           "move_acc": round(ok_move / n, 3), "open_acc": round(ok_open / n, 3),
           "latency_ms_mean": round(statistics.mean(lat), 1),
           "latency_ms_p50": round(statistics.median(lat), 1),
           "latency_ms_p95": round(sorted(lat)[int(0.95 * n) - 1], 1)}
    print(json.dumps(res, indent=1))
    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
