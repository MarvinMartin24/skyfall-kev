"""Learning curve: evaluate every snapshot the :8010 server serves (scripts/serve_latest.sh) on 300
held-out states, append to results/learning_curve.jsonl, and log to W&B as eval/* at the training step.

.venv-wandb/bin/python scripts/eval_snapshots.py --run skyfall-0.8b --port 8010 --n 300
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request

import wandb

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def served(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
            return json.load(r)["models"][0]["run"]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="skyfall-0.8b"); ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--total_steps", type=int, default=2500)
    ap.add_argument("--wandb_name", default=None)
    a = ap.parse_args()
    wb = wandb.init(project="skyfall-kev", name=a.wandb_name or f"{a.run}-eval-curve", resume="allow", config={"n_eval": a.n})
    out = os.path.join(HERE, "results", "learning_curve.jsonl")
    done = set(json.loads(l)["run_dir"] for l in open(out)) if os.path.exists(out) else set()
    while True:
        cur = served(a.port)
        if cur and cur not in done and os.path.exists(os.path.join(cur, "head.pt")):
            m = re.search(r"step-(\d+)$", cur)
            step = int(m.group(1)) if m else a.total_steps
            r = subprocess.run([sys.executable, "-m", "model.eval", "--url", f"http://127.0.0.1:{a.port}", "--n", str(a.n)],
                               capture_output=True, text=True, cwd=HERE)
            if r.returncode == 0 and served(a.port) == cur:          # the server did not switch mid-eval
                metrics = json.loads(r.stdout)
                metrics.update(step=step, run_dir=cur, time=time.strftime("%Y-%m-%d %H:%M:%S"))
                with open(out, "a") as f:
                    f.write(json.dumps(metrics) + "\n")
                wb.log({f"eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float)) and k != "step"}, step=step)
                print(f"step {step}: move {metrics['move_acc']:.3f} open {metrics['open_acc']:.3f} brier {metrics['move_brier']:.3f}/{metrics['open_brier']:.3f}", flush=True)
                done.add(cur)
                if step >= a.total_steps:
                    break
        time.sleep(60)
    wb.finish()


if __name__ == "__main__":
    main()
