"""Mirror a kev.train log to Weights & Biases without touching the training process.

kev.train has no W&B hook, so this tails its log, parses the progress lines
    ep0 step 30/2500 loss 0.688 kl 0.000 anchor 0.000 0.817s/rec
and logs them. When the run finishes it attaches training_metrics.json and
training_config.json and writes the final numbers to the run summary.

python scripts/wandb_tail.py --log runs/skyfall-0.8b.log --run runs/skyfall-0.8b --name skyfall-0.8b-mps
"""
import argparse, json, os, re, subprocess, time

import wandb

LINE = re.compile(r"ep(\d+) step (\d+)/(\d+) loss ([\d.]+) kl ([\d.]+) anchor ([\d.]+) ([\d.]+)s/rec")


def alive(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--run", required=True, help="kev.train --out directory (for the final json files)")
    ap.add_argument("--project", default="skyfall-kev")
    ap.add_argument("--name", default=None)
    ap.add_argument("--poll", type=float, default=15)
    ap.add_argument("--config", default="{}", help="JSON of the training args, attached as run config")
    a = ap.parse_args()

    run = wandb.init(project=a.project, name=a.name or os.path.basename(a.run), config=json.loads(a.config), resume="allow")
    recs_per_step = run.config.get("batch", 4) * run.config.get("accum", 2)
    pos, last_step, done, t0 = 0, 0, False, time.time()
    while not done:
        with open(a.log, errors="replace") as f:
            f.seek(pos); chunk = f.read(); pos = f.tell()
        for line in chunk.replace("\r", "\n").split("\n"):
            m = LINE.search(line)
            if m:
                ep, step, steps, loss, kl, anchor, s_rec = m.groups()
                step, steps, s_rec = int(step), int(steps), float(s_rec)
                if step <= last_step:
                    continue
                last_step = step
                wandb.log({"train/loss": float(loss), "train/epoch": int(ep), "train/progress": step / steps,
                           "perf/sec_per_record": s_rec, "perf/records_per_sec": 1 / s_rec,
                           "perf/eta_min": (steps - step) * recs_per_step * s_rec / 60}, step=step)
            elif line.startswith("saved "):
                done = True
            elif "Traceback" in line or "Error" in line:
                wandb.log({"train/error": 1}, step=max(last_step, 1)); done = True
        if not done and not alive("kev.train"):
            done = True
        if not done:
            time.sleep(a.poll)
    for name in ("training_metrics.json", "training_config.json"):
        path = os.path.join(a.run, name)
        if os.path.exists(path):
            wandb.save(path, base_path=a.run, policy="now")
            if name == "training_metrics.json":
                d = json.load(open(path))
                run.summary.update({f"final/{k}": v for k, v in d.items() if isinstance(v, (int, float))})
    run.summary["tail_wallclock_min"] = round((time.time() - t0) / 60, 1)
    run.finish()


if __name__ == "__main__":
    main()
