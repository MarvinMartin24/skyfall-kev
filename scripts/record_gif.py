"""Record jumps as GIFs, rendered headlessly (no screen capture), one GIF per player.

Frame duration carries the real decision latency (clamped to 0.33–2.0 s), so a slow
decision maker *looks* slow. The last frame of each jump is held with the outcome.

.venv-wandb/bin/python scripts/record_gif.py --player oracle --seeds 7000,7001,7002 --out results/gifs/oracle.gif
.venv-wandb/bin/python scripts/record_gif.py --player kev:http://127.0.0.1:8010 --label "Kev step 300" ...
.venv-wandb/bin/python scripts/record_gif.py --player gemini --prompt cot ...
"""
import argparse, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from PIL import Image, ImageDraw, ImageFont   # noqa: E402

from game.sim import World, WIDTH, HEIGHT, PLAYER_ROW, EMPTY, PLAYER, PASSABLE, OPEN_ALT, START_ALT, ALT_STEP   # noqa: E402
from game.oracle import oracle   # noqa: E402

BG, FG, DIM = (18, 18, 24), (230, 230, 230), (110, 110, 120)
COL = {"Y": (90, 230, 110), "X": (255, 80, 80), "B": (240, 200, 60), "P": (220, 120, 230), "A": (255, 90, 90),
       "O": (90, 220, 230), "C": (245, 245, 245), ".": (70, 70, 80)}
NAME = {"B": "bird", "P": "skydiver", "A": "airplane", "O": "balloon", "C": "cloud (harmless)"}
GREEN, YELLOW, RED, BLUE, CYAN = (90, 230, 110), (240, 200, 60), (255, 80, 80), (100, 160, 255), (90, 220, 230)
CELL, PAD = 44, 24


def font(size):
    for path in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/SFNSMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F, FS, FB = font(22), font(17), font(26)
W_IMG, H_IMG = 980, 420


def served_name(url):
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=5) as r:
            return json.load(r)["models"][0]["run"].split("/")[-1]
    except Exception:
        return "?"


def render(w, info, banner=None):
    img = Image.new("RGB", (W_IMG, H_IMG), BG)
    d = ImageDraw.Draw(img)
    gx, gy = PAD, PAD + 36
    d.text((gx, PAD), "SKYFALL", font=FB, fill=FG)
    d.text((gx + 140, PAD + 6), f"seed {info['seed']}   tick {w.ticks}   {info['label']}", font=FS, fill=DIM)
    ground = PLAYER_ROW + 1 + w.alt // ALT_STEP
    for r in range(HEIGHT):
        for c in range(WIDTH):
            x, y = gx + c * CELL, gy + r * CELL
            is_ground = r >= ground
            d.rectangle([x, y, x + CELL - 2, y + CELL - 2], fill=(40, 90, 50) if is_ground else (28, 28, 36))
            ch = w.grid[r][c]
            if r == PLAYER_ROW and c == w.col:
                ch = PLAYER if w.grid[r][c] in PASSABLE else "X"
            if ch != EMPTY:
                d.text((x + 13, y + 8), ch, font=F, fill=COL.get(ch, FG))
            elif not is_ground:
                d.text((x + 17, y + 10), "·", font=F, fill=COL["."])
    hx, hy = gx + WIDTH * CELL + 40, gy
    low = w.alt <= OPEN_ALT
    d.text((hx, hy), f"altitude {w.alt:5d} m", font=F, fill=GREEN if low else BLUE)
    bw, filled = 260, int(260 * w.alt / START_ALT)
    d.rectangle([hx + 230, hy + 6, hx + 230 + bw, hy + 22], outline=DIM)
    d.rectangle([hx + 230, hy + 6, hx + 230 + filled, hy + 22], fill=GREEN if low else BLUE)
    d.text((hx, hy + 32), "OPEN ZONE (<= 1000 m)" if low else "open zone starts at 1000 m", font=FS, fill=GREEN if low else DIM)
    if ground < HEIGHT and w.alt > 0:
        d.text((hx + 250, hy + 32), f"GROUND in {w.alt} m", font=FS, fill=YELLOW)
    y = hy + 70
    d.text((hx, y), f"player: {info['who']}", font=FS, fill=FG); y += 26
    if info.get("last"):
        a = info["last"]; d.text((hx, y), f"last: {a['move']}" + (", open" if a["open"] else ""), font=FS, fill=FG); y += 26
    if info.get("probs"):
        pr = info["probs"]; y += 6
        for k in ("left", "right", "stay"):
            d.text((hx, y), f"P({k:<5})", font=FS, fill=FG)
            d.rectangle([hx + 95, y + 4, hx + 95 + 160, y + 16], outline=DIM)
            d.rectangle([hx + 95, y + 4, hx + 95 + int(160 * pr["move"][k]), y + 16], fill=CYAN)
            d.text((hx + 265, y), f"{pr['move'][k]:.2f}", font=FS, fill=FG); y += 22
        d.text((hx, y), "P(open )", font=FS, fill=FG)
        d.rectangle([hx + 95, y + 4, hx + 95 + 160, y + 16], outline=DIM)
        d.rectangle([hx + 95, y + 4, hx + 95 + int(160 * pr["open"]), y + 16], fill=GREEN)
        d.text((hx + 265, y), f"{pr['open']:.2f}", font=FS, fill=FG); y += 28
    if info.get("latency_ms") is not None:
        d.text((hx, y), f"decision took {info['latency_ms']:6.0f} ms   fell {info['latency_ms'] / 100:4.1f} rows while deciding", font=FS, fill=YELLOW); y += 24
    if info.get("tokens"):
        d.text((hx, y), info["tokens"], font=FS, fill=DIM); y += 24
    if info.get("analysis"):
        txt = info["analysis"]
        for i in range(0, min(len(txt), 240), 60):
            d.text((hx, y), txt[i:i + 60], font=FS, fill=(200, 200, 160)); y += 20
    legend = "   ".join(f"{k} {NAME[k]}" for k in ("B", "P", "A", "C")) + ("   O balloon (never in training)" if w.surprise else "")
    d.text((gx, gy + HEIGHT * CELL + 12), legend, font=FS, fill=DIM)
    if banner:
        d.text((gx, gy + HEIGHT * CELL + 40), banner[0], font=FB, fill=banner[1])
    return img


def outcome_banner(outcome):
    res, reason = outcome
    if res == "win":
        return ("SAFE LANDING", GREEN)
    why = {"ground": "hit the ground without opening", "opened_too_high": "opened above 1000 m"}.get(reason)
    if why is None:
        kind, _, sym = reason.rpartition("_")
        why = ("canopy hit a " if kind == "canopy_hit" else "hit a ") + NAME.get(sym, sym)
    return ("LOST: " + why, RED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True, help="oracle | kev:<url> | gemini")
    ap.add_argument("--label", default=None)
    ap.add_argument("--prompt", default="cot"); ap.add_argument("--gemini_model", default="gemini-flash-lite-latest")
    ap.add_argument("--seeds", default="7000,7001,7002")
    ap.add_argument("--surprise", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_ms", type=int, default=333); ap.add_argument("--max_ms", type=int, default=2000)
    a = ap.parse_args()

    if a.player == "oracle":
        who = a.label or "the rule (oracle)"
        decide = lambda w: {"action": oracle(w), "latency_ms": None}
    elif a.player.startswith("kev:"):
        from model.client import ask, probs
        url = a.player[4:]
        who = a.label or f"Kev ({served_name(url)})"
        def decide(w):
            r = ask(url, w.render()); pr = probs(r)
            return {"action": {"move": max(pr["move"], key=pr["move"].get), "open": pr["open"] >= 0.5}, "probs": pr,
                    "latency_ms": r.get("latency_ms", r["_client_ms"]), "tokens": f"tokens {r.get('usage', {}).get('input_tokens', '?')} in, one forward pass"}
    else:
        from model.llm_policy import GeminiPolicy
        gp = GeminiPolicy(a.gemini_model, thinking=None, prompt=a.prompt)
        who = a.label or f"Gemini ({a.gemini_model}, {a.prompt} prompt)"
        def decide(w):
            d = gp.decide(w.render())
            an = ""
            try:
                an = json.loads(d["raw"]).get("analysis", "") if d["raw"].strip().startswith("{") else ""
            except Exception:
                pass
            return {"action": d["action"], "latency_ms": d["wall_ms"], "analysis": an,
                    "tokens": f"tokens {d['input_tokens']} in, {d['output_tokens']} out" + (f", served by {d['model_version']}" if d.get("model_version") else "")}

    frames, durations = [], []
    for seed in [int(s) for s in a.seeds.split(",")]:
        w = World(seed=seed, surprise=a.surprise)
        info = {"seed": seed, "label": a.label or "", "who": who, "last": None}
        while not w.done:
            d = decide(w)
            info.update(last=d["action"], probs=d.get("probs"), latency_ms=d.get("latency_ms"), tokens=d.get("tokens"), analysis=d.get("analysis"))
            frames.append(render(w, info))
            lat = d.get("latency_ms") or 0
            durations.append(int(min(max(lat, a.min_ms), a.max_ms)))
            w.step(d["action"]["move"], d["action"]["open"])
        frames.append(render(w, info, outcome_banner(w.outcome))); durations.append(1800)
        print(f"{who}: seed {seed} -> {w.outcome[0]} ({w.outcome[1]}) in {w.ticks} decisions", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
    print("wrote", a.out, f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
