"""Skyfall terminal UI: redraws in place, single-key controls, live probability bars for the model.

  python3 skyfall.py                                   # you play
  python3 skyfall.py --policy oracle --fps 4           # watch the rule
  python3 skyfall.py --policy model --fps 2 [--surprise] [--url http://127.0.0.1:8009]
  python3 skyfall.py --policy gemini --episodes 10 [--gemini_model gemini-flash-lite-latest]   # an LLM call per tick

Keys: a / ←  left     d / →  right     s / enter / ↓  stay     space / o  open
      h  toggle the rule's hint     r  replay (next seed)     q  quit
"""
import argparse
import select
import sys
import termios
import time
import tty

from game.sim import World, WIDTH, HEIGHT, PLAYER_ROW, EMPTY, PLAYER, DEAD, PASSABLE, CLOUD, OPEN_ALT, START_ALT, ALT_STEP
from game.oracle import oracle

E = "\x1b["
RESET = E + "0m"
BOLD = E + "1m"
DIM = E + "2m"
COLOR = {"Y": E + "1;32m", "X": E + "1;31m", "B": E + "33m", "P": E + "35m", "A": E + "31m", "O": E + "36m", "C": E + "1;37m"}
NAME = {"B": "bird", "P": "skydiver", "A": "airplane", "O": "balloon", "C": "cloud (harmless)"}
EMOJI = {"Y": "🧍", "X": "💥", "B": "🐦", "P": "🪂", "A": "🛫", "O": "🎈", "C": "☁️"}
KEYMAP = {"a": "left", "D": "left", "d": "right", "C": "right", "s": "stay", "\n": "stay", "\r": "stay", "B": "stay",
          " ": "open", "o": "open"}


class Keys:
    """cbreak keyboard input on an alternate screen with the cursor hidden."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        sys.stdout.write(E + "?1049h" + E + "?25l")
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        sys.stdout.write(E + "?25h" + E + "?1049l")
        sys.stdout.flush()

    def get(self, timeout=None):
        if timeout is not None:
            r, _, _ = select.select([sys.stdin], [], [], timeout)
            if not r:
                return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":                                  # arrow keys arrive as ESC [ A-D
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r and sys.stdin.read(1) == "[":
                return sys.stdin.read(1)
            return "esc"
        return ch


GROUND_BG = E + "42m"      # green background: the ground, drawn under whatever is in those rows


def cell(ch, emoji, ground=False):
    if emoji:
        if ground:
            return "🟩 " if ch == EMPTY else EMOJI[ch] + " "
        return (EMOJI[ch] + " ") if ch in EMOJI else " · "
    if ground:
        return GROUND_BG + (E + "30m" + "▒▒▒" if ch == EMPTY else E + "30;1m" + f" {ch} ") + RESET
    if ch == EMPTY:
        return DIM + " · " + RESET
    return COLOR.get(ch, "") + BOLD + f" {ch} " + RESET


def ground_row(w):
    """First grid row that is ground (visual only; the model never sees this). Each row below the
    skydiver is 100 m lower, so the ground enters the bottom row at 400 m and reaches the cell
    under Y at 0 m, exactly when the jump ends."""
    r = PLAYER_ROW + 1 + w.alt // ALT_STEP
    return r if r < HEIGHT else None


def bar(p, n=12, color=""):
    k = int(round(p * n))
    return color + "█" * k + RESET + DIM + "░" * (n - k) + RESET + f" {p:4.2f}"


def frame(w, st, emoji):
    inner = 3 * WIDTH
    rows = []
    g = ground_row(w)
    for r, row in enumerate(w.grid):
        cells = list(row)
        if r == PLAYER_ROW:
            cells[w.col] = PLAYER if row[w.col] in PASSABLE else DEAD
        rows.append("│" + "".join(cell(c, emoji, ground=g is not None and r >= g) for c in cells) + "│")
    grid = ["┌" + "─" * inner + "┐"] + rows + ["└" + "─" * inner + "┘"]

    low = w.alt <= OPEN_ALT
    alt_col = E + "1;32m" if low else E + "1;34m"
    hud = [f"{BOLD}SKYFALL{RESET}  seed {st['seed']}   tick {w.ticks:2d}   wins {st['wins']}/{st['n']}",
           f"altitude {alt_col}{w.alt:5d} m{RESET}  {bar(w.alt / START_ALT, 15, alt_col).rsplit(' ', 1)[0]}",
           (E + "1;32m" + "OPEN ZONE: at or below 1000 m" + RESET) if low else (DIM + f"open zone starts at {OPEN_ALT} m" + RESET),
           (E + "1;33m" + f"GROUND in {w.alt} m ({w.alt // ALT_STEP} ticks)" + RESET) if g is not None and w.alt > 0 else "",
           f"player: {st['who']}"]
    if st.get("last"):
        a = st["last"]
        hud.append(f"last: {a['move']}" + (", open" if a["open"] else ""))
    if st.get("hint"):
        h = st["hint"]
        hud.append(f"{DIM}rule says:{RESET} {h['move']}" + (", open" if h["open"] else ""))
    if st.get("probs"):
        pr = st["probs"]
        hud.append("")
        for k in ("left", "right", "stay"):
            hud.append(f"P({k:<5}) {bar(pr['move'][k], 12, E + '36m')}")
        hud.append(f"P(open ) {bar(pr['open'], 12, E + '32m')}")
    if st.get("llm") is not None:
        L = st["llm"]
        hud.append("")
        if L.get("status"):
            hud.append(E + "1;33m" + L["status"] + RESET)
        if L.get("latency_ms") is not None:
            hud.append(f"last call {L['latency_ms']:6.0f} ms   fell {L['latency_ms'] / 100:4.1f} rows while thinking (100 ms/row)")
            hud.append(f"tokens {L['in']}→{L['out']}" + (f"+{L['think']} thinking" if L.get("think") else "") + (f"   retries {L['retries']}" if L.get("retries") else ""))
        hud.append(f"served by {L.get('model_version') or '?'}   cost so far ${L['cost']:.4f} (list price)   {L['calls']} calls")
    if st.get("banner"):
        hud += ["", st["banner"]]

    lines = []
    for i in range(max(len(grid), len(hud))):
        left = grid[i] if i < len(grid) else " " * (inner + 2)
        lines.append(left + "   " + (hud[i] if i < len(hud) else ""))
    legend = "   ".join(f"{COLOR[c]}{c}{RESET} {NAME[c]}" for c in ("B", "P", "A", "C")) + ("   " + COLOR["O"] + "O" + RESET + " balloon (never seen in training)" if w.surprise else "") + "   " + GROUND_BG + E + "30m" + "▒" + RESET + " ground (visual only, 100 m per row)"
    keys = "a/← left   d/→ right   enter stay   space open   h hint   r replay   q quit" if st["who"] == "you" else "q quit   r next seed"
    return "\n".join(["", *lines, "", legend, DIM + keys + RESET, ""])


def draw(w, st, emoji):
    sys.stdout.write(E + "H" + E + "J" + frame(w, st, emoji))
    sys.stdout.flush()


def banner_for(outcome):
    result, reason = outcome
    if result == "win":
        return E + "1;32m" + "SAFE LANDING  ✓" + RESET
    why = {"ground": "hit the ground without opening", "opened_too_high": "opened above 1000 m"}.get(reason)
    if why is None:
        kind, _, sym = reason.rpartition("_")
        why = ("canopy hit a " if kind == "canopy_hit" else "hit a ") + NAME.get(sym, sym)
    return E + "1;31m" + f"LOST: {why}" + RESET


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=["human", "oracle", "model", "gemini"], default="human")
    ap.add_argument("--gemini_model", default="gemini-flash-lite-latest")
    ap.add_argument("--gemini_prompt", choices=["minimal", "rules", "procedure", "cot"], default="procedure")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=0, help="auto policies: stop after N jumps (0 = until q)")
    ap.add_argument("--fps", type=float, default=3.0, help="auto policies: ticks per second")
    ap.add_argument("--surprise", action="store_true", help="add the balloon O, never seen in training")
    ap.add_argument("--hint", action="store_true", help="show what the rule would do")
    ap.add_argument("--emoji", action="store_true", help="emoji glyphs (needs a terminal that renders them 2 columns wide)")
    ap.add_argument("--url", default="http://127.0.0.1:8009")
    args = ap.parse_args()
    if not sys.stdin.isatty():
        sys.exit("skyfall.py needs a terminal; for scripted runs use: python3 -m game.play")

    if args.policy == "model":
        from model.client import ask, probs
    gp = None
    if args.policy == "gemini":
        from model.llm_policy import GeminiPolicy, PRICES
        gp = GeminiPolicy(args.gemini_model, thinking=None, prompt=args.gemini_prompt)
        price = PRICES.get(args.gemini_model, (0.0, 0.0))
    who = {"human": "you", "oracle": "the rule (oracle)", "model": f"Kev @ {args.url}", "gemini": f"Gemini ({args.gemini_model}), one API call per tick"}[args.policy]
    st = {"seed": args.seed, "wins": 0, "n": 0, "who": who, "hint": None, "probs": None, "last": None, "banner": None,
          "llm": {"cost": 0.0, "calls": 0} if gp else None}
    hint_on = args.hint
    with Keys() as keys:
        while True:
            w = World(seed=st["seed"], surprise=args.surprise)
            st.update(last=None, probs=None, banner=None)
            quit_now = False
            while not w.done:
                st["hint"] = oracle(w) if hint_on else None
                if args.policy == "model":
                    st["probs"] = probs(ask(args.url, w.render()))
                if gp:
                    st["llm"]["status"] = "Gemini is thinking..."
                draw(w, st, args.emoji)
                if gp:
                    def on_retry(attempt, wait, err, _w=w):
                        code = "rate limited (429)" if "429" in (err or "") else "overloaded (503)" if "503" in (err or "") else (err or "")[:30]
                        st["llm"]["status"] = f"API {code}: retry {attempt} in {wait:.0f}s  (free tier)"
                        draw(_w, st, args.emoji)
                    d = gp.decide(w.render(), on_retry=on_retry)
                    L = st["llm"]
                    L.update(status=None, latency_ms=d["latency_ms"], **{"in": d["input_tokens"], "out": d["output_tokens"]},
                             think=d["thinking_tokens"], retries=d["retries"], model_version=d.get("model_version"), calls=L["calls"] + 1,
                             cost=L["cost"] + (d["input_tokens"] * price[0] + (d["output_tokens"] + d["thinking_tokens"]) * price[1]) / 1e6)
                    if d["error"] and not d["parsed"]:
                        L["status"] = "API error, defaulted to stay: " + d["error"][:50]
                    draw(w, st, args.emoji)
                if args.policy == "human":
                    while True:
                        k = keys.get()
                        if k == "q":
                            quit_now = True; break
                        if k == "h":
                            hint_on = not hint_on; st["hint"] = oracle(w) if hint_on else None; draw(w, st, args.emoji); continue
                        if k in KEYMAP:
                            act = KEYMAP[k]; break
                    if quit_now:
                        break
                    a = {"move": "stay", "open": True} if act == "open" else {"move": act, "open": False}
                else:
                    if args.policy == "oracle":
                        a = oracle(w)
                    elif gp:
                        a = d["action"]
                    else:
                        pr = st["probs"]
                        a = {"move": max(pr["move"], key=pr["move"].get), "open": pr["open"] >= 0.5}
                    k = keys.get(timeout=0.05 if gp else 1.0 / args.fps)
                    if k == "q":
                        quit_now = True; break
                w.step(a["move"], a["open"])
                st["last"] = a
            if quit_now:
                return
            st["n"] += 1; st["wins"] += w.outcome[0] == "win"
            st["banner"] = banner_for(w.outcome) + (DIM + "   r = next jump, q = quit" + RESET if args.policy == "human" or (args.episodes and st["n"] >= args.episodes) else "")
            st["hint"] = None
            draw(w, st, args.emoji)
            if args.policy == "human" or (args.episodes and st["n"] >= args.episodes):
                while True:
                    k = keys.get()
                    if k == "q":
                        return
                    if k == "r":
                        break
            else:
                k = keys.get(timeout=1.5)
                if k == "q":
                    return
            st["seed"] += 1


if __name__ == "__main__":
    main()
