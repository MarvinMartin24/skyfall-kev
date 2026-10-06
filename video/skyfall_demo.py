"""Skyfall: fine-tuning a Kev decision model on one Mac, and how it compares (Manim Community, ~4 min).

GIFs are NOT drawn by Manim. Each GIF gets a dark placeholder slot; the slot windows are written to
out/gif_slots.json and compose_gifs.py overlays the real GIFs (results/gifs/*.gif) with ffmpeg so they play
at full speed. Curves are read from results/learning_curve.jsonl and results/winrate_curve.jsonl.

    ./video/render.sh            # 1080p, PACE=1.5, composes the GIFs -> video/out/SkyfallDemo_final.mp4
    PACE=2 ./video/render.sh     # slower still (more room to speak)

Every number on the slides comes from results/*.json, STORY.md or DEMO.md; each slide states its sample size.
"""
import json
from pathlib import Path

from common import *

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = Path(__file__).resolve().parent / "out"
SLOTS_FILE = OUT / "gif_slots.json"
GIF_ASPECT = 420 / 980          # every recorded GIF is 980 x 420


def sp(x):
    """Playback speed of a GIF slot, scaled down with PACE so the clip still fits its (longer) slide."""
    return round(x / PACE, 2)


def speed_txt(x):
    return f"{sp(x):g}x"


SAMPLE_STATE = """altitude: 2900
. . . . .
. . Y . .
P . . . .
. P . . .
. P . . .
. P . C ."""

# illustrative state where the cell below Y is deadly, so the oracle sidesteps (label derived by hand from the rule)
DODGE_STATE = """altitude: 2900
. . . . .
. . Y . .
. . B . .
. P . . .
. P . . .
. P . C ."""


def mini_vbars(probs, truth, center, color):
    """Three vertical probability bars (left/right/stay) with the true option boxed."""
    g = VGroup()
    for i, (name, p) in enumerate(zip(["L", "R", "S"], probs)):
        h = max(1.2 * p, 0.02)
        x = center[0] + (i - 1) * 0.7
        rect = Rectangle(width=0.45, height=h, color=color, fill_opacity=0.85, stroke_width=0).move_to([x, center[1] - 0.6 + h / 2, 0])
        lab = Text(name, font_size=16).move_to([x, center[1] - 0.85, 0])
        val = Text(f"{p:.2f}", font_size=13, color=GRAY_A).next_to(rect, UP, buff=0.04)
        g.add(rect, lab, val)
        if i == truth:
            g.add(SurroundingRectangle(lab, color=YELLOW, buff=0.06))
    return g


class SkyfallDemo(Slides):
    SAY_SIZE = 26

    # ---- GIF slots (filled in later by ffmpeg) ------------------------
    def gif_slot(self, name, width, center, speed=1.0):
        ph = Rectangle(width=width, height=width * GIF_ASPECT, stroke_width=0, fill_color="#0f1117", fill_opacity=1).move_to([center[0], center[1], 0])
        self.add(ph)
        slot = {"gif": name, "t0": float(self.renderer.time), "t1": None, "cx": center[0], "cy": center[1], "w": width, "speed": speed}
        self.slots.append(slot)
        return slot

    def end_slot(self, slot):
        slot["t1"] = float(self.renderer.time)

    def construct(self):
        self.slots = []
        for fn in [
            self.s_intro, self.s_game, self.s_oracle_questions, self.s_dataset, self.s_metrics, self.s_architecture,
            self.s_before, self.s_training_setup, self.s_launch, self.s_learning_curve, self.s_montage,
            self.s_comparison, self.s_rivals, self.s_latency, self.s_balloon, self.s_takeaways,
        ]:
            fn()
            if fn != self.s_takeaways:
                self.reset()
        OUT.mkdir(exist_ok=True)
        SLOTS_FILE.write_text(json.dumps({"fps": config.frame_rate, "frame_width": config.frame_width, "slots": self.slots}, indent=1))

    # ---- 0 -------------------------------------------------------------
    def s_intro(self):
        grp = VGroup(
            Text("Teaching a decision model a new game", font_size=44, weight=BOLD, color=YELLOW),
            Text("Skyfall: fine-tuning Kev-0.8B on one MacBook", font_size=30, color=GRAY_A),
            Text("Can 20,000 synthetic examples fix a model that is at chance?", font_size=24, color=GRAY_B),
        ).arrange(DOWN, buff=0.5)
        self.play(FadeIn(grp, shift=UP * 0.2), run_time=1.0)
        self.wait(3.5)
        self.play(FadeOut(grp), run_time=0.5)

    # ---- 1 -------------------------------------------------------------
    def s_game(self):
        self.title("The game: fall from 3000 m, dodge, land")
        g = grid_mob(SAMPLE_STATE).move_to([-4.6, -0.3, 0])
        alt = Text("altitude: 2900", font_size=24, color=GRAY_A).next_to(g, UP, buff=0.25)
        self.play(FadeIn(g), FadeIn(alt))
        ybox = SurroundingRectangle(g[7], color=GREEN, buff=0.04)
        up = Arrow(g.get_right() + RIGHT * 0.2 + DOWN * 1.2, g.get_right() + RIGHT * 0.2 + UP * 1.2, color=GRAY_B, buff=0)
        uptxt = Text("scrolls up", font_size=16, color=GRAY_B).next_to(up, RIGHT, buff=0.08)
        self.play(Create(ybox), GrowArrow(up), FadeIn(uptxt))
        legend = VGroup(
            *[
                VGroup(Text(s, font_size=26, color=SYM_COL[s[0]], weight=BOLD), Text(d, font_size=22, color=GRAY_A)).arrange(RIGHT, buff=0.3)
                for s, d in [("Y", "you"), ("B  P  A", "deadly"), ("C", "cloud: harmless"), ("O", "balloon: never trained")]
            ]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([2.2, 1.6, 0])
        self.play(FadeIn(legend, lag_ratio=0.2), run_time=1.5)
        rules = VGroup(
            Text("Each tick: move left / right / stay", font_size=24, color=GRAY_A),
            Text("Open the parachute at 1000 m or lower", font_size=24, color=GRAY_A),
            Text("WIN: land over an empty cell or cloud", font_size=24, color=GREEN),
            Text("LOSE: hit anything, open too high, hit ground", font_size=24, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([2.6, -1.5, 0])
        self.play(FadeIn(rules, lag_ratio=0.4), run_time=2.0)
        self.say("Easy to state, impossible to guess from text, checkable by a rule.", wait=2.5)

    # ---- 2 -------------------------------------------------------------
    def s_oracle_questions(self):
        self.title("A rule labels the data: the oracle")
        g = grid_mob(DODGE_STATE, cell=0.55).move_to([-4.8, 0.0, 0])
        cap = Text("illustrative state", font_size=16, color=GRAY_C).next_to(g, DOWN, buff=0.2)
        self.play(FadeIn(g), FadeIn(cap))
        self.play(Create(SurroundingRectangle(g[12], color=RED, buff=0.03)))
        self.play(Create(SurroundingRectangle(VGroup(g[8], g[13]), color=GREEN, buff=0.03)))
        rule = VGroup(
            Text("Cell below is clear?  stay", font_size=24, color=GRAY_A),
            Text("Blocked?  sidestep to a safe column", font_size=24, color=GRAY_A),
            Text("open = 1000 m or lower AND clear below", font_size=24, color=GRAY_A),
            Text("Wins 94% of 500 jumps", font_size=24, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([2.0, 1.5, 0])
        self.play(FadeIn(rule, lag_ratio=0.4), run_time=2.5)
        qm = VGroup(
            Text("move  (Choice)", font_size=24, weight=BOLD, color=OPTION),
            Text("left / right / stay", font_size=20, color=GRAY_A),
            Text("label here: right", font_size=20, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        qo = VGroup(
            Text("open  (Noul, yes/no)", font_size=24, weight=BOLD, color=OPTION),
            Text("open the parachute now?", font_size=20, color=GRAY_A),
            Text("label here: no (2900 m)", font_size=20, color=GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        VGroup(qm, qo).arrange(RIGHT, buff=0.9).move_to([1.8, -1.9, 0])
        self.play(FadeIn(qm), FadeIn(qo))
        self.say("Two typed questions per state. Labels are exact.", wait=2.5)

    # ---- 3 -------------------------------------------------------------
    def s_dataset(self):
        self.title("Dataset: 20,000 train + 2,000 eval")
        steps = VGroup(
            Text("1. The oracle plays thousands of jumps", font_size=26, color=GRAY_A),
            Text("2. 30% random moves: we see bad positions too", font_size=26, color=GRAY_A),
            Text("3. Labels always come from the oracle", font_size=26, color=GRAY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([0, 1.7, 0])
        self.play(FadeIn(steps, lag_ratio=0.5), run_time=2.5)
        parts = [("dodge", 0.40, BLUE_C), ("open", 0.15, GREEN_C), ("blocked\nlow", 0.05, ORANGE), ("idle (stay)", 0.40, GRAY_B)]
        x, bar = -5.5, VGroup()
        for name, frac, col in parts:
            w = 11.0 * frac
            r = Rectangle(width=w, height=0.8, color=col, fill_opacity=0.7, stroke_width=1).move_to([x + w / 2, -0.7, 0])
            lab = Text(f"{int(frac * 100)}%", font_size=22 if w > 0.7 else 14, weight=BOLD).move_to(r)
            bar.add(VGroup(r, lab, Text(name, font_size=16, color=col).next_to(r, DOWN, buff=0.12)))
            x += w
        self.play(LaggedStart(*[FadeIn(b) for b in bar], lag_ratio=0.3), run_time=2.0)
        self.play(FadeIn(Text("Fixed mix: 'always stay' cannot score well", font_size=22, color=GRAY_B).move_to([0, -2.2, 0])))
        self.play(FadeIn(Text("Eval built first, train excludes it: a test caught 42 leaked states", font_size=22, color=YELLOW).move_to([0, -2.9, 0])))
        self.wait(3.0)
        self.cap = None

    # ---- 4. metrics ------------------------------------------------------
    def s_metrics(self):
        self.title("What we measure")
        cards = [
            ("move accuracy", "Top option = oracle's move", "question 1: left / right / stay", OPTION),
            ("open accuracy", "P(yes) > 0.5 = oracle's open", "question 2: open the parachute?", OPTION),
            ("Brier score", "Squared error vs the truth", "0 = perfect. Lower is better", YELLOW),
            ("win rate", "Jumps ending in a safe landing", "live play, 50 or 100 jumps", GREEN),
        ]
        grp = VGroup()
        for i, (h, d, s, col) in enumerate(cards):
            r = RoundedRectangle(corner_radius=0.1, width=6.2, height=1.25, color=col, fill_opacity=0.08)
            t = VGroup(Text(h, font_size=26, weight=BOLD, color=col), Text(d, font_size=22, color=GRAY_A), Text(s, font_size=16, color=GRAY_C)).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
            t.move_to(r).align_to(r, LEFT).shift(RIGHT * 0.3)
            grp.add(VGroup(r, t).move_to([(-3.4 if i % 2 == 0 else 3.4), 2.3 - (i // 2) * 1.5, 0]))
        self.play(LaggedStart(*[FadeIn(c) for c in grp], lag_ratio=0.4), run_time=3.0)
        self.wait(1.0)
        head = Text("Brier example: the truth is 'right'", font_size=24, weight=BOLD, color=YELLOW).move_to([0, -0.55, 0])
        self.play(FadeIn(head))
        exs = [
            ([0.1, 0.8, 0.1], GREEN, "0.1² + 0.2² + 0.1² = 0.06", "right and sure"),
            ([1 / 3, 1 / 3, 1 / 3], GRAY_B, "(1/3)² + (2/3)² + (1/3)² = 0.67", "guessing"),
            ([0.0, 0.0, 1.0], RED, "0 + 1 + 1 = 2.00", "sure and wrong"),
        ]
        out = VGroup()
        for k, (p, col, formula, tag) in enumerate(exs):
            cx = -4.3 + 4.3 * k
            ch = mini_vbars(p, 1, [cx, -1.35], col)
            ft = Text(formula, font_size=17, color=col).move_to([cx, -2.75, 0])
            tg = Text(tag, font_size=20, weight=BOLD, color=col).move_to([cx, -3.2, 0])
            out.add(VGroup(ch, ft, tg))
        self.play(LaggedStart(*[FadeIn(o) for o in out], lag_ratio=0.6), run_time=3.0)
        self.wait(4.0)
        self.cap = None

    # ---- 5. Kev architecture on Skyfall --------------------------------
    def s_architecture(self):
        self.title("Kev on Skyfall: one forward pass")
        seq = [
            ("<state> altitude: 2900 + grid", STATE),
            ("<q> Which way should you move?", QUESTION),
            ("<opt> left: move one column left </opt>", OPTION),
            ("<opt> right: move one column right </opt>", OPTION),
            ("<opt> stay: do not move sideways </opt>", OPTION),
            ("<decide>", DECIDE),
            ("<q> Open the parachute this tick?", QUESTION),
            ("<opt> no </opt>   <opt> yes </opt>", OPTION),
            ("<decide>", DECIDE),
        ]
        rows = VGroup(*[box(t, c, 5.0, height=0.4, font=15) for t, c in seq]).arrange(DOWN, buff=0.08).move_to([-4.3, 0.1, 0])
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.15) for r in rows], lag_ratio=0.15), run_time=2.2)
        tok = Text("state = 43 tokens, encoded once", font_size=16, color=GRAY_B).next_to(rows, DOWN, buff=0.15)
        self.play(FadeIn(tok))
        model = box("Qwen3.5-0.8B\nfrozen + LoRA\none pass", WHITE, 1.9, height=3.4, font=17, fill=0.08).move_to([-0.3, 0.1, 0])
        self.play(FadeIn(model), GrowArrow(Arrow(rows.get_right(), model.get_left(), buff=0.08)))
        picks = VGroup(*[SurroundingRectangle(rows[i], color=YELLOW, buff=0.03) for i in [2, 3, 4, 5, 7, 8]])
        self.play(Create(picks), run_time=1.2)
        self.say("Read the hidden state at each </opt> and at <decide>.", wait=1.5)

        head = Text("pointer head: logit = key · query / 16", font_size=20, color=TEAL).move_to([4.2, 3.0, 0])
        self.play(FadeIn(head), GrowArrow(Arrow(model.get_right(), [2.3, 1.7, 0], buff=0.08, color=TEAL)))

        raw_mv, raw_op = [-1.2, 3.4, -2.6], [2.8, -3.1]  # illustrative; gold: move = right, open = no
        names_mv, names_op = ["left", "right", "stay"], ["no", "yes"]
        mv_y, op_y = 1.9, -0.55
        mv_title = Text("move", font_size=22, weight=BOLD, color=OPTION).move_to([3.0, 2.45, 0])
        op_title = Text("open", font_size=22, weight=BOLD, color=OPTION).move_to([3.0, 0.0, 0])
        labs = VGroup(
            *[Text(n, font_size=20).move_to([3.7, mv_y - i * 0.5, 0], aligned_edge=RIGHT) for i, n in enumerate(names_mv)],
            *[Text(n, font_size=20).move_to([3.7, op_y - i * 0.5, 0], aligned_edge=RIGHT) for i, n in enumerate(names_op)],
        )
        b_mv = bars(raw_mv, [5.9, mv_y], 0.3, TEAL, dy=0.5)
        b_op = bars(raw_op, [5.9, op_y], 0.3, TEAL, dy=0.5)
        self.play(FadeIn(mv_title), FadeIn(op_title), FadeIn(labs))
        self.play(FadeIn(b_mv, lag_ratio=0.2), FadeIn(b_op, lag_ratio=0.2), run_time=1.5)
        self.say("One raw logit per option: K options, K logits.", wait=2.0)

        p_mv, p_op = softmax(raw_mv), softmax(raw_op)
        self.play(
            Transform(b_mv, bars(list(p_mv), [4.1, mv_y], 1.3, GREEN, dy=0.5, fmt="{:.3f}")),
            Transform(b_op, bars(list(p_op), [4.1, op_y], 1.3, GREEN, dy=0.5, fmt="{:.3f}")),
            run_time=1.5,
        )
        self.say("Softmax per question. T = 1: this fine-tune was not refit.", wait=2.0)
        loss = Text(f"gold = right:  loss = -log {p_mv[1]:.3f} = {-np.log(p_mv[1]):.3f}", font_size=22, color=YELLOW).move_to([3.9, -2.05, 0])
        loss2 = Text(f"gold = no:  loss = -log {p_op[0]:.3f} = {-np.log(p_op[0]):.3f}", font_size=22, color=YELLOW).move_to([3.9, -2.55, 0])
        self.play(FadeIn(loss), FadeIn(loss2))
        self.say("Illustrative numbers. Training minimises this loss.", wait=4.0)

    # ---- 6. before ---------------------------------------------------------
    def s_before(self):
        self.title("Before training: released Kev-0.8B")
        slot = self.gif_slot("kev-released.gif", 7.6, (-3.1, 0.5), speed=sp(1.0))
        stats = VGroup(
            Text("2,000 held-out states", font_size=18, color=GRAY_C),
            Text("move  0.554", font_size=32, color=RED),
            Text("= always say 'stay'", font_size=20, color=GRAY_B),
            Text("open  0.467", font_size=32, color=RED),
            Text("below chance", font_size=20, color=GRAY_B),
            Text("0 wins in 200 jumps", font_size=28, color=RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([4.5, 0.2, 0])
        self.play(FadeIn(stats, lag_ratio=0.25), run_time=2.5)
        self.say("Nothing in its data covers this game: it opens on tick one.", wait=5.0)
        self.end_slot(slot)

    # ---- 7. training -----------------------------------------------------
    def s_training_setup(self):
        self.title("Training: LoRA on a laptop")
        txt = VGroup(
            Text("Start from released Kev ('delta mode')", font_size=26, color=GRAY_A),
            Text("Backbone frozen. LoRA rank 16 + pointer head trained", font_size=26, color=GRAY_A),
            Text("Loss: cross-entropy on oracle labels. No RL", font_size=26, color=GRAY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([0, 2.4, 0])
        self.play(FadeIn(txt, lag_ratio=0.4), run_time=2.0)
        full_w = 10.0
        frozen = Rectangle(width=full_w, height=0.6, color=GRAY_B, fill_opacity=0.35).move_to([0, 0.8, 0])
        sliver = Rectangle(width=full_w * 11.3 / 800, height=0.6, color=ORANGE, fill_opacity=0.95, stroke_width=1).align_to(frozen, LEFT).align_to(frozen, UP)
        l1 = Text("frozen: ~800M parameters", font_size=20, color=GRAY_A).next_to(frozen, UP, buff=0.12)
        l2 = Text("trained: ~11.3M = 1.4%", font_size=22, color=ORANGE).next_to(sliver, DOWN, buff=0.45).align_to(frozen, LEFT).shift(RIGHT * 0.6)
        self.play(FadeIn(frozen), FadeIn(l1))
        self.play(FadeIn(sliver), Create(Line(sliver.get_bottom(), l2.get_top(), color=ORANGE, stroke_width=2)), FadeIn(l2))
        tiles = [("2,500", "optimizer steps"), ("20,000", "records, 1 epoch"), ("8", "batch (4 x 2)"), ("5.8 h", "M2 Max, 64 GB"), ("3.2 GB", "peak memory"), ("fp32", "no bf16 on Mac")]
        cards = VGroup()
        for big, small in tiles:
            r = RoundedRectangle(corner_radius=0.1, width=2.7, height=1.05, color=GRAY_C, fill_opacity=0.08)
            cards.add(VGroup(r, Text(big, font_size=30, weight=BOLD, color=YELLOW).move_to(r.get_center() + UP * 0.2), Text(small, font_size=17, color=GRAY_A).move_to(r.get_center() + DOWN * 0.27)))
        cards.arrange_in_grid(rows=2, cols=3, buff=0.25).move_to([0, -2.0, 0])
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.15) for c in cards], lag_ratio=0.2), run_time=2.5)
        self.say("Final adapter + head: a 62 MB file.", wait=3.5)

    # ---- 7b. launching training from the Hugging Face model ------------------
    def s_launch(self):
        self.title("Launching training from the HF model")
        pre = Text("git clone jaredpalmer/kev  (tested at commit 1d77363)   |   uv sync", font="Menlo", font_size=17, color=GRAY_B).move_to([0, 2.55, 0])
        self.play(FadeIn(pre))
        rows = [
            ("uv run python -m kev.train \\", "Kev's training entry point", GRAY_A),
            ("  --data data/train.jsonl \\", "our 20,000 oracle-labelled rows", YELLOW),
            ("  --init_from jaredpalmer/kev-0.8b \\", "released LoRA + head, from the HF Hub", GREEN),
            ("  --base Qwen/Qwen3.5-0.8B-Base \\", "frozen backbone, from the Hub", GREEN),
            ("  --epochs 1 --lr 2e-5 --batch 4 --accum 2 \\", "2,500 steps of 8 records", ORANGE),
            ("  --device mps --dtype fp32 \\", "Apple GPU; bf16 is CUDA-only", RED),
            ("  --out runs/skyfall-0.8b", "adapter + head + tokenizer: 62 MB", TEAL),
        ]
        y0 = 1.7
        code = VGroup(*[Text(c.strip(), font="Menlo", font_size=20, color=WHITE).move_to([0, y0 - i * 0.6, 0]) for i, (c, _, _) in enumerate(rows)])
        for m in code:
            m.align_to([-6.3 if m.text.startswith('uv') else -5.9, 0, 0], LEFT)
        panel = RoundedRectangle(corner_radius=0.12, width=13.4, height=4.5, color=GRAY_C, fill_color="#0f1117", fill_opacity=1, stroke_width=1.5).move_to([0, 0.05, 0])
        self.play(FadeIn(panel))
        notes = VGroup()
        for i, ((c, n, col), m) in enumerate(zip(rows, code)):
            note = Text(n, font_size=21, color=col).move_to([0, m.get_center()[1], 0])
            note.align_to([1.3, 0, 0], LEFT)
            notes.add(note)
            self.play(FadeIn(m, shift=RIGHT * 0.15), FadeIn(note, shift=LEFT * 0.15), run_time=0.8)
            self.wait(0.6)
        self.say("Two Hub downloads: the Kev checkpoint and its Qwen base.", wait=2.0)
        cuda = Text("On a CUDA GPU: --device cuda --dtype bf16 --checkpointing 1", font="Menlo", font_size=16, color=GRAY_B).move_to([0, -2.9, 0])
        self.play(FadeIn(cuda))
        self.wait(3.0)

    # ---- 8. learning curve -----------------------------------------------
    def s_learning_curve(self):
        self.title("Watching it learn")
        lc = sorted((json.loads(l) for l in open(RESULTS / "learning_curve.jsonl")), key=lambda r: r["step"])
        wr = sorted((json.loads(l) for l in open(RESULTS / "winrate_curve.jsonl")), key=lambda r: r["step"])
        ax = Axes(x_range=[0, 2500, 500], y_range=[0, 1, 0.25], x_length=9.4, y_length=4.3, tips=False).move_to([-1.5, -0.2, 0])
        xl = VGroup(*[Text(str(v), font_size=16).move_to(ax.c2p(v, -0.06)) for v in range(0, 2501, 500)])
        yl = VGroup(*[Text(f"{v:g}", font_size=16).move_to(ax.c2p(-85, v)) for v in [0, 0.25, 0.5, 0.75, 1.0]])
        xt = Text("optimizer step", font_size=18, color=GRAY_B).next_to(ax, DOWN, buff=0.4)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt))

        def line(xs, ys, color):
            m = VMobject(color=color, stroke_width=4)
            m.set_points_as_corners([ax.c2p(x, y) for x, y in zip(xs, ys)])
            return m

        steps = [0] + [r["step"] for r in lc]
        l_move = line(steps, [0.554] + [r["move_acc"] for r in lc], BLUE_C)
        l_open = line(steps, [0.467] + [r["open_acc"] for r in lc], GREEN_C)
        ws, wv = [0] + [r["step"] for r in wr], [0.0] + [r["win_rate_50"] for r in wr]
        l_win = VGroup(line(ws, wv, ORANGE), *[Dot(ax.c2p(x, y), radius=0.05, color=ORANGE) for x, y in zip(ws, wv)])
        leg = VGroup(
            Text("move accuracy", font_size=20, color=BLUE_C),
            Text("open accuracy", font_size=20, color=GREEN_C),
            Text("win rate", font_size=20, color=ORANGE),
            Text("oracle 0.94", font_size=20, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([5.9, 1.6, 0])
        self.play(FadeIn(leg), Create(DashedLine(ax.c2p(0, 0.94), ax.c2p(2500, 0.94), color=GRAY_B, stroke_width=2)))
        self.play(Create(l_open), run_time=2.5)
        self.say("Altitude rule first: open 0.47 to 0.89 in 100 steps.", wait=1.5)
        self.play(Create(l_move), run_time=3.5)

        def vmark(step, label, color):
            return VGroup(DashedLine(ax.c2p(step, 0), ax.c2p(step, 1.0), color=color, stroke_width=2), Text(label, font_size=15, color=color).next_to(ax.c2p(step, 1.0), UP, buff=0.05))

        self.play(FadeIn(vmark(250, "warm-up ends", YELLOW)))
        self.say("The grid rule only appears after warm-up.", wait=1.5)
        self.play(Create(l_win), run_time=3.0)
        self.play(FadeIn(vmark(700, "dip", RED)))
        self.say("Win-rate dip: it sees the obstacle, picks the wrong side.", wait=2.0)
        self.play(FadeIn(vmark(1300, "step 1300", GREEN)))
        self.say("Step 1300: 48 of 50 jumps. Then it only sharpens.", wait=2.5)
        foot = Text("accuracy: 300 held-out states. Win rate: 50 jumps. No win-rate points at steps 100, 200, 400.", font_size=14, color=GRAY_C).next_to(self.cap, UP, buff=0.1)
        self.play(FadeIn(foot))
        self.wait(1.5)

    # ---- 9. montage --------------------------------------------------------
    def s_montage(self):
        self.title("One jump (seed 7000), six models")
        slot = self.gif_slot("montage.gif", 10.6, (0, 0.2), speed=sp(2.0))
        self.say(f"Released, step 300, 700, 1000, 1300, final. Shown at {speed_txt(2.0)}.", wait=19.0)
        self.end_slot(slot)

    # ---- 10. comparison ----------------------------------------------------
    def s_comparison(self):
        self.title("Before vs after vs Jev vs Gemini")
        cols = ["", "Released\nKev", "Fine-tuned\nKev", "Jev\n(hosted)", "Gemini\nbare", "Gemini\n+ CoT"]
        rows = [
            ("move acc.", "0.554", "0.9975", "0.343", "0.37*", "0.86*"),
            ("open acc.", "0.467", "0.999", "0.949", "1.00*", "0.99*"),
            ("jumps won", "0/200", "20/20", "3/20", "0/10", "15/20"),
            ("latency p50", "56 ms", "56 ms", "176 ms", "745 ms", "809 ms"),
            ("$ / 1k dec.", "0.0075", "0.0075", "0.020", "0.07", "0.27"),
        ]
        colw = [2.0, 1.9, 1.9, 1.9, 1.9, 1.9]
        xs = np.cumsum([0] + colw[:-1]) + np.array(colw) / 2 - sum(colw) / 2
        hdr_col = [None, RED, GREEN, TEAL, ORANGE, ORANGE]
        for c, name in enumerate(cols):
            if name:
                self.add(Text(name, font_size=22, weight=BOLD, color=hdr_col[c], line_spacing=0.8).move_to([xs[c], 2.45, 0]))
        self.add(Line([-sum(colw) / 2, 1.85, 0], [sum(colw) / 2, 1.85, 0], color=GRAY_C))
        y, cells = 1.3, VGroup()
        for label, *vals in rows:
            row = VGroup(Text(label, font_size=20, color=GRAY_A).move_to([xs[0], y, 0]))
            for c, v in enumerate(vals, start=1):
                row.add(Text(v, font_size=24, color=hdr_col[c]).move_to([xs[c], y, 0]))
            cells.add(row)
            y -= 0.75
        self.play(LaggedStart(*[FadeIn(r) for r in cells], lag_ratio=0.5), run_time=4.0)
        notes = VGroup(
            Text("Jev: zero-shot, our wording, 2,000 held-out states; jumps are seeds 6000-6019, 20 each", font_size=17, color=GRAY_B),
            Text("* Gemini accuracy = agreement with the oracle on the states it visited (not the 2,000)", font_size=17, color=GRAY_B),
            Text("Gemini bare: 10 jumps, seeds 5000-5009. Costs: list prices; Kev priced at Jev's rate", font_size=17, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([0, -2.55, 0])
        self.play(FadeIn(notes))
        self.say("Thinking out loud is what makes Gemini competitive. Jev fails move zero-shot here.", wait=4.5)

    # ---- 10b. four models, same jumps, side by side -------------------------
    def s_rivals(self):
        self.title("Same jumps, four players")
        W, dx, dy = 6.5, 3.5, 1.5
        cells = [
            ("kev-released.gif", "Released Kev", RED, "lost all 3 (opens too high)", sp(1.0), (-dx, dy + 0.05)),
            ("kev-final.gif", "Fine-tuned Kev", GREEN, "won all 3", sp(2.0), (dx, dy + 0.05)),
            ("jev.gif", "Jev (hosted)", TEAL, "lost all 3", sp(1.5), (-dx, -dy - 0.15)),
            ("gemini-cot.gif", "Gemini + CoT", ORANGE, "won 2 of 3", sp(6.0), (dx, -dy - 0.15)),
        ]
        slots = []
        for gif, name, col, res, speed, (cx, cy) in cells:
            slots.append(self.gif_slot(gif, W, (cx, cy), speed=speed))
            self.add(Text(f"{name}: {res}", font_size=19, weight=BOLD, color=col).move_to([cx, cy + W * GIF_ASPECT / 2 + 0.17, 0]))
        self.wait(1.0)
        self.say(f"Seeds 7000-7002. Playback: released Kev {speed_txt(1.0)}, fine-tuned Kev {speed_txt(2.0)}, Jev {speed_txt(1.5)}, Gemini {speed_txt(6.0)} (it takes ~0.8 s a move).", wait=13.0)
        for sl in slots:
            self.end_slot(sl)

    # ---- 11. latency -------------------------------------------------------
    def s_latency(self):
        self.title("Latency per decision")
        x0, vmax, width = -3.4, 900.0, 7.0
        items = [("Kev fine-tuned (local)", 56.2, GREEN, "56 ms"), ("Jev (remote API)", 176.0, TEAL, "176 ms"), ("Gemini bare prompt", 745.0, ORANGE, "745 ms"), ("Gemini + CoT", 809.0, ORANGE, "809 ms")]
        bs = VGroup(*[hbar(l, v, vmax, width, c, 1.9 - i * 0.75, x0, t, font=22) for i, (l, v, c, t) in enumerate(items)])
        self.play(FadeIn(bs[0]))
        self.play(FadeIn(bs[1]), FadeIn(bs[2]), FadeIn(bs[3]), run_time=1.5)
        self.say("Kev is 3x faster than Jev (which includes network) and ~14x faster than Gemini.", wait=2.0)
        notes = VGroup(
            Text("Kev: M2 Max, MLX server, no network", font_size=18, color=GRAY_B),
            Text("Jev, Gemini: remote APIs, network included", font_size=18, color=GRAY_B),
            Text("Gemini CoT: 627 tokens in, 73 out per decision", font_size=18, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([-3.4, -2.1, 0])
        self.play(FadeIn(notes))
        cost = VGroup(
            Text("Cost per 1,000 decisions", font_size=22, weight=BOLD, color=YELLOW),
            Text("Kev $0.0075 (priced at Jev's rate)", font_size=20, color=GREEN),
            Text("Jev $0.020 (474 tokens per call)", font_size=20, color=TEAL),
            Text("Gemini $0.07 bare, $0.27 with CoT", font_size=20, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([4.0, -2.0, 0])
        self.play(FadeIn(cost))
        self.wait(5.0)

    # ---- 12. balloon -------------------------------------------------------
    def s_balloon(self):
        self.title("The surprise: a balloon it never saw")
        slot = self.gif_slot("kev-final-balloons.gif", 7.4, (-3.3, 0.5), speed=sp(2.0))
        head = Text("P(stay) with one symbol under Y", font_size=22, color=GRAY_A).move_to([4.2, 2.3, 0])
        probe = [("B", 0.0, BLUE_C, "trained"), ("P", 0.0, BLUE_C, "trained"), ("A", 0.0, BLUE_C, "trained"), ("Z", 0.0, GREEN, "unseen"), ("X", 0.0, GREEN, "unseen"), ("O", 0.33, RED, "balloon"), ("0", 0.61, RED, "digit")]
        pb = VGroup(*[hbar(f"{s} {tag}", v, 1.0, 2.6, col, 1.65 - i * 0.5, 3.5, f"{v:.2f}", font=18) for i, (s, v, col, tag) in enumerate(probe)])
        self.play(FadeIn(head), FadeIn(pb, lag_ratio=0.2), run_time=2.0)
        stats = VGroup(
            Text("100 jumps with balloons: 83 wins (oracle 97)", font_size=22, color=ORANGE),
            Text("14 deaths by balloon", font_size=20, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([-3.3, -2.3, 0])
        self.play(FadeIn(stats))
        self.say("Unseen letters are treated as deadly. Round glyphs are half-read as empty.", wait=9.0)
        self.end_slot(slot)

    # ---- 13. takeaways -----------------------------------------------------
    def s_takeaways(self):
        self.title("Takeaways")
        pts = VGroup(
            Text("A decision model changes the readout: one pass, a probability per option", font_size=24, color=GRAY_A),
            Text("Rule-made labels work when tests rule out leakage", font_size=24, color=GRAY_A),
            Text("1.4% of the weights, 5.8 h on a Mac: chance to oracle parity", font_size=24, color=GRAY_A),
            Text("Easy signals are learned first, positional ones later", font_size=24, color=GRAY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to([0, 0.9, 0])
        self.play(FadeIn(pts, lag_ratio=0.5), run_time=4.0)
        cav = VGroup(
            Text("Caveats", font_size=22, weight=BOLD, color=YELLOW),
            Text("A tiny CNN solves this grid. This is imitation, not RL.", font_size=19, color=GRAY_B),
            Text("Fine-tuned model is uncalibrated. 50 to 100 jumps: +/- a few points.", font_size=19, color=GRAY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([0, -2.1, 0])
        self.play(FadeIn(cav))
        self.wait(6.0)
        self.play(FadeIn(Text("github.com/MarvinMartin24/skyfall-kev   |   huggingface.co/jaredpalmer/kev-0.8b", font_size=16, color=GRAY_C).to_edge(DOWN, buff=0.3)))
        self.wait(3.0)
