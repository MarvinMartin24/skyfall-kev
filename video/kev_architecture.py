"""How a Kev decision model works, from text to loss and calibration (Manim Community).

    manim -ql  kev_architecture.py KevDeepDive                                   # 480p preview
    manim -r 1920,1080 --fps 30 --media_dir out/media kev_architecture.py KevDeepDive   # 1080p
    PACE=2 manim ...                                                             # slower, more room to talk

Every logit and gold label here is ILLUSTRATIVE (made up); T = 2.35 is the released Kev-0.8B value.
Shapes (hidden size 1024, 256-d pointer space, scale 1/sqrt(256)) follow EXPLAINER.md.
"""
from common import *

LEVELS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]      # the Score example
RAW = [-3.0, -1.2, 0.5, 3.0, 1.0]
TEAMS = ["billing", "technical", "sales", "other"]                    # the main Choice example: ticket routing
TEAM_RAW = [4.2, -0.6, 0.3, -1.0]                                     # made-up logits; gold = billing

# a support-ticket routing question (made-up ticket), one row per segment of the single token sequence
SEQ = [
    ("<state> I was charged twice for my March subscription.", STATE),
    ("<q> Which team should handle this ticket?", QUESTION),
    ("<opt> billing: payments, invoices, refunds </opt>", OPTION),
    ("<opt> technical: bugs, errors, outages </opt>", OPTION),
    ("<opt> sales: pricing, plans, upgrades </opt>", OPTION),
    ("<opt> other: anything else </opt>", OPTION),
    ("<decide>", DECIDE),
]
PROBS = list(softmax(np.array(TEAM_RAW) / T))

# name, text fed to the model (abbreviated), options, raw logits (made up), gold index (made up), note
EXAMPLES = [
    ("Choice", "<state> I was charged twice for my March subscription.  <q> Which team should handle this ticket?  <opt> billing </opt> ... <decide>",
     TEAMS, TEAM_RAW, 0, None),
    ("Noul", "<state> A venomoid is a venomous snake that has undergone a surgical procedure ...  <q> can you remove venom glands from a snake?  <opt> no </opt> <opt> yes </opt> <decide>",
     ["no", "yes"], [-1.9, 2.4], 1, "API returns only p(yes)"),
    ("Score", "<state> Great food, but we waited 45 minutes for a table.  <q> How many stars did this reviewer give?  <opt> 1 star </opt> ... <opt> 5 stars </opt> <decide>",
     LEVELS, RAW, 2, "expected level index = sum(i * p_i)"),
]


class KevDeepDive(Slides):
    def construct(self):
        self.frame_problem()
        self.reset()
        self.frame_sequence()
        self.frame_backbone()
        self.reset()
        self.frame_pointer_head()
        self.reset()
        self.frame_shared_head()
        self.reset()
        for name, _, opts, raw, _, _ in EXAMPLES:
            self.frame_matmul(name, opts, raw)
            self.reset()
        for ex in EXAMPLES:
            self.frame_type(*ex)
            self.reset()
        self.frame_cross_entropy()
        self.reset()
        self.frame_temperature()

    # ---- 1. generate vs read off -----------------------------------------
    def frame_problem(self):
        self.title("Generating an answer vs reading one off")
        left_lbl = Text("LLM: decode token by token, then parse", font_size=22, color=YELLOW).move_to([-3.4, 1.8, 0])
        toks = VGroup(*[box(s, GRAY_B, 0.9, font=18) for s in ['{', '"team"', ':', '"billing"', '}']]).arrange(RIGHT, buff=0.1).move_to([-3.4, 0.6, 0])
        left_note = Text("5 sequential forward passes\n+ a parser you must trust", font_size=20, color=GRAY_B).move_to([-3.4, -1.0, 0])
        right_lbl = Text("Decision model: one pass", font_size=22, color=GREEN).move_to([3.2, 1.8, 0])
        one_pass = box("1 forward pass", GREEN, 2.8, height=0.6, font=20).move_to([3.2, 0.8, 0])
        mini = bars(PROBS, [2.0, -0.2], 3.0, GREEN, dy=0.45, fmt="{:.2f}")
        arrow = Arrow(one_pass.get_bottom(), [3.2, 0.1, 0], buff=0.05, color=GREEN)
        self.play(FadeIn(left_lbl), FadeIn(right_lbl))
        self.play(LaggedStart(*[FadeIn(t, shift=RIGHT * 0.2) for t in toks], lag_ratio=0.6), run_time=2.5)
        self.play(FadeIn(left_note))
        self.play(FadeIn(one_pass), GrowArrow(arrow))
        self.play(FadeIn(mini, lag_ratio=0.2), run_time=1.0)
        self.say("A decision model returns a probability per option from a single prefill pass.", wait=2.0)

    # ---- 2. one sequence in ------------------------------------------------
    def frame_sequence(self):
        self.title("One token sequence in")
        self.rows = VGroup(*[box(t, c, 5.4) for t, c in SEQ]).arrange(DOWN, buff=0.12).move_to([-4.2, -0.1, 0])
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in self.rows], lag_ratio=0.25), run_time=3.0)
        legend = VGroup(*[
            VGroup(Square(0.2, color=c, fill_opacity=0.6, stroke_width=0), Text(t, font_size=18)).arrange(RIGHT, buff=0.15)
            for c, t in [(STATE, "state: the text to judge"), (QUESTION, "instructions of the question"),
                          (OPTION, "one <opt> ... </opt> per option"), (DECIDE, "<decide>: last token, sees everything")]
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([3.4, 0.6, 0])
        note = Text("Delimiters are existing Qwen special tokens\n(<|fim_prefix|>, <|box_end|>, ...): no new embeddings.\nUser text is escaped so it cannot forge them.",
                    font_size=16, color=GRAY_B).move_to([3.4, -1.7, 0])
        self.play(FadeIn(legend), FadeIn(note))
        self.say("State and question share one stream. Delimiters mark where each option ends.", wait=2.5)
        self.legend_group = VGroup(legend, note)

    # ---- 3. backbone, then pick rows ---------------------------------------
    def frame_backbone(self):
        self.title("Backbone, then read only a few hidden states")
        model = box("Qwen3.5-0.8B\nfrozen + LoRA r16\ncausal, prefill only", WHITE, 2.4, height=4.6, font=18, fill=0.08).move_to([0.9, -0.1, 0])
        self.play(FadeOut(self.legend_group), FadeIn(model), GrowArrow(Arrow(self.rows.get_right(), model.get_left(), buff=0.1)))
        hs = VGroup(*[box(f"h[{i}]", GRAY_B, 1.6, fill=0.0) for i in range(len(SEQ))]).arrange(DOWN, buff=0.12).move_to([5.0, -0.1, 0])
        self.play(GrowArrow(Arrow(model.get_right(), hs.get_left(), buff=0.1)), LaggedStart(*[FadeIn(h) for h in hs], lag_ratio=0.15), run_time=1.8)
        self.say("Output: one 1024-d hidden vector per token. Matrix H has shape [L, 1024].", wait=1.5)
        dim = [self.rows[0], self.rows[1], hs[0], hs[1]]
        picks = VGroup(*[SurroundingRectangle(hs[i], color=OPTION if i < len(SEQ) - 1 else DECIDE, buff=0.05) for i in range(2, len(SEQ))])
        self.play(*[m.animate.set_opacity(0.15) for m in dim], Create(picks), run_time=1.2)
        self.say("Only K+1 are used: one at each </opt> (the options) and one at <decide> (the question).", wait=2.5)

    # ---- 4. pointer head ---------------------------------------------------
    def frame_pointer_head(self):
        self.title("Pointer head: which option does the decision point at?")
        n = len(TEAMS)
        ys = [1.8 - i * 0.6 for i in range(n)]
        h_opts = VGroup(*[box(f"h_opt[{i}]", OPTION, 1.5, fill=0.3) for i in range(n)])
        keys = VGroup(*[box(f"k{i}", OPTION, 1.0, fill=0.5) for i in range(n)])
        for i in range(n):
            h_opts[i].move_to([-5.4, ys[i], 0])
            keys[i].move_to([-2.0, ys[i], 0])
        wk = box("W_k\n1024 to 256", WHITE, 1.4, height=3.0, font=16, fill=0.08).move_to([-3.7, 0.6, 0])
        h_dec = box("h_decide", DECIDE, 1.5, fill=0.3).move_to([-5.4, -2.0, 0])
        wq = box("W_q", WHITE, 1.0, height=0.6, font=16, fill=0.08).move_to([-3.7, -2.0, 0])
        query = box("query q", DECIDE, 1.3, fill=0.5).move_to([-2.0, -2.0, 0])
        self.play(FadeIn(h_opts), FadeIn(h_dec))
        self.play(FadeIn(wk), FadeIn(wq))
        self.play(FadeIn(keys), FadeIn(query), run_time=1.0)
        self.say("Two linear maps into a shared 256-d space: a key per option, one query for the question.", wait=1.5)
        labels = VGroup(*[Text(TEAMS[i], font_size=16).move_to([1.0, ys[i], 0]) for i in range(n)])
        dot = Text("k_i . q / sqrt(256)", font_size=16, color=DECIDE).move_to([-0.4, 2.45, 0])
        raw_bars = bars(TEAM_RAW, [4.2, 1.8], 0.5, TEAL)
        self.play(FadeIn(labels), FadeIn(dot))
        self.play(LaggedStart(*[FadeIn(b) for b in raw_bars], lag_ratio=0.2), run_time=1.5)
        self.say("Dot product per option gives one raw logit each. K options means K logits (illustrative numbers).", wait=2.5)

    # ---- 5. shared W_q / W_k across the three types ----------------------------
    def frame_shared_head(self):
        self.title("Same W_q and W_k for every question")
        lanes = [("Choice", "4 logits", 1.7), ("Noul", "2 logits", 0.1), ("Score", "5 logits", -1.5)]
        wks, wqs = [], []
        for name, outl, y0 in lanes:
            lab = Text(name, font_size=24, weight=BOLD).move_to([-6.2, y0, 0], aligned_edge=LEFT)
            ho = box("h_opt x K", OPTION, 1.6, height=0.4, font=14, fill=0.3).move_to([-3.9, y0 + 0.35, 0])
            wk = box("W_k", TEAL, 1.0, height=0.4, font=16, fill=0.5).move_to([-1.9, y0 + 0.35, 0])
            ks = box("keys [K,256]", OPTION, 1.6, height=0.4, font=14, fill=0.3).move_to([0.2, y0 + 0.35, 0])
            hd = box("h_decide", DECIDE, 1.6, height=0.4, font=14, fill=0.3).move_to([-3.9, y0 - 0.35, 0])
            wq = box("W_q", YELLOW, 1.0, height=0.4, font=16, fill=0.5).move_to([-1.9, y0 - 0.35, 0])
            qv = box("query [256]", DECIDE, 1.6, height=0.4, font=14, fill=0.3).move_to([0.2, y0 - 0.35, 0])
            out = box(outl + " (dot/16)", WHITE, 2.0, height=0.5, font=14, fill=0.1).move_to([3.0, y0, 0])
            arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.05, stroke_width=3, tip_length=0.12, color=GRAY_B)
                              for a, b in [(ho, wk), (wk, ks), (hd, wq), (wq, qv), (ks, out), (qv, out)]])
            wks.append(wk)
            wqs.append(wq)
            self.play(FadeIn(VGroup(lab, ho, wk, ks, hd, wq, qv, out, arrows)), run_time=0.9)
        legend = VGroup(Text("teal = the SAME W_k", font_size=16, color=TEAL), Text("yellow = the SAME W_q", font_size=16, color=YELLOW),
                        Text("2 x Linear(1024, 256)\n~0.52M params, tied", font_size=16, color=GRAY_B)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([5.6, 0.1, 0])
        self.play(FadeIn(legend))
        self.play(*[Indicate(w, scale_factor=1.3, color=TEAL) for w in wks], run_time=1.2)
        self.play(*[Indicate(w, scale_factor=1.3, color=YELLOW) for w in wqs], run_time=1.2)
        self.say("Only K changes between types: 4, 2 or 5 option rows. The head weights are literally reused.", wait=2.5)

    # ---- 6. the matrix multiplications -------------------------------------------
    def _mat(self, rows, w, color, ch=0.32, height=None):
        h = height if height is not None else rows * ch
        grp = VGroup(Rectangle(width=w, height=h, color=color, fill_opacity=0.18))
        if height is None:
            for i in range(1, rows):
                y = h / 2 - i * ch
                grp.add(Line([-w / 2, y, 0], [w / 2, y, 0], color=color, stroke_width=1.5))
        return grp

    def frame_matmul(self, name, opts, raw):
        K, ch = len(opts), 0.32
        self.title(f"{name}: K = {K} options give {K} logits")
        op = lambda s: Text(s, font_size=26, color=GRAY_A)
        A, Wk, C = self._mat(K, 1.3, OPTION), self._mat(0, 1.3, TEAL, height=1.6), self._mat(K, 1.3, OPTION)
        D, Wq, F = self._mat(1, 1.3, DECIDE), self._mat(0, 1.3, YELLOW, height=1.6), self._mat(1, 1.3, DECIDE)
        spacer = Rectangle(width=0.4, height=0.1, stroke_width=0, fill_opacity=0)
        top = VGroup(A, op("×"), Wk, op("="), C, spacer, D, op("×"), Wq, op("="), F).arrange(RIGHT, buff=0.2)
        if top.width > 12.6:
            top.scale_to_fit_width(12.6)
        top.move_to([0, 1.5, 0])
        tags = VGroup(*[Text(t, font_size=13, color=GRAY_B).next_to(m, DOWN, buff=0.12) for m, t in [
            (A, f"h_opt\n[{K}, 1024]"), (Wk, "W_k\n[1024, 256]"), (C, f"keys\n[{K}, 256]"),
            (D, "h_decide\n[1, 1024]"), (Wq, "W_q\n[1024, 256]"), (F, "query\n[1, 256]")]])
        head1 = Text("1. project into the shared 256-d space: W_k and W_q are the SAME in every type", font_size=18, color=GRAY_B).move_to([0, 2.8, 0])
        self.play(FadeIn(head1))
        self.play(LaggedStart(*[FadeIn(m) for m in top], lag_ratio=0.12), FadeIn(tags), run_time=2.2)
        self.play(Indicate(Wk, scale_factor=1.15, color=TEAL), Indicate(Wq, scale_factor=1.15, color=YELLOW), run_time=1.2)

        keys, qcol, out = self._mat(K, 1.3, OPTION), self._mat(0, 0.5, DECIDE, height=1.6), self._mat(K, 0.9, GREEN)
        bot = VGroup(keys, op("×"), qcol, op("/ 16  ="), out).arrange(RIGHT, buff=0.3).move_to([1.0, -1.75, 0])
        btags = VGroup(*[Text(t, font_size=13, color=GRAY_B).next_to(m, DOWN, buff=0.12)
                         for m, t in [(keys, f"keys\n[{K}, 256]"), (qcol, "queryᵀ\n[256, 1]"), (out, f"logits\n[{K}, 1]")]])
        head2 = Text("2. one dot product per option: each key row against the one query", font_size=18, color=GRAY_B).move_to([0, -0.55, 0])
        row_y = lambda m, i: m[0].get_top()[1] - (i + 0.5) * ch
        names = VGroup(*[Text(o, font_size=15, color=GRAY_A).move_to([keys[0].get_left()[0] - 0.15, row_y(keys, i), 0], aligned_edge=RIGHT) for i, o in enumerate(opts)])
        vals = VGroup(*[Text(f"{raw[i]:+.1f}", font_size=16).move_to([out[0].get_center()[0], row_y(out, i), 0]).set_opacity(0) for i in range(K)])
        self.play(FadeIn(head2), FadeIn(bot), FadeIn(btags), FadeIn(names), run_time=1.2)
        self.add(vals)
        for i in range(K):
            s1 = Rectangle(width=1.3, height=ch, color=YELLOW, stroke_width=3).move_to([keys[0].get_center()[0], row_y(keys, i), 0])
            s2 = Rectangle(width=0.9, height=ch, color=YELLOW, stroke_width=3).move_to([out[0].get_center()[0], row_y(out, i), 0])
            self.play(Create(s1), Create(s2), run_time=0.3)
            self.play(vals[i].animate.set_opacity(1), run_time=0.3)
            self.play(FadeOut(s1), FadeOut(s2), run_time=0.15)
        self.play(FadeIn(Text("logit_i = (k_i · q) / 16\n(then ÷ T at inference)", font_size=15, color=GRAY_A).move_to([6.0, -1.75, 0])))
        self.say(f"K = {K}: the query stays one [256] vector; only the number of key rows changes, so the output has {K} logits.", wait=2.5)

    # ---- 7. one worked example: logits -> / T -> softmax -> prediction -> loss -----
    def frame_type(self, name, seq_text, opts, raw, gold, note):
        n = len(opts)
        probs = softmax(np.array(raw) / T)
        pred = int(np.argmax(probs))
        loss = -float(np.log(probs[gold]))
        self.title(f"{name}: from logits to loss")
        seq = Text(seq_text, font_size=16, color=GRAY_B)
        if seq.width > 12.5:
            seq.scale_to_fit_width(12.5)
        seq.move_to([0, 2.95, 0])
        ys = [1.3 - i * 0.6 for i in range(n)]
        labels = VGroup(*[Text(o, font_size=20).move_to([-3.2, y, 0], aligned_edge=RIGHT) for o, y in zip(opts, ys)])
        origin = [-0.8, ys[0]]
        self.play(FadeIn(seq), FadeIn(labels))
        header = lambda txt, color: Text(txt, font_size=18, color=color).move_to([-0.8, 2.15, 0])
        h = header("raw logit = key . query / 16", TEAL)
        cur = bars(raw, origin, 0.4, TEAL)
        self.play(FadeIn(h), FadeIn(cur, lag_ratio=0.2), run_time=1.2)
        self.wait(0.8)
        self.play(Transform(h, header(f"divide by T = {T}", YELLOW)), Transform(cur, bars(list(np.array(raw) / T), origin, 0.4, YELLOW)), run_time=1.2)
        self.wait(0.8)
        self.play(Transform(h, header("softmax: probabilities sum to 1", GREEN)), Transform(cur, bars(list(probs), origin, 3.0, GREEN, fmt="{:.2f}")), run_time=1.2)
        self.wait(0.6)
        marks = VGroup()
        if pred == gold:
            marks.add(Text("<- prediction = gold", font_size=18, color=GREEN).move_to([2.6, ys[pred], 0], aligned_edge=LEFT))
        else:
            marks.add(Text("<- prediction", font_size=18, color=RED).move_to([2.6, ys[pred], 0], aligned_edge=LEFT))
            marks.add(Text("<- gold label", font_size=18, color=YELLOW).move_to([2.6, ys[gold], 0], aligned_edge=LEFT))
        self.play(FadeIn(marks), Create(SurroundingRectangle(labels[gold], color=YELLOW, buff=0.08)))
        if note:
            extra = f"expected level index = {sum(i * p for i, p in enumerate(probs)):.2f}" if name == "Score" else f"API returns only p(yes) = {probs[1]:.2f}"
            self.play(FadeIn(Text(extra, font_size=18, color=GRAY_A).move_to([0, -1.75, 0])))
        formula = Text(f"loss = -log p(gold) = -log {probs[gold]:.2f} = {loss:.2f}", font_size=24, color=YELLOW)
        panel = VGroup(RoundedRectangle(corner_radius=0.1, width=formula.width + 0.6, height=0.7, color=YELLOW, fill_opacity=0.08), formula)
        formula.move_to(panel)
        panel.move_to([0, -2.5, 0])
        self.play(FadeIn(panel))
        gold_txt = "(logits and gold label are made up)" if name in ("Choice", "Score") else "(logits made up; BoolQ example)"
        verdict = "Right and confident: small loss." if pred == gold else "Confident and wrong: large loss."
        self.say(f"{verdict} {gold_txt}".strip(), wait=2.5)

    # ---- 8. cross-entropy --------------------------------------------------------------
    def frame_cross_entropy(self):
        self.title("Cross-entropy: the one loss for all three types")
        ax = Axes(x_range=[0, 1, 0.25], y_range=[0, 4, 1], x_length=5.0, y_length=3.4, tips=False).move_to([-3.4, -0.1, 0])
        curve = ax.plot(lambda p: -np.log(p), x_range=[0.02, 1], color=YELLOW)
        xl = VGroup(*[Text(s, font_size=16).move_to(ax.c2p(v, -0.35)) for v, s in [(0, "0"), (0.5, "0.5"), (1, "1")]])
        yl = VGroup(*[Text(s, font_size=16).move_to(ax.c2p(-0.07, v)) for v, s in [(0, "0"), (2, "2"), (4, "4")]])
        xt = Text("p(gold)", font_size=18).next_to(ax, DOWN, buff=0.55)
        yt = Text("-log p", font_size=18).next_to(ax, UP, buff=0.1).shift(LEFT * 1.7)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), FadeIn(yt))
        self.play(Create(curve), run_time=1.5)
        dots = VGroup()
        for (name, _, _, raw, gold, _), color in zip(EXAMPLES, [GREEN, GREEN, RED]):
            p = float(softmax(np.array(raw) / T)[gold])
            d = Dot(ax.c2p(p, -np.log(p)), color=color)
            dots.add(VGroup(d, Text(f"{name} {-np.log(p):.2f}", font_size=16, color=color).next_to(d, UR, buff=0.08)))
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.5), run_time=2.0)
        lines = VGroup(
            Text("L = -log p(gold)", font_size=24, color=YELLOW),
            Text("= -sum_k y_k log p_k   (y is one-hot)", font_size=18, color=GRAY_A),
            Text("gradient on logit k:  p_k - y_k", font_size=18, color=GRAY_A),
            Text("mean over all questions in the batch;\nno per-type weights", font_size=18, color=GRAY_A),
            Text("one backward pass updates the LoRA weights\nand the shared W_q, W_k", font_size=18, color=GRAY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.3, 0.1, 0])
        self.play(LaggedStart(*[FadeIn(l) for l in lines], lag_ratio=0.5), run_time=3.0)
        self.say("Low p(gold) is punished steeply. Choice, Noul and Score all feed the same formula.", wait=3.0)

    # ---- 9. how T is set -----------------------------------------------------------------
    def frame_temperature(self):
        self.title("How T is set: after training, on held-out data")
        # toy world: the model's logits are 2.35x too sharp, so the best T should land near 2.35
        rng = np.random.default_rng(1)
        N, K = 4000, 4
        u = rng.normal(0, 1.6, (N, K))
        pt = np.exp(u - u.max(1, keepdims=True))
        pt /= pt.sum(1, keepdims=True)
        y = (pt.cumsum(1) > rng.random((N, 1))).argmax(1)
        z = 2.35 * u
        grid = np.round(np.arange(1.0, 4.001, 0.05), 2)

        def stats(t):
            p = np.exp(z / t - (z / t).max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            return float(-np.log(p[np.arange(N), y]).mean()), float(p.max(1).mean())

        nll = np.array([stats(t)[0] for t in grid])
        conf = np.array([stats(t)[1] for t in grid])
        acc = float((z.argmax(1) == y).mean())
        t_star = float(grid[nll.argmin()])
        lo, hi = float(nll.min()) - 0.03, float(nll.max()) + 0.03
        ax = Axes(x_range=[1, 4, 1], y_range=[lo, hi, (hi - lo) / 4], x_length=5.0, y_length=3.0, tips=False).move_to([-3.3, 0.1, 0])
        curve = ax.plot(lambda t: float(np.interp(t, grid, nll)), x_range=[1, 4], color=YELLOW)
        ticks = VGroup(*[Text(str(v), font_size=16).move_to(ax.c2p(v, lo - 0.04 * (hi - lo))) for v in [1, 2, 3, 4]])
        xt = Text("T", font_size=18).next_to(ax, DOWN, buff=0.55)
        yt = Text("held-out NLL", font_size=18).next_to(ax, UP, buff=0.1)
        self.play(Create(ax), FadeIn(ticks), FadeIn(xt), FadeIn(yt))
        self.play(Create(curve), run_time=1.2)

        tr = ValueTracker(1.0)
        dot = always_redraw(lambda: Dot(ax.c2p(tr.get_value(), float(np.interp(tr.get_value(), grid, nll))), color=RED))
        readout = always_redraw(lambda: Text(f"T = {tr.get_value():.2f}    NLL = {float(np.interp(tr.get_value(), grid, nll)):.3f}", font_size=20).move_to([-3.3, -2.55, 0]))
        names = TEAMS
        raw4 = EXAMPLES[0][3]
        rows_y = [1.3 - i * 0.6 for i in range(4)]
        lab4 = VGroup(*[Text(o, font_size=16).move_to([3.3, y, 0], aligned_edge=RIGHT) for o, y in zip(names, rows_y)])
        bars4 = always_redraw(lambda: bars(list(softmax(np.array(raw4) / tr.get_value())), [3.5, rows_y[0]], 2.2, GREEN, fmt="{:.2f}"))
        head4 = Text("one example question at this T", font_size=16, color=GRAY_B).move_to([4.6, 2.1, 0])
        calib = always_redraw(lambda: Text(f"mean confidence {float(np.interp(tr.get_value(), grid, conf)):.2f}   accuracy {acc:.2f}", font_size=18, color=GREEN).move_to([4.2, -1.6, 0]))
        self.add(dot, readout, bars4, calib)
        self.play(FadeIn(lab4), FadeIn(head4))
        self.say("Step 1: training ends with T = 1. Step 2: freeze all weights, run the model on held-out questions.", wait=2.0)
        self.say("Step 3: try every T, and keep the one with the lowest negative log-likelihood. One scalar, no backprop.", wait=0.5)
        self.play(tr.animate.set_value(t_star), run_time=7, rate_func=smooth)
        self.play(FadeIn(Text(f"T* = {t_star:.2f}", font_size=22, color=RED).next_to(dot, UP, buff=0.2)))
        self.say("At T* the confidence matches accuracy. Accuracy never moves, because dividing by T keeps the argmax.", wait=3.0)
        self.say(f"Toy simulation: its logits were built {2.35}x too sharp, so T* lands near 2.35. Kev stores T in head.pt.", wait=3.0)
