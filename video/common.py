"""Shared helpers for the Skyfall / Kev explainer videos (Manim Community).

PACE: every animation and every wait is multiplied by PACE (env var, default 1.5) so there is room to
speak over the slides. PACE=1 gives the original, faster cut.
"""
import os

import numpy as np
from manim import *

PACE = float(os.environ.get("PACE", "1.5"))
STATE, QUESTION, OPTION, DECIDE = BLUE_C, GREEN_C, ORANGE, RED_C
SYM_COL = {".": GRAY_D, "Y": GREEN, "B": YELLOW, "P": PINK, "A": RED, "C": WHITE, "O": TEAL}
T = 2.35   # calibration temperature of the released Kev-0.8B (head.pt)


def softmax(x):
    e = np.exp(np.asarray(x, dtype=float) - np.max(x))
    return e / e.sum()


def box(label, color, width, height=0.42, font=16, fill=0.15):
    rect = RoundedRectangle(corner_radius=0.08, width=width, height=height, color=color, fill_opacity=fill)
    txt = Text(label, font_size=font)
    if txt.width > width - 0.2:
        txt.scale_to_fit_width(width - 0.2)
    txt.move_to(rect)
    return VGroup(rect, txt)


def bars(values, origin, scale, color, dy=0.6, fmt="{:+.2f}"):
    """Horizontal bars growing right (positive) or left (negative) from a vertical axis at origin[0]."""
    group = VGroup()
    for i, v in enumerate(values):
        w = max(abs(v) * scale, 0.02)
        rect = Rectangle(width=w, height=0.38, color=color, fill_opacity=0.85, stroke_width=0)
        rect.move_to([origin[0] + np.sign(v) * w / 2, origin[1] - i * dy, 0])
        val = Text(fmt.format(v), font_size=16)
        val.next_to(rect, RIGHT if v >= 0 else LEFT, buff=0.1)
        group.add(VGroup(rect, val))
    return group


def hbar(label, value, vmax, width, color, y, x0, fmt, font=18):
    w = max(width * value / vmax, 0.02)
    lab = Text(label, font_size=font).move_to([x0 - 0.15, y, 0], aligned_edge=RIGHT)
    rect = Rectangle(width=w, height=0.36, color=color, fill_opacity=0.85, stroke_width=0).move_to([x0 + w / 2, y, 0])
    val = Text(fmt, font_size=font, color=color).next_to(rect, RIGHT, buff=0.12)
    return VGroup(lab, rect, val)


def grid_mob(text, cell=0.55):
    """A Skyfall state ('altitude: N' + rows of space-separated cells) as a grid of squares."""
    rows = [l.split() for l in text.strip().split("\n")[1:]]
    g = VGroup()
    for r, row in enumerate(rows):
        for c, s in enumerate(row):
            sq = Square(cell, color=GRAY_C, stroke_width=1.5, fill_color="#1d2029", fill_opacity=1).move_to([c * cell, -r * cell, 0])
            t = Text(s if s != "." else "·", font_size=24, color=SYM_COL.get(s, WHITE), weight=BOLD if s == "Y" else NORMAL).move_to(sq)
            g.add(VGroup(sq, t))
    return g.move_to(ORIGIN)


class Slides(Scene):
    """Dark background, a title line, a caption line, and global pacing."""

    SAY_SIZE = 22

    def play(self, *anims, **kw):
        # Scene.wait() is implemented as play(Wait(...)); wait() below already applies PACE, so Wait is skipped
        # here, otherwise every pause would be scaled twice (PACE squared).
        if "run_time" in kw:
            kw["run_time"] *= PACE
        else:
            for a in anims:
                if isinstance(a, Animation) and not isinstance(a, Wait):
                    a.run_time *= PACE
        return super().play(*anims, **kw)

    def wait(self, duration=1.0, *a, **kw):
        return super().wait(duration * PACE, *a, **kw)

    def setup(self):
        self.camera.background_color = "#14161c"
        self.cap = None
        self.ttl = None

    def reset(self):
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)
        self.cap = None
        self.ttl = None

    def title(self, text):
        new = Text(text, font_size=32, weight=BOLD).to_edge(UP, buff=0.35)
        if self.ttl is None:
            self.play(FadeIn(new), run_time=0.5)
        else:
            self.play(FadeOut(self.ttl), FadeIn(new), run_time=0.5)
        self.ttl = new

    def say(self, text, wait=1.5):
        new = Text(text, font_size=self.SAY_SIZE, color=GRAY_A)
        if new.width > 12.4:
            new.scale_to_fit_width(12.4)
        new.to_edge(DOWN, buff=0.4)
        if self.cap is None:
            self.play(FadeIn(new), run_time=0.5)
        else:
            self.play(FadeOut(self.cap), FadeIn(new), run_time=0.5)
        self.cap = new
        self.wait(wait)
