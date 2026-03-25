from manim import *
from manim_slides import Slide

from theme import BG_COLOR, BODY_FS, HEADER_FS, MATH_FS, TEXT_COLOR, add_logo
from utils import *


class WhyAnalyzeTrainingDynamics(Slide):
    def construct(self):
        self.camera.background_color = BG_COLOR
        Text.set_default(color=TEXT_COLOR)
        MathTex.set_default(color=TEXT_COLOR)

        self.clear()
        add_logo(self)

        label_color = BLUE
        bullet_color = GRAY_A

        # ---------------------------------------------------------------------
        # Title
        # ---------------------------------------------------------------------
        title = Text(
            "Why analyze training dynamics?",
            font_size=HEADER_FS,
            weight=BOLD,
        ).to_edge(UP)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.6)
        self.next_slide()

        # ---------------------------------------------------------------------
        # Narrative header (short, spoken prompt)
        # ---------------------------------------------------------------------
        lead = (
            Text(
                "Start with a linear model. For convenience, use gradient flow.",
                font_size=BODY_FS * 0.88,
                color=GRAY_A,
            )
            .next_to(title, DOWN, buff=0.45)
            .to_edge(LEFT, buff=1.15)
        )

        self.play(FadeIn(lead, shift=UP * 0.05), run_time=0.45)
        self.next_slide()

        # ---------------------------------------------------------------------
        # Core ODE block (left)
        # ---------------------------------------------------------------------
        ode1 = MathTex(
            r"\dot r(t)=-A\,r(t),\qquad r(t)=f(t)-y",
            font_size=MATH_FS * 0.92,
        )
        ode1.next_to(lead, DOWN, buff=0.55).align_to(lead, LEFT)

        q = (
            Text(
                "How does this system evolve as we train?",
                font_size=BODY_FS * 0.85,
                color=GRAY_A,
            )
            .next_to(ode1, DOWN, buff=0.35)
            .align_to(ode1, LEFT)
        )

        self.play(FadeIn(ode1, shift=UP * 0.05), run_time=0.5)
        self.next_slide()
        self.play(FadeIn(q, shift=UP * 0.05), run_time=0.45)
        self.next_slide()

        ode2 = (
            MathTex(
                r"\text{Homogeneous linear ODE}\ \Rightarrow\ r(t)=e^{-At}\,r(0)",
                font_size=MATH_FS * 0.86,
                color=GREEN,
            )
            .next_to(q, DOWN, buff=0.28)
            .align_to(ode1, LEFT)
        )

        ode3 = (
            MathTex(
                r"A=Q\Lambda Q^\top\ \Rightarrow\ |r_i(t)|=e^{-\lambda_i t}|r_i(0)|",
                font_size=MATH_FS * 0.82,
            )
            .next_to(ode2, DOWN, buff=0.22)
            .align_to(ode1, LEFT)
        )

        self.play(FadeIn(ode2, shift=UP * 0.05), run_time=0.5)
        self.next_slide()
        self.play(FadeIn(ode3, shift=UP * 0.05), run_time=0.5)
        self.next_slide()

        # ---------------------------------------------------------------------
        # Right plot: exponentials with different eigenvalues
        # ---------------------------------------------------------------------
        # --- build plot elements ---
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 1.05, 0.2],
            x_length=5.2,
            y_length=2.8,
            tips=False,
        )

        xlab = MathTex(r"t", font_size=BODY_FS).next_to(axes.x_axis, DOWN, buff=0.10)
        ylab = MathTex(r"|r_i(t)|", font_size=BODY_FS).next_to(
            axes.y_axis, LEFT, buff=0.10
        )

        curves = VGroup()  # empty for now
        labels = VGroup()  # empty for now

        punch = (
            Text(
                "Decay rates are set by the spectrum of A.",
                font_size=BODY_FS * 0.88,
                color=GRAY_A,
            )
            .next_to(axes, UP, buff=0.25)
            .align_to(axes, LEFT)
        )

        # --- group EARLY and place at final location BEFORE any animation ---
        plot_group = VGroup(axes, xlab, ylab, curves, labels, punch)

        plot_group.scale(0.78)
        plot_group.to_edge(RIGHT, buff=1.05).shift(DOWN * 0.15 + RIGHT * 0.25)

        # if you want punch to appear later, hide it initially
        punch.set_opacity(0)

        # --- now animate (already placed correctly) ---
        self.play(
            FadeIn(axes, shift=UP * 0.06), FadeIn(xlab), FadeIn(ylab), run_time=0.55
        )
        self.next_slide()

        # --- create curves + labels (they are in the group so they appear in the right place) ---
        lambdas = [0.05, 0.3, 0.8]
        colors = [BLUE, GREEN, RED]

        for i, (lam, colr) in enumerate(zip(lambdas, colors)):
            curve = axes.plot(
                lambda t, lam=lam: np.exp(-lam * t), x_range=[0, 6], color=colr
            )
            curves.add(curve)

            lbl = MathTex(rf"\lambda={lam}_{i}", font_size=BODY_FS * 0.8, color=colr)
            lbl.next_to(axes.c2p(5.6, np.exp(-lam * 5.6)), RIGHT, buff=0.35)
            labels.add(lbl)

        self.play(Create(curves), run_time=0.9)
        self.play(FadeIn(labels, shift=UP * 0.05 + RIGHT * 0.5), run_time=0.4)
        self.next_slide()

        # reveal punch line now (since it was hidden)
        self.play(punch.animate.set_opacity(1), run_time=0.45)
        self.next_slide()
        # ---------------------------------------------------------------------
        # NOW bring in the 3 bullets (the “why”)
        # ---------------------------------------------------------------------
        bullet_fs = BODY_FS * 0.85

        def bullet(text, t2c=None):
            sq = Square(0.10, fill_opacity=1.0, stroke_width=0).set_color(bullet_color)
            body = Text(
                text,
                font_size=bullet_fs,
                line_spacing=1.15,
                t2c=t2c or {},
            )
            g = VGroup(sq, body).arrange(RIGHT, buff=0.35, aligned_edge=UP)
            return g

        b1 = bullet(
            "Predict training time / compute:\n which components take longest?",
            t2c={"training time": YELLOW, "compute": YELLOW},
        )
        b2 = bullet(
            "Explain what learns first:\n learning order (spectral bias).",
            t2c={"what learns first": YELLOW, "spectral bias": YELLOW},
        )
        b3 = bullet(
            "Bridge theory ↔ practice:\n turn a nonlinear problem into a tractable approximation.",
            t2c={"tractable": YELLOW, "approximation": YELLOW},
        )

        bullets = VGroup(b1, b2, b3).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        # place bullets to the left, but above the ODE block so the slide stays balanced
        bullets.next_to(lead, DOWN, buff=0.25).align_to(lead, LEFT)

        # We already used this space for ODE; so fade ODE slightly to de-emphasize, then show bullets.
        ode_block = VGroup(ode1, q, ode2, ode3)
        self.play(ode_block.animate.set_opacity(0), run_time=0.35)
        self.play(lead.animate.set_opacity(0), run_time=0.35)

        self.play(FadeIn(b1, shift=UP * 0.06), run_time=0.45)
        self.next_slide()
        self.play(FadeIn(b2, shift=UP * 0.06), run_time=0.45)
        self.next_slide()
        self.play(FadeIn(b3, shift=UP * 0.06), run_time=0.45)
        self.next_slide()

        # ---------------------------------------------------------------------
        # Transition line (bottom)
        # ---------------------------------------------------------------------
        transition = Text(
            "So we look for regimes where the dynamics become linear again.",
            font_size=BODY_FS * 0.82,
            color=GRAY_A,
        ).to_edge(DOWN, buff=1.55)

        self.play(FadeIn(transition, shift=UP * 2.5), run_time=0.45)
        self.next_slide()
