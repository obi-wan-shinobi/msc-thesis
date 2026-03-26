from manim import *
from manim_slides import Slide

from theme import BG_COLOR, BODY_FS, HEADER_FS, MATH_FS, TEXT_COLOR, add_logo
from utils import *


class FiniteSampleStory(Slide):
    def construct(self):
        # -----------------------------
        # Global styling
        # -----------------------------
        self.camera.background_color = BG_COLOR
        Text.set_default(color=TEXT_COLOR)
        MathTex.set_default(color=TEXT_COLOR)

        self.clear()
        logo = add_logo(self)  # DO NOT move it

        label_color = BLUE
        math_fs = MATH_FS * 0.80
        row_lab_fs = BODY_FS * 0.75

        # -----------------------------
        # Title
        # -----------------------------
        title = Text(
            "Finite samples: discretizing the circle",
            font_size=HEADER_FS,
            weight=BOLD,
        ).to_edge(UP)

        self.add(logo)
        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.6)
        self.next_slide()

        # -----------------------------
        # Helper: one-line row (label + body)  (your style)
        # -----------------------------
        def row(label: str, body: Mobject, buff=0.35):
            lab = Text(label, font_size=row_lab_fs, color=label_color, weight=BOLD)
            g = VGroup(lab, body).arrange(RIGHT, buff=buff, aligned_edge=UP)
            return g

        # ------------------------------------------------------------
        # RIGHT: unit circle + axes + sampled points (NO delta)
        # ------------------------------------------------------------
        ax2 = Axes(
            x_range=[-1.4, 1.4, 1],
            y_range=[-1.4, 1.4, 1],
            x_length=3.4,
            y_length=3.4,
            tips=False,
        )
        ax2.set_stroke(color=TEXT_COLOR, width=2.0, opacity=0.35)
        ax2.to_edge(RIGHT, buff=0.9).shift(UP * 0.35)

        unit_r = ax2.x_axis.unit_size * 1.0
        circle = Circle(radius=unit_r).move_to(ax2.c2p(0, 0))
        circle.set_stroke(TEXT_COLOR, width=2.2, opacity=0.85)

        circle_lbl = MathTex(
            r"S^1", font_size=BODY_FS * 0.70, color=TEXT_COLOR
        ).next_to(ax2, UP, buff=0.12)

        circle_group = VGroup(ax2, circle, circle_lbl)

        self.play(FadeIn(circle_group, shift=UP * 0.05), run_time=0.7)
        self.next_slide()

        # ------------------------------------------------------------
        # LEFT: narrative rows + definitions
        # ------------------------------------------------------------
        # 1) Assumptions issue
        r0 = row(
            "Issue:",
            Text(
                "Earlier picture assumes infinite width\nand training on the continuum [0,2π).",
                font_size=BODY_FS * 0.72,
                line_spacing=1.1,
                color=TEXT_COLOR,
            ),
            buff=0.42,
        )

        # 2) Simplify: keep infinite width, sample data
        r1 = row(
            "Discretize:",
            Text(
                "Keep infinite width, but replace the continuum\nwith uniformly sampled points.",
                font_size=BODY_FS * 0.72,
                line_spacing=1.1,
                color=TEXT_COLOR,
            ),
            buff=0.42,
        )

        # 3) Sampling statement (one line)
        r2 = row(
            "Sampling:",
            MathTex(
                r"\varphi_{1},\dots,\varphi_{n}\ \sim\ \mathrm{Unif}[0,2\pi)",
                font_size=math_fs,
            ),
            buff=0.42,
        )

        # 4) Φ, M, A
        r3 = row(
            "Feature matrix:",
            MathTex(
                r"\Phi_{i,p}=\varphi_{p}(\phi_{i}),\qquad p\in\{0,(k,c),(k,s)\}",
                font_size=math_fs,
            ),
            buff=0.42,
        )

        feat_basis = MathTex(
            r"\varphi_0(\phi)=1,\quad"
            r"\varphi_{k,c}(\phi)=\sqrt{2}\cos(k\phi),\quad"
            r"\varphi_{k,s}(\phi)=\sqrt{2}\sin(k\phi),\quad k=1,\dots,K.",
            font_size=math_fs * 0.7,
        )
        feat_basis.next_to(r3, DOWN, buff=0.12).align_to(
            r3[1], LEFT
        )  # align under equation, not label

        kernel_pair = VGroup(
            Text("Kernel:", font_size=BODY_FS * 0.70, color=label_color, weight=BOLD),
            MathTex(r"M_{i,j}=\Theta(\varphi_i-\varphi_j)", font_size=math_fs),
        ).arrange(RIGHT, buff=0.25, aligned_edge=UP)

        op_pair = VGroup(
            Text("Operator:", font_size=BODY_FS * 0.70, color=label_color, weight=BOLD),
            MathTex(r"A=\frac{1}{n}\,M", font_size=math_fs),
        ).arrange(RIGHT, buff=0.25, aligned_edge=UP)

        kernel_op_body = VGroup(kernel_pair, op_pair).arrange(
            RIGHT, buff=0.8, aligned_edge=UP
        )

        r4 = row("", kernel_op_body, buff=0.42)

        # r5 = row(
        #     "Empirical operator:",
        #     MathTex(r"A=\frac{1}{n}\,M", font_size=math_fs),
        #     buff=0.42,
        # )

        left = VGroup(r0, r1, r2, r3, feat_basis, r4).arrange(
            DOWN, buff=0.3, aligned_edge=LEFT
        )
        left.to_edge(LEFT, buff=0.85).shift(DOWN * 0.15)

        # Start hidden; reveal line-by-line
        for mob in left:
            mob.set_opacity(0)

        # Reveal narrative first
        for mob in [r0, r1, r2]:
            mob.set_opacity(1)
            self.play(FadeIn(mob, shift=UP * 0.05), run_time=0.55)
            self.next_slide()

        # ------------------------------------------------------------
        # Animate sampled points on the circle
        # ------------------------------------------------------------
        rng = np.random.default_rng(2)
        n_show_1 = 12
        n_show_2 = 36

        angles = rng.uniform(0, 2 * np.pi, size=n_show_2)

        dots = []
        for a in angles:
            p = circle.point_at_angle(a)
            dots.append(Dot(p, radius=0.032, color=label_color))

        dots_first = VGroup(*dots[:n_show_1])
        dots_rest = VGroup(*dots[n_show_1:])

        # First batch
        self.play(
            LaggedStart(
                *[FadeIn(d, shift=UP * 0.02) for d in dots_first], lag_ratio=0.06
            ),
            run_time=1.0,
        )
        self.next_slide()

        # Second batch (adds more)
        self.play(
            LaggedStart(
                *[FadeIn(d, shift=UP * 0.02) for d in dots_rest], lag_ratio=0.03
            ),
            run_time=1.0,
        )
        self.next_slide()

        # ------------------------------------------------------------
        # Reveal the matrices/operator rows
        # ------------------------------------------------------------
        for mob in [r3, feat_basis, r4]:
            mob.set_opacity(1)
            self.play(FadeIn(mob, shift=UP * 0.05), run_time=0.55)
            self.next_slide()

        # ------------------------------------------------------------
        # Key line (bottom)
        # ------------------------------------------------------------
        key = (
            Text(
                "Goal: the low-frequency Fourier block still almost diagonalizes the sampled operator.",
                font_size=BODY_FS * 0.72,
                color=TEXT_COLOR,
            )
            .to_edge(DOWN, buff=0.35)
            .shift(UP * 0.8)
        )

        self.play(FadeIn(key, shift=UP * 0.05), run_time=0.5)
        self.next_slide()
