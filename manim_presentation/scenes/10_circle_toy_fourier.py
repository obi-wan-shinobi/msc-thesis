from manim import *
from manim_slides import Slide

from theme import BG_COLOR, BODY_FS, HEADER_FS, MATH_FS, TEXT_COLOR, add_logo
from utils import *


class CircleToyFourierDiagonalization(Slide):
    def construct(self):
        self.camera.background_color = BG_COLOR
        Text.set_default(color=TEXT_COLOR)
        MathTex.set_default(color=TEXT_COLOR)

        self.clear()
        logo = add_logo(self)  # do NOT move it

        label_color = BLUE
        title = Text(
            "Frequency principle: Fourier diagonalization",
            font_size=HEADER_FS,
            weight=BOLD,
        ).to_edge(UP)

        self.add(logo)
        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.6)
        self.next_slide()

        # -----------------------------
        # Helper: one-line row (label + body)
        # -----------------------------
        def row(label: str, body: Mobject, buff=0.35):
            lab = Text(label, font_size=BODY_FS * 0.75, color=label_color, weight=BOLD)
            g = VGroup(lab, body).arrange(RIGHT, buff=buff, aligned_edge=UP)
            return g

        math_fs = MATH_FS * 0.80

        r1 = row(
            "Setting:",
            MathTex(
                r"\varphi\in S^1,\qquad d\mu(\varphi)=\frac{d\varphi}{2\pi}",
                font_size=math_fs,
            ),
        )

        r2 = row(
            "Translation-invariant NTK:",
            MathTex(
                r"\Theta(x(\varphi),x(\psi))=\Theta(\varphi-\psi)=:\Theta(\delta)",
                font_size=math_fs,
            ),
        )

        r3 = row(
            "Convolution operator:",
            MathTex(
                r"(T_{\Theta}f)(\varphi)=\int_0^{2\pi} \Theta(\varphi-\psi)\,f(\psi)\,d\mu(\psi)",
                font_size=math_fs,
            ),
        )

        r4 = row(
            "Fourier diagonalization:",
            MathTex(
                r"1,\ \cos(k\varphi),\ \sin(k\varphi)\ \text{are eigenfunctions of }T_{\Theta}",
                font_size=math_fs * 0.92,
            ),
        )

        r5 = row(
            "Eigenvalues:",
            MathTex(
                r"\lambda_k=\int_{0}^{2\pi}\Theta(\delta)\cos(k\delta)\,d\mu(\delta)",
                font_size=math_fs,
            ),
        )

        r6 = row(
            "Mode-wise decay:",
            MathTex(
                r"\dot r=-T_{\Theta}r\ \Rightarrow\ r_k(t)=e^{-\lambda_k t}\,r_k(0)",
                font_size=math_fs,
            ),
        )

        left = VGroup(r1, r2, r3, r4, r5, r6).arrange(
            DOWN, buff=0.33, aligned_edge=LEFT
        )
        left.to_edge(LEFT, buff=0.85).shift(DOWN * 0.25)

        # -----------------------------
        # Right: axes + unit circle + δ on arc
        # -----------------------------
        # Small 2D axes to make it clear it's the unit circle
        ax2 = Axes(
            x_range=[-1.4, 1.4, 1],
            y_range=[-1.4, 1.4, 1],
            x_length=3.4,
            y_length=3.4,
            tips=False,
        )
        ax2.set_stroke(color=TEXT_COLOR, width=2.0, opacity=0.35)

        # Place this group top-right
        ax2.to_edge(RIGHT, buff=0.9).shift(UP * 0.65)

        # Unit circle in the same coordinate system
        unit_r = ax2.x_axis.unit_size * 1.0
        circle = Circle(radius=unit_r).move_to(ax2.c2p(0, 0))
        circle.set_stroke(TEXT_COLOR, width=2.2, opacity=0.85)

        # Two points on the circle
        phi = 55 * DEGREES
        psi = -35 * DEGREES

        p_phi = circle.point_at_angle(phi)
        p_psi = circle.point_at_angle(psi)

        dot_phi = Dot(p_phi, radius=0.045, color=TEXT_COLOR)
        dot_psi = Dot(p_psi, radius=0.045, color=TEXT_COLOR)

        lbl_phi = MathTex(r"\varphi", font_size=BODY_FS * 0.70).next_to(
            dot_phi, UP, buff=0.12
        )
        lbl_psi = MathTex(r"\psi", font_size=BODY_FS * 0.70).next_to(
            dot_psi, DOWN, buff=0.12
        )

        # Arc between psi -> phi (for δ)
        arc_angle = phi - psi
        arc = Arc(
            radius=unit_r,
            start_angle=psi,
            angle=arc_angle,
            arc_center=circle.get_center(),
        )
        arc.set_stroke(label_color, width=5, opacity=0.9)

        # δ label ON the arc (not inside)
        mid_angle = psi + 0.5 * arc_angle
        center = circle.get_center()
        on_circle_mid = center + unit_r * np.array(
            [np.cos(mid_angle), np.sin(mid_angle), 0.0]
        )
        outward = on_circle_mid - center
        outward = outward / np.linalg.norm(outward)

        lbl_delta = MathTex(
            r"\delta=\varphi-\psi", font_size=BODY_FS * 0.65, color=label_color
        )
        lbl_delta.move_to(on_circle_mid + 0.22 * outward)

        # Tiny label above circle (optional)
        kernel_lbl = MathTex(
            r"\Theta(\varphi,\psi)=\Theta(\delta)",
            font_size=BODY_FS * 0.62,
            color=TEXT_COLOR,
        ).next_to(ax2, UP, buff=0.10)

        circle_group = VGroup(
            ax2, circle, arc, dot_phi, dot_psi, lbl_phi, lbl_psi, lbl_delta, kernel_lbl
        )

        # -----------------------------
        # Animate
        # -----------------------------
        self.play(FadeIn(circle_group, shift=UP * 0.05), run_time=0.7)
        self.next_slide()

        for mob in left:
            self.play(FadeIn(mob, shift=UP * 0.05), run_time=0.45)
            self.next_slide()
