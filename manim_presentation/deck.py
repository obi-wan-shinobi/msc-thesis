from manim import *
from manim_slides import Slide

from scenes import (
    IntroProblemSetup,
    LinearModelsAndGD,
    SupervisedLearningFramework,
    ThesisIntro,
)
from theme import BG_COLOR, BODY_FS, HEADER_FS, MATH_FS, TEXT_COLOR, add_logo
from utils import *


class LinearModelsAndGD(Slide):
    """
    Scalar linear model + GD (kept consistent with deep-linear toy later).
    """

    def construct(self):
        self.camera.background_color = BG_COLOR
        Text.set_default(color=TEXT_COLOR)
        MathTex.set_default(color=TEXT_COLOR)
        self.clear()
        add_logo(self)

        label_color = BLUE

        # ---------------------------------------------------------------------
        # Title
        # ---------------------------------------------------------------------
        title = Text(
            "Linear models and gradient descent",
            font_size=HEADER_FS,
            weight=BOLD,
        ).to_edge(UP)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.6)
        self.next_slide()

        # ---------------------------------------------------------------------
        # LEFT COLUMN — consistent 3-slot summary (Setup / Loss / Update)
        # ---------------------------------------------------------------------
        setup_lab = Text("Setup:", font_size=BODY_FS, color=label_color, weight=BOLD)
        setup_eq = MathTex(
            r"f_w(x)=x^\top w,\quad w\in\mathbb{R}^d",
            font_size=MATH_FS,
        )
        setup = VGroup(setup_lab, setup_eq).arrange(RIGHT, buff=0.45, aligned_edge=UP)
        setup.next_to(title, DOWN, buff=0.6).to_edge(LEFT, buff=0.9)

        self.play(Write(setup_lab), run_time=0.35)
        self.play(Write(setup_eq), run_time=0.7)
        self.next_slide()

        loss_lab = Text("Loss:", font_size=BODY_FS, color=label_color, weight=BOLD)
        loss_eq = MathTex(
            r"\mathcal{L}(w)=\tfrac12\|X^\top w-y\|_2^2",
            font_size=MATH_FS,
        )
        loss = VGroup(loss_lab, loss_eq).arrange(RIGHT, buff=0.45, aligned_edge=UP)
        loss.next_to(setup, DOWN, buff=0.35).align_to(setup, LEFT)

        self.play(Write(loss_lab), run_time=0.35)
        self.play(Write(loss_eq), run_time=0.75)
        self.next_slide()

        upd_lab = Text("GD update:", font_size=BODY_FS, color=label_color, weight=BOLD)
        upd_eq = MathTex(
            r"w_{t+1}=w_t-\eta\,X\big(X^\top w_t-y\big)",
            font_size=MATH_FS,
        )
        update = VGroup(upd_lab, upd_eq).arrange(RIGHT, buff=0.45, aligned_edge=UP)
        update.next_to(loss, DOWN, buff=0.35).align_to(setup, LEFT)

        self.play(Write(upd_lab), run_time=0.35)
        self.play(Write(upd_eq), run_time=0.9)
        self.next_slide()

        # ---------------------------------------------------------------------
        # RIGHT COLUMN — Network (top-right), persistent anchor
        # ---------------------------------------------------------------------
        neuron_stroke = GRAY_B
        edge_color = GRAY_C

        nn = NetworkMobject(
            layer_sizes=(7, 1),
            neuron_radius=0.10,
            neuron_to_neuron_buff=0.30,
            layer_to_layer_buff=1.75,
            neuron_stroke_color=neuron_stroke,
            neuron_stroke_width=2.2,
            neuron_fill_color=BLUE_E,
            neuron_fill_opacity=0.0,
            edge_color=edge_color,
            edge_stroke_width=1.2,
            edge_propagation_color=YELLOW,
            edge_propagation_time=0.55,
            brace_for_large_layers=False,
        ).deactivate()

        nn.scale(0.72)
        nn.to_edge(RIGHT, buff=0.85).shift(UP * 1.25)  # room for inset later
        nn.shift(LEFT * 1.25)  # room for inset later

        x_lbl = MathTex(r"x", font_size=BODY_FS).next_to(nn.layers[0], LEFT, buff=0.18)
        fx_lbl = MathTex(r"f_w(x)", font_size=BODY_FS).next_to(
            nn.layers[-1], RIGHT, buff=0.18
        )
        w_lbl = MathTex(r"w", font_size=BODY_FS, color=label_color).move_to(
            nn.get_center() + DOWN * 0.25
        )

        self.play(
            FadeIn(nn, shift=RIGHT * 0.15),
            FadeIn(x_lbl, shift=RIGHT * 0.1),
            FadeIn(fx_lbl, shift=LEFT * 0.1),
            FadeIn(w_lbl, shift=UP * 0.1),
            run_time=0.6,
        )
        self.next_slide()

        # Deterministic activations so renders are stable
        def ramp(n, lo, hi):
            if n <= 1:
                return np.array([(lo + hi) * 0.5])
            return np.linspace(lo, hi, n)

        acts = [
            np.full(len(nn.layers[0].neurons), 0.35),  # input pulse
            ramp(len(nn.layers[1].neurons), 0.35, 0.80),  # output
        ]

        self.play(nn.layer_activate_anim(0, acts[0], run_time=0.25), run_time=0.25)
        self.play(nn.forward_pass_anim(activations=acts), run_time=1.35)
        self.next_slide()

        # fade fills back down (network stays subtle)
        zeros = [np.zeros(len(layer.neurons)) for layer in nn.layers]
        self.play(
            nn.layer_activate_anim(1, zeros[1], run_time=0.25),
            run_time=0.35,
        )
        self.next_slide()

        # ---------------------------------------------------------------------
        # Bottom-right inset — 1D loss + GD steps (consistent with L(w)=1/2(w-c)^2)
        # ---------------------------------------------------------------------
        c = 0.8
        eta = 0.8  # stable for this quadratic (0 < eta < 2)

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 8, 1],
            x_length=3.7,
            y_length=2.2,
            tips=False,
        )
        axes.to_corner(DR, buff=0.85).shift(UP * 0.25)

        xlab = MathTex(r"w", font_size=BODY_FS).next_to(axes.x_axis, DOWN, buff=0.12)
        ylab = MathTex(r"\mathcal{L}(w)", font_size=BODY_FS).next_to(
            axes.y_axis, LEFT, buff=0.12
        )

        self.play(
            FadeIn(axes, shift=UP * 0.08), FadeIn(xlab), FadeIn(ylab), run_time=0.55
        )
        self.next_slide()

        def L_scalar(w):
            return 0.5 * (w - c) ** 2

        curve = axes.plot(lambda x: L_scalar(x), x_range=[-3, 3])
        self.play(Create(curve), run_time=0.6)
        self.next_slide()

        w0 = -2.4
        T = 3

        ws = [w0]
        for _ in range(T):
            wt = ws[-1]
            ws.append(wt - eta * (wt - c))

        pts = [axes.c2p(w, L_scalar(w)) for w in ws]
        dot = Dot(pts[0], radius=0.055)

        def gd_arrow_at(w):
            w_next = w - eta * (w - c)
            return Arrow(
                axes.c2p(w, L_scalar(w)),
                axes.c2p(w_next, L_scalar(w_next)),
                buff=0.0,
                max_tip_length_to_length_ratio=0.25,
                color=RED,
            )

        arr = gd_arrow_at(ws[0])

        step_eq = (
            MathTex(
                r"w_{t+1}=w_t-\eta(w_t-c)",
                font_size=MATH_FS * 0.85,
            )
            .next_to(axes, UP, buff=0.18)
            .align_to(axes, LEFT)
        )

        self.play(FadeIn(dot), FadeIn(arr), FadeIn(step_eq), run_time=0.5)
        self.next_slide()

        for i in range(T):
            self.play(
                dot.animate.move_to(pts[i + 1]),
                Transform(arr, gd_arrow_at(ws[i + 1])),
                run_time=0.45,
                rate_func=smooth,
            )
            self.next_slide()

        wstar_dot = Dot(axes.c2p(c, L_scalar(c)), radius=0.055)
        wstar_lbl = MathTex(r"w^\star", font_size=BODY_FS, color=GREEN).next_to(
            wstar_dot, UP, buff=0.12
        )
        self.play(FadeIn(wstar_dot), FadeIn(wstar_lbl), run_time=0.45)
        self.next_slide()


class ThesisDeck(Slide):
    """
    A wrapper that forces slide order for manim-slides.
    Render/present THIS class.
    """

    def construct(self):
        # 1) Intro
        ThesisIntro.construct(self)
        self.clear()

        # 2) Overview
        SupervisedLearningFramework.construct(self)
        self.clear()

        # 3) Models
        IntroProblemSetup.construct(self)
        self.clear()

        # 3) Linear models & GD
        LinearModelsAndGD.construct(self)
        self.clear()
