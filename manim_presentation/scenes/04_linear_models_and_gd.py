from manim import *
from manim_slides import Slide

from theme import BG_COLOR, BODY_FS, HEADER_FS, MATH_FS, TEXT_COLOR, add_logo
from utils import *


class LinearModelsAndGD(Slide):
    """
    Linear least-squares in operator form + GD intuition.
    Convention: X in R^{d x n} (columns are x_i), predictions X^T w.
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
        # LEFT COLUMN — Setup
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

        # ---------------------------------------------------------------------
        # RIGHT COLUMN — Network (top-right) as anchor
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
        nn.to_edge(RIGHT, buff=0.85).shift(UP * 1.25)
        nn.shift(LEFT * 1.25)  # leave room for inset

        x_lbl = MathTex(r"x", font_size=BODY_FS).next_to(nn.layers[0], LEFT, buff=0.18)
        fx_lbl = MathTex(r"f_w(x)", font_size=BODY_FS).next_to(
            nn.layers[-1], RIGHT, buff=0.18
        )
        w_lbl = MathTex(r"w", font_size=BODY_FS, color=label_color).move_to(
            nn.get_center() + DOWN * 0.75
        )

        self.play(
            FadeIn(nn, shift=RIGHT * 0.15),
            FadeIn(x_lbl, shift=RIGHT * 0.1),
            FadeIn(fx_lbl, shift=LEFT * 0.1),
            FadeIn(w_lbl, shift=UP * 0.1),
            run_time=0.6,
        )
        self.next_slide()

        # quick deterministic pulse
        acts = [
            np.full(len(nn.layers[0].neurons), 0.35),
            np.array([0.70]),
        ]
        self.play(nn.layer_activate_anim(0, acts[0], run_time=0.22), run_time=0.22)
        self.play(nn.forward_pass_anim(activations=acts), run_time=0.9)
        self.next_slide()

        self.play(
            nn.layer_activate_anim(1, np.array([0.0]), run_time=0.22), run_time=0.25
        )
        self.next_slide()

        # ---------------------------------------------------------------------
        # LEFT COLUMN — Loss
        # ---------------------------------------------------------------------
        loss_lab = Text("Loss:", font_size=BODY_FS, color=label_color, weight=BOLD)
        loss_eq = MathTex(
            r"\mathcal{L}(w)=\tfrac12\|X^\top w-y\|_2^2",
            font_size=MATH_FS,
        )
        loss = VGroup(loss_lab, loss_eq).arrange(RIGHT, buff=0.45, aligned_edge=UP)
        loss.next_to(setup, DOWN, buff=0.35).align_to(setup, LEFT)

        self.play(Write(loss_lab), run_time=0.35)
        self.play(Write(loss_eq), run_time=0.8)
        self.next_slide()

        # ---------------------------------------------------------------------
        # LEFT COLUMN — GD update as a HEADER line, equations start BELOW it
        # ---------------------------------------------------------------------
        upd_lab = Text("GD update:", font_size=BODY_FS, color=label_color, weight=BOLD)
        upd_lab.next_to(loss, DOWN, buff=0.42).align_to(setup, LEFT)

        self.play(Write(upd_lab), run_time=0.35)
        self.next_slide()

        # Equation column alignment: align to loss_eq (so it matches Setup/Loss equation column)
        upd_generic = MathTex(
            r"w_{t+1}=w_t-\eta\,\nabla_w\mathcal{L}(w_t)",
            font_size=MATH_FS,
        )
        upd_generic.next_to(upd_lab, DOWN, buff=0.18).align_to(loss_eq, LEFT)

        self.play(Write(upd_generic), run_time=0.75)
        self.next_slide()

        grad_eq = MathTex(
            r"\nabla_w\mathcal{L}(w)=X(X^\top w-y)=XX^\top w-Xy",
            font_size=MATH_FS * 0.92,
        )
        grad_eq.next_to(upd_generic, DOWN, buff=0.18).align_to(upd_generic, LEFT)

        self.play(FadeIn(grad_eq, shift=UP * 0.06), run_time=0.55)
        self.next_slide()

        convex_line = MathTex(
            r"\mathcal{L}\ \text{is convex}\ \Rightarrow\ \nabla_w\mathcal{L}(w^\star)=0",
            font_size=MATH_FS * 0.78,
            color=GRAY_A,
        )
        convex_line.next_to(grad_eq, DOWN, buff=0.22).align_to(upd_generic, LEFT)

        opt_eq = MathTex(
            r"XX^\top w^\star=Xy",
            font_size=MATH_FS * 0.98,
            color=GREEN,
        )
        opt_eq.next_to(convex_line, DOWN, buff=0.14).align_to(upd_generic, LEFT)

        uniq_line = MathTex(
            r"\text{(unique if }XX^\top \succ 0\text{)}",
            font_size=MATH_FS * 0.72,
            color=GRAY_B,
        )
        uniq_line.next_to(opt_eq, DOWN, buff=0.10).align_to(upd_generic, LEFT)

        self.play(FadeIn(convex_line, shift=UP * 0.05), run_time=0.45)
        self.play(FadeIn(opt_eq, shift=UP * 0.06), run_time=0.55)
        self.play(FadeIn(uniq_line, shift=UP * 0.04), run_time=0.35)
        self.next_slide()

        # ---------------------------------------------------------------------
        # Bottom-right inset — keep EXACTLY your w/c plot code
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

        self.play(FadeIn(dot), FadeIn(arr), run_time=0.5)
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
