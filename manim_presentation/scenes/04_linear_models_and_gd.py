from manim import *
from manim_slides import Slide

from theme import BG_COLOR, BODY_FS, HEADER_FS, MATH_FS, TEXT_COLOR, add_logo


class LinearModelsAndGD(Slide):
    """
    Linear least-squares in operator form + gradient descent intuition.
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
        # LEFT COLUMN — structured derivation + gradient derivation
        # ---------------------------------------------------------------------

        # --- Model ---
        model_lab = Text("Model:", font_size=BODY_FS, color=label_color, weight=BOLD)
        model_eq = MathTex(r"f_w(x)=x^\top w,\quad w\in\mathbb{R}^d", font_size=MATH_FS)
        model = VGroup(model_lab, model_eq).arrange(RIGHT, buff=0.45, aligned_edge=UP)
        model.next_to(title, DOWN, buff=0.6).to_edge(LEFT, buff=0.9)

        self.play(Write(model_lab), run_time=0.35)
        self.play(Write(model_eq), run_time=0.7)
        self.next_slide()

        # --- Data ---
        data_lab = Text("Data:", font_size=BODY_FS, color=label_color, weight=BOLD)
        data_eq = MathTex(
            r"X=[x_1,\dots,x_n]\in\mathbb{R}^{d\times n},\quad y\in\mathbb{R}^n",
            font_size=MATH_FS,
        )
        data = VGroup(data_lab, data_eq).arrange(RIGHT, buff=0.45, aligned_edge=UP)
        data.next_to(model, DOWN, buff=0.35).align_to(model, LEFT)

        self.play(Write(data_lab), run_time=0.35)
        self.play(Write(data_eq), run_time=0.8)
        self.next_slide()

        # --- Empirical Risk ---
        risk_lab = Text(
            "Empirical risk:", font_size=BODY_FS, color=label_color, weight=BOLD
        )
        risk_eq = MathTex(
            r"\mathcal{L}(w)=\tfrac12\|X^\top w-y\|_2^2",
            font_size=MATH_FS,
        )
        risk = VGroup(risk_lab, risk_eq).arrange(RIGHT, buff=0.45, aligned_edge=UP)
        risk.next_to(data, DOWN, buff=0.35).align_to(model, LEFT)

        self.play(Write(risk_lab), run_time=0.35)
        self.play(Write(risk_eq), run_time=0.8)
        self.next_slide()

        # --- Derive gradient (animate) ---
        grad_lab = Text("Gradient:", font_size=BODY_FS, color=label_color, weight=BOLD)

        # Step A: expand norm
        risk_expanded = MathTex(
            r"\mathcal{L}(w)=\tfrac12\,(X^\top w-y)^\top(X^\top w-y)",
            font_size=MATH_FS,
        ).move_to(risk_eq, aligned_edge=LEFT)

        self.play(TransformMatchingTex(risk_eq, risk_expanded), run_time=0.9)
        self.next_slide()
        risk_eq = risk_expanded

        # Step B: chain rule
        grad_eq1 = MathTex(
            r"\nabla_w\mathcal{L}(w)=X(X^\top w-y)",
            font_size=MATH_FS,
        )
        grad = VGroup(grad_lab, grad_eq1).arrange(RIGHT, buff=0.45, aligned_edge=UP)
        grad.next_to(risk, DOWN, buff=0.35).align_to(model, LEFT)

        self.play(Write(grad_lab), run_time=0.35)
        self.play(Write(grad_eq1), run_time=0.75)
        self.next_slide()

        # Step C: operator form
        grad_eq2 = MathTex(
            r"\nabla_w\mathcal{L}(w)=XX^\top w-Xy",
            font_size=MATH_FS,
        ).move_to(grad_eq1, aligned_edge=LEFT)

        self.play(TransformMatchingTex(grad_eq1, grad_eq2), run_time=0.8)
        self.next_slide()

        # --- Stationary condition ---
        stat_lab = Text(
            "Stationary point:", font_size=BODY_FS, color=label_color, weight=BOLD
        )
        stat_eq = MathTex(r"XX^\top w^\star=Xy", font_size=MATH_FS, color=GREEN)
        stat = VGroup(stat_lab, stat_eq).arrange(RIGHT, buff=0.45, aligned_edge=UP)
        stat.next_to(grad, DOWN, buff=0.35).align_to(model, LEFT)

        self.play(Write(stat_lab), run_time=0.35)
        self.play(FadeIn(stat_eq, shift=UP * 0.1), run_time=0.6)
        self.next_slide()

        # ---------------------------------------------------------------------
        # RIGHT COLUMN — 1D gradient descent (arrow follows GD step on curve)
        # ---------------------------------------------------------------------

        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 6, 1],
            x_length=5.0,
            y_length=3.0,
            tips=False,
        )
        axes.to_edge(RIGHT, buff=0.8).shift(UP * 0.1)

        xlab = MathTex(r"w", font_size=BODY_FS).next_to(axes.x_axis, DOWN, buff=0.15)
        ylab = MathTex(r"\mathcal{L}(w)", font_size=BODY_FS).next_to(
            axes.y_axis, LEFT, buff=0.15
        )

        self.play(
            FadeIn(axes, shift=UP * 0.1), FadeIn(xlab), FadeIn(ylab), run_time=0.6
        )
        self.next_slide()

        a, b = 1.4, 0.8

        def L_scalar(w):
            return 0.5 * a * (w - b) ** 2

        curve = axes.plot(lambda x: L_scalar(x), x_range=[-3, 3])
        self.play(Create(curve), run_time=0.7)
        self.next_slide()

        # GD parameters
        eta = 0.55
        w0 = -2.4
        T = 4

        ws = [w0]
        for _ in range(T):
            wt = ws[-1]
            ws.append(wt - eta * a * (wt - b))

        pts = [axes.c2p(w, L_scalar(w)) for w in ws]

        dot = Dot(pts[0], radius=0.06)

        def gd_arrow_at(w):
            grad = a * (w - b)
            w_next = w - eta * grad
            return Arrow(
                axes.c2p(w, L_scalar(w)),
                axes.c2p(w_next, L_scalar(w_next)),
                buff=0.0,
                max_tip_length_to_length_ratio=0.25,
                color=RED,
            )

        arr = gd_arrow_at(ws[0])

        step_eq = MathTex(
            r"w_{t+1}=w_t-\eta\,\nabla\mathcal{L}(w_t)",
            font_size=MATH_FS,
        )
        step_eq.next_to(axes, DOWN, buff=0.25).align_to(axes, LEFT)
        step_eq.shift(DOWN * 0.2)

        self.play(FadeIn(dot), FadeIn(arr), FadeIn(step_eq), run_time=0.5)
        self.next_slide()

        for i in range(T):
            self.play(
                dot.animate.move_to(pts[i + 1]),
                Transform(arr, gd_arrow_at(ws[i + 1])),
                run_time=0.5,
                rate_func=smooth,
            )
            self.next_slide()

        wstar_dot = Dot(axes.c2p(b, L_scalar(b)), radius=0.06)
        wstar_lbl = MathTex(r"w^\star", font_size=BODY_FS, color=GREEN).next_to(
            wstar_dot, UP, buff=0.15
        )
        self.play(FadeIn(wstar_dot), FadeIn(wstar_lbl), run_time=0.5)
        self.next_slide()
