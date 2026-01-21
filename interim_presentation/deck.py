from manim import *
from manim_slides import Slide

from scenes import SupervisedLearningFramework, ThesisIntro
from theme import BG_COLOR, BODY_FS, HEADER_FS, MATH_FS, TEXT_COLOR, add_logo


class SupervisedLearningFramework(Slide):
    """
    Slide 1: Supervised learning -> population risk -> empirical risk (ERM)
    Slide 2: ERM -> linear model -> vectorize -> expand -> gradient -> normal equations
    """

    def construct(self):
        self.camera.background_color = BG_COLOR
        Text.set_default(color=TEXT_COLOR)
        MathTex.set_default(color=TEXT_COLOR)
        self.clear()
        add_logo(self)

        label_color = BLUE

        # ---------------------------------------------------------------------
        # Helpers
        # ---------------------------------------------------------------------
        def row(label_text: str, math_tex: str, label_fs=BODY_FS, math_fs=MATH_FS):
            lab = Text(label_text, font_size=label_fs, color=label_color, weight=BOLD)
            expr = MathTex(math_tex, font_size=math_fs)
            g = VGroup(lab, expr).arrange(RIGHT, buff=0.45, aligned_edge=UP)
            return g, lab, expr

        def left_align_under(mob: Mobject, anchor: Mobject, buff=0.6):
            mob.next_to(anchor, DOWN, buff=buff)
            mob.align_to(anchor, LEFT)
            return mob

        # ---------------------------------------------------------------------
        # Title
        # ---------------------------------------------------------------------
        title = Text(
            "The supervised learning framework",
            font_size=HEADER_FS,
            weight=BOLD,
        ).to_edge(UP)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.6)
        self.next_slide()

        # ---------------------------------------------------------------------
        # Slide 1: Framework -> population risk -> ERM
        # ---------------------------------------------------------------------
        r1, lab1, ex1 = row(
            "Data:",
            r"n\ \text{observations}\ (x_i,y_i)\in\mathbb{R}^d\times\mathbb{R},\ i=1,\dots,n,\ \ \text{i.i.d.}\sim \rho",
        )
        r2, lab2, ex2 = row("Prediction function:", r"f(x)\in\mathbb{R}")
        r3, lab3, ex3 = row(
            "Aim:",
            r"\text{given a new }x,\ \text{predict its label }y\ \text{(generalize)}",
        )

        how_label = Text("How:", font_size=BODY_FS, color=label_color, weight=BOLD)
        how_a = MathTex(r"\text{find } f^\star", font_size=MATH_FS)
        how_b = MathTex(
            r"\text{that minimizes the population risk}",
            font_size=MATH_FS,
        )
        risk = MathTex(
            r"L_{\rho}(f)\ :=\ \mathbb{E}_{\rho}\!\left[(f(X)-Y)^2\right]",
            font_size=MATH_FS,
            color=GREEN,
        )

        how_a.next_to(how_label, RIGHT, buff=0.45, aligned_edge=UP)
        how_b.next_to(how_a, RIGHT, buff=0.35, aligned_edge=UP)
        how_line_full = VGroup(how_label, how_a, how_b)

        risk.next_to(how_line_full, DOWN, buff=0.22).align_to(ex1, LEFT)
        how_block = VGroup(how_line_full, risk).arrange(DOWN, aligned_edge=LEFT)

        prob_label = Text("Problem:", font_size=BODY_FS, color=RED, weight=BOLD)
        prob_expr_unknown = MathTex(r"\rho\ \text{is unknown!}", font_size=MATH_FS)
        prob = VGroup(prob_label, prob_expr_unknown).arrange(
            RIGHT, buff=0.45, aligned_edge=UP
        )

        # Layout: left column
        rows = VGroup(r1, r2, r3, how_block, prob).arrange(
            DOWN, aligned_edge=LEFT, buff=0.55
        )
        rows.next_to(title, DOWN, buff=0.6).to_edge(LEFT, buff=1.0)

        # Animate row-by-row
        self.play(Write(lab1), run_time=0.4)
        self.play(Write(ex1), run_time=0.9)
        self.next_slide()

        self.play(Write(lab2), run_time=0.4)
        self.play(Write(ex2), run_time=0.6)
        self.next_slide()

        self.play(Write(lab3), run_time=0.4)
        self.play(Write(ex3), run_time=0.7)
        self.next_slide()

        self.play(Write(how_label), run_time=0.35)
        self.play(Write(how_a), run_time=0.45)
        self.next_slide()

        self.play(Write(how_b), run_time=0.8)
        self.next_slide()

        self.play(Write(risk), run_time=1.0)
        self.next_slide()

        self.play(Write(prob_label), run_time=0.35)
        self.play(Write(prob_expr_unknown), run_time=0.55)
        self.next_slide()

        self.play(Indicate(prob_expr_unknown, scale_factor=1.03), run_time=0.7)
        self.next_slide()

        # Transform "Problem" -> "Solution: empirical risk"
        sol_label = Text("Solution:", font_size=BODY_FS, color=GREEN, weight=BOLD)
        sol_expr = MathTex(
            r"L(f) = \frac{1}{n}\sum_{i=1}^{n}\left(f(x_i)-y_i\right)^2",
            font_size=MATH_FS,
        )
        sol_punchline = Text(
            "Minimize the empirical risk",
            font_size=BODY_FS,
            color=RED,
        )

        # Place targets over the current "Problem" row
        sol_label.move_to(prob_label)
        sol_expr.move_to(prob_expr_unknown, aligned_edge=LEFT)

        self.play(
            ReplacementTransform(prob_label, sol_label),
            ReplacementTransform(prob_expr_unknown, sol_expr),
            run_time=1.0,
        )
        self.next_slide()

        solution_row = VGroup(sol_label, sol_expr)
        # solution_row.arrange(
        #     RIGHT, buff=0.45, aligned_edge=UP
        # )
        # Swap prob -> solution_row inside rows (so rows is no longer stale)
        rows.remove(prob)
        rows.add(solution_row)

        rows_target = rows.copy()
        rows_target.arrange(DOWN, aligned_edge=LEFT, buff=0.55)
        rows_target.next_to(title, DOWN, buff=0.6).to_edge(LEFT, buff=1.0)

        self.play(Transform(rows, rows_target), run_time=0.6)

        sol_punchline.next_to(sol_expr, RIGHT, buff=0.6).align_to(sol_expr, UP)
        self.play(Write(sol_punchline), run_time=0.55)
        self.next_slide()

        self.play(Indicate(sol_punchline, scale_factor=1.03), run_time=0.7)
        self.next_slide()

        # Fade out framework blocks first (keep only the solution row + title)
        framework_block = VGroup(r1, r2, r3, how_block, sol_punchline)
        self.play(FadeOut(framework_block, shift=LEFT * 0.2), run_time=0.6)
        self.next_slide()

        # Compute a TARGET position WITHOUT moving the original object
        solution_target = solution_row.copy()
        solution_target.next_to(title, DOWN, buff=0.55).to_edge(LEFT, buff=1.0)

        self.play(
            solution_row.animate.move_to(solution_target, aligned_edge=LEFT), run_time=1
        )
        self.next_slide()

        # # Anchor equation is now the moved empirical risk expression
        # anchor_eq = prob_expr_unknown
        #
        # # ---------------------------------------------------------------------
        # # Slide 2: ERM -> linear model -> vectorized -> expand -> gradient
        # # ---------------------------------------------------------------------
        #
        # # 1) Introduce linear model
        # model_lab = Text(
        #     "Choose a model:", font_size=BODY_FS, color=label_color, weight=BOLD
        # )
        # model_eq = MathTex(r"f_w(x) = x^\top w", font_size=MATH_FS)
        # model = VGroup(model_lab, model_eq).arrange(RIGHT, buff=0.45, aligned_edge=UP)
        # left_align_under(model, anchor_eq, buff=0.55)
        #
        # self.play(Write(model_lab), run_time=0.4)
        # self.play(Write(model_eq), run_time=0.6)
        # self.next_slide()
        #
        # # 2) Substitute into empirical risk: L(w) sum form
        # Lw_sum = MathTex(
        #     r"L(w)=\frac{1}{n}\sum_{i=1}^{n}\left(x_i^\top w-y_i\right)^2",
        #     font_size=MATH_FS,
        # ).move_to(anchor_eq, aligned_edge=LEFT)
        #
        # self.play(TransformMatchingTex(anchor_eq, Lw_sum), run_time=1.0)
        # self.next_slide()
        # anchor_eq = Lw_sum  # keep anchor handle consistent
        #
        # # 3) Define X and y
        # X_def = MathTex(
        #     r"X=\begin{bmatrix}x_1^\top\\ \vdots\\ x_n^\top\end{bmatrix}\in\mathbb{R}^{n\times d}",
        #     font_size=MATH_FS,
        # )
        # y_def = MathTex(
        #     r"y=\begin{bmatrix}y_1\\ \vdots\\ y_n\end{bmatrix}\in\mathbb{R}^{n}",
        #     font_size=MATH_FS,
        # )
        # defs = VGroup(X_def, y_def).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        # defs.next_to(model, RIGHT, buff=1.0).align_to(model, UP)
        #
        # self.play(FadeIn(defs, shift=UP * 0.2), run_time=0.8)
        # self.next_slide()
        #
        # # 4) Vectorized loss
        # Lw_vec = MathTex(
        #     r"L(w)=\frac{1}{n}\left\|Xw-y\right\|^2",
        #     font_size=MATH_FS,
        # ).move_to(anchor_eq, aligned_edge=LEFT)
        #
        # self.play(TransformMatchingTex(anchor_eq, Lw_vec), run_time=1.0)
        # self.next_slide()
        # anchor_eq = Lw_vec
        #
        # # 5) Expand norm to quadratic form
        # expand1 = MathTex(
        #     r"L(w)=\frac{1}{n}(Xw-y)^\top(Xw-y)",
        #     font_size=MATH_FS,
        # ).move_to(anchor_eq, aligned_edge=LEFT)
        #
        # self.play(TransformMatchingTex(anchor_eq, expand1), run_time=0.9)
        # self.next_slide()
        # anchor_eq = expand1
        #
        # expand2 = MathTex(
        #     r"L(w)=\frac{1}{n}\Big(w^\top X^\top X w - 2y^\top X w + y^\top y\Big)",
        #     font_size=MATH_FS,
        # ).move_to(anchor_eq, aligned_edge=LEFT)
        #
        # self.play(TransformMatchingTex(anchor_eq, expand2), run_time=1.1)
        # self.next_slide()
        # anchor_eq = expand2
        #
        # # Emphasize geometry term (optional)
        # geom_box = SurroundingRectangle(expand2, buff=0.18, color=YELLOW)
        # self.play(Create(geom_box), run_time=0.45)
        # self.next_slide()
        # self.play(FadeOut(geom_box), run_time=0.35)
        # self.next_slide()
        #
        # # 6) Gradient + set to zero -> normal equations
        # grad = MathTex(
        #     r"\nabla_w L(w)=\frac{2}{n}\Big(X^\top X w - X^\top y\Big)",
        #     font_size=MATH_FS,
        # )
        # left_align_under(grad, expand2, buff=0.5)
        #
        # self.play(Write(grad), run_time=0.9)
        # self.next_slide()
        #
        # zero = MathTex(r"\nabla_w L(w)=0", font_size=MATH_FS, color=GREEN)
        # zero.next_to(grad, RIGHT, buff=0.6).align_to(grad, UP)
        #
        # self.play(FadeIn(zero, shift=UP * 0.1), run_time=0.5)
        # self.next_slide()
        #
        # normal = MathTex(r"X^\top X w = X^\top y", font_size=MATH_FS, color=GREEN)
        # left_align_under(normal, grad, buff=0.35)
        #
        # self.play(TransformFromCopy(grad, normal), run_time=0.8)
        # self.next_slide()
        #
        # tag = Text("Normal equations", font_size=BODY_FS, color=GREEN)
        # tag.next_to(normal, RIGHT, buff=0.5).align_to(normal, UP)
        # self.play(FadeIn(tag, shift=UP * 0.1), run_time=0.5)
        # self.next_slide()

        self.wait(0.2)


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
        # self.clear()
