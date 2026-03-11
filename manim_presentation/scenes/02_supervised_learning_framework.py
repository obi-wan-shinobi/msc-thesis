from manim import *
from manim_slides import Slide

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
            r"\mathcal{L}_{\rho}(f)\ :=\ \mathbb{E}_{\rho}\!\left[(f(X)-Y)^2\right]",
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
            r"\mathcal{L}(f) = \frac{1}{n}\sum_{i=1}^{n}\left(f(x_i)-y_i\right)^2",
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
