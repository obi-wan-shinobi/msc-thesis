from manim import *
from manim_slides import Slide

from theme import add_logo


class ThesisIntro(Slide):
    def construct(self):
        title = Paragraph(
            "Training and Generalization in",
            "overparameterized neural networks",
            alignment="center",
            font_size=56,
            weight=BOLD,
        ).to_edge(UP, buff=1.2)

        subtitle = Text(
            "Interim Thesis Presentation",
            font_size=28,
            color=BLUE,
        ).next_to(title, DOWN, buff=0.4)

        author = Text("Shreyas Kalvankar", font_size=32)

        affiliation = Text(
            "MSc Computer Science",
            font_size=26,
            color=BLUE,
        ).next_to(author, DOWN, buff=0.15)

        author_group = VGroup(author, affiliation).to_edge(DOWN + RIGHT, buff=1)

        add_logo(self, scale=0.2)

        self.play(Write(title), run_time=1)
        self.next_slide()
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.next_slide()
        self.play(
            FadeIn(author_group, shift=DOWN * 0.3),
            run_time=1.2,
        )
        self.next_slide()
        self.wait(0.2)


class SupervisedLearningFramework(Slide):
    def construct(self):
        # Ensure the slide starts clean
        self.clear()

        add_logo(self)

        # ---- Title ----
        title = Text(
            "The supervised learning framework", font_size=44, weight=BOLD
        ).to_edge(UP)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.6)
        self.next_slide()

        # ---- Styling helpers ----
        label_color = BLUE
        base_fs = 30
        math_fs = 34

        def row(label_text: str, math_tex: str, label_fs=base_fs, math_fs=math_fs):
            lab = Text(label_text, font_size=label_fs, color=label_color, weight=BOLD)
            expr = MathTex(math_tex, font_size=math_fs)
            # two-column layout
            g = VGroup(lab, expr).arrange(RIGHT, buff=0.45, aligned_edge=UP)
            return g, lab, expr

        # ---- Rows (like your screenshot) ----
        r1, lab1, ex1 = row(
            "Data:",
            r"n\ \text{observations}\ (x_i,y_i)\in\mathbb{R}^d\times\mathbb{R},\ i=1,\dots,n,\ \ \text{i.i.d.}\sim \rho",
        )
        r2, lab2, ex2 = row("Prediction function:", r"f(x)\in\mathbb{R}")
        r3, lab3, ex3 = row(
            "Aim:",
            r"\text{given a new }x,\ \text{predict its label }y\ \text{(generalize)}",
        )

        # We'll build the "How:" line more dynamically:
        how_label = Text("How:", font_size=base_fs, color=label_color, weight=BOLD)

        # Stepwise build-up for the risk:
        how_a = MathTex(r"\text{find } f^\star", font_size=math_fs)
        how_b = MathTex(r"\text{that minimizes the population risk}", font_size=math_fs)
        risk = MathTex(
            r"L_{\rho}(f)\ :=\ \mathbb{E}_{\rho}\!\left[(f(X)-Y)^2\right]",
            font_size=math_fs,
            color=GREEN,
        )

        how_line = VGroup(how_label, how_a).arrange(RIGHT, buff=0.45, aligned_edge=UP)
        risk.align_to(how_a, LEFT)
        risk.next_to(how_line, DOWN, buff=0.22)

        how_block = VGroup(how_line, risk).arrange(DOWN, aligned_edge=LEFT)

        # Problem line
        prob_label = Text("Problem:", font_size=base_fs, color=RED, weight=BOLD)
        prob_expr = MathTex(r"\rho\ \text{is unknown!}", font_size=math_fs)
        prob = VGroup(prob_label, prob_expr).arrange(RIGHT, buff=0.45, aligned_edge=UP)

        # Solution line
        sol_label = Text("Solution:", font_size=base_fs, color=GREEN, weight=BOLD)
        sol_expr = MathTex(
            r"L(f) = \frac{1}{n}\sum_{i=1}^{n}\left(f(x_i)-y_i\right)^2",
            font_size=math_fs,
        )

        # Punchline text
        sol_punchline = Text(
            "Minimize the empirical risk",
            font_size=base_fs,
            color=RED,
        )

        # ---- Layout the rows vertically ----
        rows = VGroup(r1, r2, r3, how_block, prob).arrange(
            DOWN, aligned_edge=LEFT, buff=0.55
        )
        rows.next_to(title, DOWN, buff=0.6).to_edge(LEFT, buff=1.0)

        # ---- Animate row-by-row ----
        # Data
        self.play(Write(lab1), run_time=0.4)
        self.play(Write(ex1), run_time=0.9)
        self.next_slide()

        # Prediction function
        self.play(Write(lab2), run_time=0.4)
        self.play(Write(ex2), run_time=0.6)
        self.next_slide()

        # Aim
        self.play(Write(lab3), run_time=0.4)
        self.play(Write(ex3), run_time=0.7)
        self.next_slide()

        # How: build it up nicely
        self.play(Write(how_label), run_time=0.35)
        self.play(Write(how_a), run_time=0.45)
        self.next_slide()

        # Replace "find f*" with the full sentence, then drop the risk underneath
        how_b.next_to(how_a, RIGHT, buff=0.35, aligned_edge=UP)
        self.play(Write(how_b), run_time=0.8)
        self.next_slide()

        risk.next_to(VGroup(how_label, how_a, how_b), DOWN, buff=0.25).align_to(
            ex1, LEFT
        )
        self.play(Write(risk), run_time=1.0)
        self.next_slide()

        # Problem
        self.play(Write(prob_label), run_time=0.35)
        self.play(Write(prob_expr), run_time=0.55)
        self.next_slide()

        # subtle emphasis on the "unknown rho" punchline
        self.play(Indicate(prob_expr, scale_factor=1.03), run_time=0.7)
        self.next_slide()

        # Animate label and equation transformations
        sol_label.move_to(prob_label)
        sol_expr.move_to(prob_expr, aligned_edge=LEFT)
        print("About to transform Problem -> Solution")
        self.play(
            Transform(prob_label, sol_label),
            TransformMatchingTex(prob_expr, sol_expr),
            run_time=1.0,
        )
        self.next_slide()

        # Place punchline to the RIGHT of the equation
        sol_punchline.next_to(sol_expr, RIGHT, buff=0.6)

        # Align vertically (centers look best here)
        sol_punchline.align_to(sol_expr, UP)
        self.play(Write(sol_punchline), run_time=0.55)
        self.next_slide()

        self.play(Indicate(sol_punchline, scale_factor=1.03), run_time=0.7)
        self.next_slide()

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
