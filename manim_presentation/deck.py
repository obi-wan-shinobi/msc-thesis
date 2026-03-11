from manim import *
from manim_slides import Slide

from scenes import LinearModelsAndGD, SupervisedLearningFramework, ThesisIntro
from theme import BG_COLOR, BODY_FS, HEADER_FS, MATH_FS, TEXT_COLOR, add_logo
from utils import *


class IntroProblemSetup(Slide):
    """
    Intro & problem setup slide:
    - title
    - sentence block
    - linear vs neural rows
    - animated neural network (3b1b-style)
    - side arrows
    """

    def construct(self):
        # --- global style ---
        self.camera.background_color = BG_COLOR
        Text.set_default(color=TEXT_COLOR)
        MathTex.set_default(color=TEXT_COLOR)
        self.clear()
        add_logo(self)

        label_color = BLUE

        # ---------------------------
        # Helper: row(label, math)
        # ---------------------------
        def row(label_text: str, math_tex: str, label_fs=BODY_FS, math_fs=MATH_FS):
            lab = Text(label_text, font_size=label_fs, color=label_color, weight=BOLD)
            expr = MathTex(math_tex, font_size=math_fs, color=TEXT_COLOR)
            g = VGroup(lab, expr).arrange(RIGHT, buff=0.45, aligned_edge=UP)
            return g, lab, expr

        # ============================================================
        # CHECKPOINT 1: Title
        # ============================================================
        title = Text(
            "Introduction & Problem setup",
            font_size=HEADER_FS,
            color=TEXT_COLOR,
            weight=BOLD,
        )
        title.to_edge(UP).to_edge(LEFT, buff=1.0)

        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.6)

        s1 = Text(
            "In order to minimize the empirical risk we consider a\n"
            "family of parameterized functions",
            font_size=BODY_FS,
            color=TEXT_COLOR,
        )
        s2 = MathTex(
            r"f_\theta:\mathbb{R}^d\to\mathbb{R}", font_size=MATH_FS, color=TEXT_COLOR
        )

        sentence = VGroup(s1, s2).arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        sentence.next_to(title, DOWN, buff=0.55).to_edge(LEFT, buff=1.0)

        self.play(Write(s1), run_time=1.0)
        self.play(Write(s2), run_time=0.6)
        self.next_slide()

        r_lin, lab_lin, ex_lin = row("Linear Networks:", r"f_\theta(x)=x^\top\theta")
        r_nn, lab_nn, ex_nn = row("Neural Networks:", r"f_\theta(x)=")

        rows = VGroup(r_lin, r_nn).arrange(DOWN, buff=1.5, aligned_edge=LEFT)

        # 1. Instantiate 'nn' EARLY so we can measure its width for centering
        edge_color = interpolate_color(BG_COLOR, TEXT_COLOR, 0.60)
        neuron_stroke = interpolate_color(BG_COLOR, TEXT_COLOR, 0.85)

        nn = NetworkMobject(
            layer_sizes=(3, 7, 5, 2),
            neuron_radius=0.10,
            neuron_to_neuron_buff=0.30,
            layer_to_layer_buff=1.05,
            neuron_stroke_color=neuron_stroke,
            neuron_stroke_width=2.2,
            neuron_fill_color=BLUE_E,
            neuron_fill_opacity=0.0,
            edge_color=edge_color,
            edge_stroke_width=1.2,
            edge_propagation_color=YELLOW,
            edge_propagation_time=0.6,
            brace_for_large_layers=False,
        ).deactivate()

        nn.scale(0.7)
        nn.next_to(ex_nn, RIGHT, buff=0.4).shift(DOWN * 0.05)

        # 2. Bundle the text rows and network together, then center them
        network_section = VGroup(rows, nn)
        network_section.next_to(sentence, DOWN, buff=0.55)
        network_section.set_x(0)  # This centers the entire block on the screen!

        # 3. Now play the animations in your original order
        self.play(FadeIn(r_lin, shift=UP * 0.1), run_time=0.7)
        self.next_slide()

        self.play(FadeIn(r_nn, shift=UP * 0.1), run_time=0.7)

        # Important: animate neurons first (layer-by-layer)
        self.play(
            LaggedStart(
                *[Create(layer.neurons) for layer in nn.layers],
                lag_ratio=0.18,
            ),
            run_time=0.9,
        )

        nn.edge_groups.set_stroke(opacity=0.55)

        self.play(
            LaggedStart(
                *[Create(eg) for eg in nn.edge_groups],
                lag_ratio=0.15,
            ),
            run_time=0.9,
        )
        self.play(nn.forward_pass_anim(), run_time=1.2)
        self.next_slide()

        left_arrow = (
            Arrow(start=DOWN * 2.0, end=UP * 0.25, color=RED, buff=0, stroke_width=8)
            .to_edge(LEFT, buff=0.9)
            .shift(DOWN * 0.1 + RIGHT * 1.5)
        )

        left_text = (
            Text(
                "Simplicity\nof\nanalysis",
                font_size=BODY_FS * 0.8,
                color=RED,
                weight=BOLD,
            )
            .next_to(left_arrow, LEFT, buff=0.15)
            .align_to(left_arrow, DOWN)
        )

        right_arrow = (
            Arrow(start=UP * 0.25, end=DOWN * 2.0, color=BLUE, buff=0, stroke_width=8)
            .to_edge(RIGHT, buff=1.5)
            .shift(DOWN * 0.1 + LEFT * 1)
        )

        right_text = (
            Text(
                "Expressivity\npower", font_size=BODY_FS * 0.8, color=BLUE, weight=BOLD
            )
            .next_to(right_arrow, RIGHT, buff=0.15)
            .align_to(right_arrow, UP)
        )

        self.play(FadeIn(right_arrow), FadeIn(right_text), run_time=0.6)
        self.next_slide()

        self.play(FadeIn(left_arrow), FadeIn(left_text), run_time=0.6)
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
