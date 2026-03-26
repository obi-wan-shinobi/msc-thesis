import numpy as np
from manim import *
from manim_slides import Slide, ThreeDSlide

from scenes import (
    NTK,
    CircleToyFourierDiagonalization,
    DeepLinearModelsAndGD,
    FiniteSampleResults,
    FiniteSampleStory,
    IntroProblemSetup,
    LemmaConcentrationBounds,
    LinearModelsAndGD,
    NonLinearModelsAndGD,
    SpectralBias,
    SupervisedLearningFramework,
    ThesisIntro,
    WhyAnalyzeTrainingDynamics,
)
from theme import BG_COLOR, BODY_FS, HEADER_FS, MATH_FS, TEXT_COLOR, add_logo
from utils import *


class ConclusionNextSteps(Slide):
    def construct(self):
        self.camera.background_color = BG_COLOR
        Text.set_default(color=TEXT_COLOR)
        MathTex.set_default(color=TEXT_COLOR)

        self.clear()
        logo = add_logo(self)  # don't move

        label_color = BLUE
        bullet_color = BLUE
        bullet_fs = BODY_FS * 0.78

        title = Text(
            "Conclusion and next steps",
            font_size=HEADER_FS,
            weight=BOLD,
        ).to_edge(UP)

        self.add(logo)
        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.6)
        self.next_slide()

        # -----------------------------
        # Bullet helper (square + Text)
        # -----------------------------
        def bullet(text, t2c=None):
            sq = Square(0.10, fill_opacity=1.0, stroke_width=0).set_color(bullet_color)
            body = Text(
                text,
                font_size=bullet_fs,
                line_spacing=1.10,
                t2c=t2c or {},
            )
            return VGroup(sq, body).arrange(RIGHT, buff=0.35, aligned_edge=UP)

        # -----------------------------
        # Short conclusion line (optional)
        # -----------------------------
        lead = Text(
            "Takeaway: for low modes, finite sampling preserves the continuum spectral picture.",
            font_size=BODY_FS * 0.80,
            color=TEXT_COLOR,
        )
        lead.next_to(title, DOWN, buff=0.60).to_edge(LEFT, buff=0.85)

        self.play(FadeIn(lead, shift=UP * 0.05), run_time=0.5)
        self.next_slide()

        # -----------------------------
        # Next steps bullets
        # -----------------------------
        b1 = bullet(
            "Tighten the concentration bounds\n(using sharper tools than crude union bounds).",
            t2c={"concentration bounds": YELLOW, "sharper tools": YELLOW},
        )

        b2 = bullet(
            "Precondition learning across modes:\nrescale steps so all eigendirections converge similarly.",
            t2c={"Precondition": YELLOW, "eigendirections": YELLOW},
        )

        b3 = bullet(
            "Finite width:\n Gaussian concentration, Chernoff-style bounds).",
            t2c={
                "Finite width": YELLOW,
                "kernel drift": YELLOW,
                "Gaussian": YELLOW,
                "Chernoff": YELLOW,
            },
        )

        bullets = VGroup(b1, b2, b3).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        bullets.next_to(lead, DOWN, buff=0.45).align_to(lead, LEFT)

        # Reveal one-by-one
        for b in bullets:
            b.set_opacity(0)

        for b in bullets:
            b.set_opacity(1)
            self.play(FadeIn(b, shift=UP * 0.05), run_time=0.55)
            self.next_slide()

        # -----------------------------
        # Tiny closing line (optional)
        # -----------------------------
        close = Text(
            "Thanks!",
            font_size=BODY_FS * 0.90,
            color=TEXT_COLOR,
        )
        close.to_edge(DOWN, buff=0.35).shift(RIGHT * 0.6 + UP)  # avoid logo region

        self.play(FadeIn(close, shift=UP * 0.05), run_time=0.4)
        self.next_slide()


class ThesisDeck(ThreeDSlide):
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

        # 4) Linear models & GD
        LinearModelsAndGD.construct(self)
        self.clear()

        # 5) Deep Linear models & GD
        DeepLinearModelsAndGD.construct(self)
        self.clear()
        self.set_camera_orientation(
            phi=0 * DEGREES, theta=-90 * DEGREES, gamma=0 * DEGREES
        )

        # 6) Non-linear models & GD
        NonLinearModelsAndGD.construct(self)
        self.clear()
        self.set_camera_orientation(
            phi=0 * DEGREES, theta=-90 * DEGREES, gamma=0 * DEGREES
        )

        # 7) Why analyze these dynamics?
        WhyAnalyzeTrainingDynamics.construct(self)
        self.clear()

        # 8) NTK
        NTK.construct(self)
        self.clear()
        self.set_camera_orientation(
            phi=0 * DEGREES, theta=-90 * DEGREES, gamma=0 * DEGREES
        )

        # 9) Spectral Bias
        SpectralBias.construct(self)
        self.clear()

        # 10) Circle Fourier
        CircleToyFourierDiagonalization.construct(self)
        self.clear()

        # 11) Finite Sampel story
        FiniteSampleStory.construct(self)
        self.clear()

        # 12) Lemma Concentration bounds
        LemmaConcentrationBounds.construct(self)
        self.clear()

        # 13) Finite sample results
        FiniteSampleResults.construct(self)
        self.clear()

        # 14) Conclusion
        ConclusionNextSteps.construct(self)
        self.clear()
