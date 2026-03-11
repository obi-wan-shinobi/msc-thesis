from manim import *
from manim_slides import Slide

from theme import BG_COLOR, BODY_FS, SUBTITLE_FS, TEXT_COLOR, TITLE_FS, add_logo


class ThesisIntro(Slide):
    def construct(self):
        self.camera.background_color = BG_COLOR
        Text.set_default(color=TEXT_COLOR)
        MathTex.set_default(color=TEXT_COLOR)

        title = Paragraph(
            "Training and Generalization in",
            "overparameterized neural networks",
            alignment="center",
            font_size=TITLE_FS,
            weight=BOLD,
        ).to_edge(UP, buff=1.2)

        subtitle = Text(
            "Interim Thesis Presentation",
            font_size=SUBTITLE_FS,
            color=BLUE,
        ).next_to(title, DOWN, buff=0.4)

        author = Text("Shreyas Kalvankar", font_size=BODY_FS)

        affiliation = Text(
            "MSc Computer Science",
            font_size=SUBTITLE_FS,
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
