from manim import *
from manim_slides import Slide

from theme import BG_COLOR, BODY_FS, HEADER_FS, MATH_FS, TEXT_COLOR, add_logo
from utils import *


class SpectralBias(Slide):
    def construct(self):
        # -----------------------------
        # Global styling
        # -----------------------------
        self.camera.background_color = BG_COLOR
        Text.set_default(color=TEXT_COLOR)
        MathTex.set_default(color=TEXT_COLOR)

        self.clear()
        logo = add_logo(self)

        title = Text("Spectral bias", font_size=HEADER_FS, weight=BOLD).to_edge(UP)
        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.6)
        self.add(logo)
        self.next_slide()

        def bullet(body: Mobject, bullet_color=BLUE, bullet_size=0.10, buff=0.35):
            sq = Square(
                side_length=bullet_size, fill_opacity=1.0, stroke_width=0
            ).set_color(bullet_color)

            # Small nudge helps vertical centering (esp. MathTex baselines)
            sq.shift(DOWN * 0.02)

            g = VGroup(sq, body).arrange(RIGHT, buff=buff, aligned_edge=UP)
            return g

        # -----------------------------
        # Example: your spectral-bias slide bullets
        # -----------------------------
        bullet_color = BLUE
        bullet_fs = BODY_FS * 0.80
        math_fs = MATH_FS * 0.85

        # 1) Constant NTK dynamics
        body1 = MathTex(
            r"\text{Constant NTK:}\ \dot r = -\Theta r",
            font_size=math_fs,
            tex_to_color_map={r"\Theta": YELLOW, r"\text{NTK}": YELLOW},
        )

        b1 = bullet(body1, bullet_color=bullet_color)

        # 2) Spectrum sets rates (Text + MathTex stacked)
        body2 = VGroup(
            Text("Spectrum sets rates:", font_size=bullet_fs, t2c={"Spectrum": YELLOW}),
            MathTex(
                r"\text{larger }\lambda\ \Rightarrow\ \text{learned earlier}",
                font_size=math_fs,
                tex_to_color_map={r"\lambda": YELLOW},
            ),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)

        b2 = bullet(body2, bullet_color=bullet_color)

        # 3) Width realism
        body3 = Text(
            "Infinite width isn’t realistic,\n so why care?",
            font_size=bullet_fs,
            t2c={"Infinite width": YELLOW},
        )
        b3 = bullet(body3, bullet_color=bullet_color)

        # 4) Frequency principle (empirical)
        body4 = VGroup(
            Text(
                "Empirically observed learning:",
                font_size=bullet_fs,
                t2c={"Empirically": YELLOW},
            ),
            MathTex(
                r"\text{low freq}\ \to\ \text{high freq}\ \ (\text{frequency principle})",
                font_size=math_fs,
                tex_to_color_map={r"\text{frequency principle}": YELLOW},
            ),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)

        b4 = bullet(body4, bullet_color=bullet_color)

        # 5) Beyond signals
        body5 = Text(
            "Beyond signals: bias persists without a frequency axis",
            font_size=bullet_fs,
            t2c={"bias": YELLOW},
        )
        b5 = bullet(body5, bullet_color=bullet_color)

        bullets = VGroup(b1, b2, b3, b4, b5).arrange(DOWN, buff=0.45, aligned_edge=LEFT)
        # Place wherever you want
        bullets.to_edge(LEFT, buff=0.9).shift(DOWN * 0.25)

        self.play(FadeIn(bullets[0]), run_time=0.5)
        self.next_slide()
        self.play(FadeIn(bullets[1]), run_time=0.5)
        self.next_slide()
        self.play(FadeIn(bullets[2]), run_time=0.5)
        self.next_slide()

        # -----------------------------
        # Right: image + caption
        # -----------------------------
        # Put the image file next to your script, or use an absolute/relative path.
        img_path = (
            ASSETS_PATH / "images/spectral-bias.png"
        )  # <- change to your file name
        fig = ImageMobject(img_path)

        # Keep image crisp: scale by width rather than arbitrary factor
        # fig.set_resampling_algorithm(RESAMPLING_ALGORITHMS["lanczos"])
        fig.set_width(7)  # tweak for your layout
        fig.to_edge(RIGHT, buff=0.7).shift(UP * 0.1)

        # Tiny citation caption (bottom-right under image)
        caption = Text(
            "Xu, Zhang & Luo (2022), Frequency Principle / Spectral Bias (overview).",
            font_size=BODY_FS * 0.45,
            color=TEXT_COLOR,
        )
        caption.set_opacity(0.75)
        caption.next_to(fig, DOWN, buff=0.18).align_to(fig, RIGHT)

        self.play(FadeIn(fig, shift=UP * 0.05), run_time=0.7)
        self.play(FadeIn(caption), run_time=0.35)
        self.next_slide()

        self.play(FadeIn(bullets[3]), run_time=0.5)
        self.next_slide()
        self.play(FadeIn(bullets[4]), run_time=0.5)
        self.next_slide()
