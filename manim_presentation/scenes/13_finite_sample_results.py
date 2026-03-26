from manim import *
from manim_slides import Slide

from theme import BG_COLOR, BODY_FS, HEADER_FS, MATH_FS, TEXT_COLOR, add_logo
from utils import *


class FiniteSampleResults(Slide):
    def construct(self):
        self.camera.background_color = BG_COLOR
        Text.set_default(color=TEXT_COLOR)
        MathTex.set_default(color=TEXT_COLOR)

        self.clear()
        logo = add_logo(self)  # don't move

        title = Text(
            "Empirical results: concentration with sample size",
            font_size=HEADER_FS,
            weight=BOLD,
        ).to_edge(UP)

        self.add(logo)
        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.6)
        self.next_slide()

        # ---- Paths (edit if needed) ----
        img_gram_path = ASSETS_PATH / "images/concentration_plot_gram_err_max.png"
        img_comp_path = ASSETS_PATH / "images/concentration_plot_comp_err_max.png"
        img_action_path = ASSETS_PATH / "images/concentration_plot_action_err_max.png"

        gram = ImageMobject(img_gram_path)
        comp = ImageMobject(img_comp_path)
        action = ImageMobject(img_action_path)

        ims = [gram, comp, action]

        # Optional crisp resampling (if available)
        try:
            for im in ims:
                im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["lanczos"])
        except Exception:
            pass

        # ---- Layout constants ----
        THUMB_W = 4.05  # thumbnail width (3 across)
        BIG_W = 7  # zoomed width
        DIM = 0.10  # opacity for non-focused plots

        # Arrange as horizontal row (thumbnails)
        thumbs = Group(*ims)
        for im in ims:
            im.set_width(THUMB_W)

        thumbs.arrange(RIGHT, buff=0.40)
        thumbs.next_to(title, DOWN, buff=0.70).shift(DOWN * 0.15)

        # Store thumb positions for restoring
        thumb_centers = [im.get_center().copy() for im in ims]

        # Define where the "zoomed" plot should sit
        # (same vertical band, just centered and larger)
        big_center = thumbs.get_center() + DOWN * 0.15

        self.play(FadeIn(thumbs, shift=UP * 0.05), run_time=0.7)
        self.next_slide()

        def focus(idx: int):
            # Bring selected to front and enlarge; dim others.
            sel = ims[idx]
            sel.set_z_index(10)

            anims = [
                sel.animate.set_width(BIG_W).move_to(big_center),
            ]
            for j, im in enumerate(ims):
                if j != idx:
                    anims.append(im.animate.set_opacity(DIM))
            return anims

        def unfocus():
            # Restore all to thumbnails.
            anims = []
            for im, c in zip(ims, thumb_centers):
                anims.append(im.animate.set_opacity(1.0).set_width(THUMB_W).move_to(c))
                im.set_z_index(0)
            return anims

        # --- Zoom each plot one-by-one ---
        # 1) Gram
        self.play(*focus(0), run_time=0.6)
        self.next_slide()
        self.play(*unfocus(), run_time=0.5)
        self.next_slide()

        # 2) Compressed operator
        self.play(*focus(1), run_time=0.6)
        self.next_slide()
        self.play(*unfocus(), run_time=0.5)
        self.next_slide()

        # 3) Eigen-action
        self.play(*focus(2), run_time=0.6)
        self.next_slide()
        self.play(*unfocus(), run_time=0.5)
        self.next_slide()

        # ---- Takeaway ----
        takeaway = (
            Text(
                "Finite sampling: low modes behave like the continuum once n is big enough.",
                font_size=BODY_FS * 0.80,
                color=TEXT_COLOR,
            )
            .to_edge(DOWN, buff=0.35)
            .shift(RIGHT * 0.6 + UP * 1)
        )

        self.play(FadeIn(takeaway, shift=UP * 0.05), run_time=0.5)
        self.next_slide()
