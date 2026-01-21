from manim_slides import Slide

from scenes import SupervisedLearningFramework, ThesisIntro


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
