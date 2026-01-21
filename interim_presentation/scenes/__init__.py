from importlib import import_module

ThesisIntro = import_module(".01_thesis_intro", __name__).ThesisIntro
SupervisedLearningFramework = import_module(
    ".02_supervised_learning_framework", __name__
).SupervisedLearningFramework

__all__ = ["SupervisedLearningFramework", "ThesisIntro"]
