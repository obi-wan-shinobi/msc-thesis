from importlib import import_module

ThesisIntro = import_module(".01_thesis_intro", __name__).ThesisIntro
SupervisedLearningFramework = import_module(
    ".02_supervised_learning_framework", __name__
).SupervisedLearningFramework
LinearModelsAndGD = import_module(
    ".04_linear_models_and_gd", __name__
).LinearModelsAndGD
IntroProblemSetup = import_module(
    ".03_introduction_to_problem_setup", __name__
).IntroProblemSetup

__all__ = [
    "SupervisedLearningFramework",
    "ThesisIntro",
    "IntroProblemSetup",
    "LinearModelsAndGD",
]
