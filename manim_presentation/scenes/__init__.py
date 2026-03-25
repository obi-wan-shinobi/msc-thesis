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

DeepLinearModelsAndGD = import_module(
    ".05_deep_linear_models_and_gd", __name__
).DeepLinearModelsAndGD

NonLinearModelsAndGD = import_module(
    ".06_nonlinear_models_and_gd", __name__
).NonlinearModelsAndGD

WhyAnalyzeTrainingDynamics = import_module(
    ".07_why_analyze_dynamics", __name__
).WhyAnalyzeTrainingDynamics

NTK = import_module(".08_ntk", __name__).NTK

__all__ = [
    "SupervisedLearningFramework",
    "ThesisIntro",
    "IntroProblemSetup",
    "LinearModelsAndGD",
    "DeepLinearModelsAndGD",
    "NonLinearModelsAndGD",
    "WhyAnalyzeTrainingDynamics",
    "NTK",
]
