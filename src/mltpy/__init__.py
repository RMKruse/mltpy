"""mltpy — Conditional Transformation Models in Python."""

from mltpy._auglag import AugLagOptions, AugLagResult
from mltpy.basis import (
    BernsteinBasis,
    InteractionBasis,
    InterceptBasis,
    LegendreBasis,
    LogBasis,
    LogBernsteinBasis,
    OneHotBasis,
    OrdinalBasis,
    PolynomialBasis,
)
from mltpy.likelihood import (
    BaseDistribution,
    InfeasibleParameterError,
    hessian,
    log_likelihood,
    negative_log_likelihood,
    score_matrix,
)
from mltpy.model import (
    MLT,
    AnovaResult,
    ConditionalTransformationModel,
    ConvergenceWarning,
    NotFittedError,
    WaldTestResult,
    anova,
)
from mltpy.optimizer import OptimizationResult, OptimizerConfig
from mltpy.tram import BoxCox, Colr, Coxph, Lehmann, Lm, Polr, Survreg
from mltpy.variables import CensoredData, CensoringType, OrderedVariable

__version__ = "0.4.0"

__all__ = [
    # Models
    "ConditionalTransformationModel",
    "MLT",
    # Tram convenience models
    "BoxCox",
    "Coxph",
    "Lehmann",
    "Colr",
    "Lm",
    "Polr",
    "Survreg",
    # Inference
    "anova",
    "AnovaResult",
    "WaldTestResult",
    # Likelihood primitives
    "log_likelihood",
    "negative_log_likelihood",
    "hessian",
    "score_matrix",
    "BaseDistribution",
    # Exceptions / warnings
    "NotFittedError",
    "ConvergenceWarning",
    "InfeasibleParameterError",
    # Data types
    "CensoredData",
    "CensoringType",
    "OrderedVariable",
    "BernsteinBasis",
    "LogBernsteinBasis",
    "OrdinalBasis",
    "OneHotBasis",
    "PolynomialBasis",
    "LegendreBasis",
    "LogBasis",
    "InterceptBasis",
    "InteractionBasis",
    # Config
    "OptimizerConfig",
    "OptimizationResult",
    "AugLagOptions",
    "AugLagResult",
    "__version__",
]
