"""pymlt — Conditional Transformation Models in Python."""

from pymlt.basis import OrdinalBasis
from pymlt.likelihood import (
    InfeasibleParameterError,
    hessian,
    log_likelihood,
    negative_log_likelihood,
    score_matrix,
)
from pymlt.model import (
    MLT,
    AnovaResult,
    ConditionalTransformationModel,
    ConvergenceWarning,
    NotFittedError,
    anova,
)
from pymlt.optimizer import OptimizerConfig
from pymlt.tram import BoxCox, Colr, Coxph, Lm, Polr
from pymlt.variables import CensoredData, CensoringType, OrderedVariable

__version__ = "0.3.0"

__all__ = [
    # Models
    "ConditionalTransformationModel",
    "MLT",
    # Tram convenience models
    "BoxCox",
    "Coxph",
    "Colr",
    "Lm",
    "Polr",
    # Inference
    "anova",
    "AnovaResult",
    # Likelihood primitives
    "log_likelihood",
    "negative_log_likelihood",
    "hessian",
    "score_matrix",
    # Exceptions / warnings
    "NotFittedError",
    "ConvergenceWarning",
    "InfeasibleParameterError",
    # Data types
    "CensoredData",
    "CensoringType",
    "OrderedVariable",
    "OrdinalBasis",
    # Config
    "OptimizerConfig",
    "__version__",
]
