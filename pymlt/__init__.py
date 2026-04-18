"""pymlt — Conditional Transformation Models in Python."""

from pymlt.model import (
    MLT,
    AnovaResult,
    ConditionalTransformationModel,
    ConvergenceWarning,
    NotFittedError,
    anova,
)
from pymlt.optimizer import OptimizerConfig
from pymlt.tram import BoxCox, Colr, Coxph
from pymlt.variables import (
    CensoredData,
    CensoringType,
    NumericVariable,
    OrderedVariable,
    SurvivalVariable,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "ConditionalTransformationModel",
    "MLT",
    # Tram convenience models
    "BoxCox",
    "Coxph",
    "Colr",
    # Inference
    "anova",
    "AnovaResult",
    # Exceptions / warnings
    "NotFittedError",
    "ConvergenceWarning",
    # Data types
    "CensoredData",
    "CensoringType",
    "NumericVariable",
    "OrderedVariable",
    "SurvivalVariable",
    # Config
    "OptimizerConfig",
    "__version__",
]
