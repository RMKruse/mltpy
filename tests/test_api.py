"""Tests for pymlt public API surface — imports and __all__ completeness."""
from __future__ import annotations

import importlib

import pymlt


def test_public_api_importable() -> None:
    """Every symbol in __all__ must be importable directly from pymlt."""
    from pymlt import (
        MLT,
        BoxCox,
        CensoredData,
        CensoringType,
        Colr,
        ConditionalTransformationModel,
        ConvergenceWarning,
        Coxph,
        NotFittedError,
        NumericVariable,
        OptimizerConfig,
        OrderedVariable,
        SurvivalVariable,
    )
    # Silence unused-import warnings from linters — we are testing importability.
    assert all(
        x is not None for x in [
            MLT, ConditionalTransformationModel,
            BoxCox, Coxph, Colr,
            NotFittedError, ConvergenceWarning,
            CensoredData, CensoringType,
            NumericVariable, OrderedVariable, SurvivalVariable,
            OptimizerConfig,
        ]
    )


def test_version_string() -> None:
    assert isinstance(pymlt.__version__, str)
    assert pymlt.__version__  # non-empty


def test_all_list_matches_imports() -> None:
    """__all__ must not reference names that don't exist on the module."""
    for name in pymlt.__all__:
        assert hasattr(pymlt, name), f"pymlt.__all__ lists {name!r} but it is not defined"


def test_init_reload() -> None:
    """Force re-execution of __init__.py during test-execution phase.

    importlib.reload() re-runs the module body unconditionally, so
    coverage.py records all 6 executable statements in __init__.py
    regardless of when the initial import happened.
    """
    reloaded = importlib.reload(pymlt)
    assert reloaded is pymlt          # same module object
    assert hasattr(reloaded, "MLT")
    assert hasattr(reloaded, "__version__")
    assert hasattr(reloaded, "__all__")
