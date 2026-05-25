"""Tests for mltpy public API surface — imports and __all__ completeness."""

from __future__ import annotations

import importlib

import mltpy


def test_public_api_importable() -> None:
    """Every symbol in __all__ must be importable directly from mltpy."""
    from mltpy import (
        MLT,
        BoxCox,
        CensoredData,
        CensoringType,
        Colr,
        ConditionalTransformationModel,
        ConvergenceWarning,
        Coxph,
        NotFittedError,
        OptimizerConfig,
    )

    # Silence unused-import warnings from linters — we are testing importability.
    assert all(
        x is not None
        for x in [
            MLT,
            ConditionalTransformationModel,
            BoxCox,
            Coxph,
            Colr,
            NotFittedError,
            ConvergenceWarning,
            CensoredData,
            CensoringType,
            OptimizerConfig,
        ]
    )


def test_version_string() -> None:
    assert isinstance(mltpy.__version__, str)
    assert mltpy.__version__  # non-empty


def test_all_list_matches_imports() -> None:
    """__all__ must not reference names that don't exist on the module."""
    for name in mltpy.__all__:
        assert hasattr(mltpy, name), (
            f"mltpy.__all__ lists {name!r} but it is not defined"
        )


def test_init_reload() -> None:
    """Force re-execution of __init__.py during test-execution phase.

    importlib.reload() re-runs the module body unconditionally, so
    coverage.py records all 6 executable statements in __init__.py
    regardless of when the initial import happened.
    """
    reloaded = importlib.reload(mltpy)
    assert reloaded is mltpy  # same module object
    assert hasattr(reloaded, "MLT")
    assert hasattr(reloaded, "__version__")
    assert hasattr(reloaded, "__all__")
