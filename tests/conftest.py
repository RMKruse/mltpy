"""Shared pytest fixtures."""
import numpy as np
import pytest

from pymlt.variables import CensoredData, NumericVariable, OrderedVariable, SurvivalVariable


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
