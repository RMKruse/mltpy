"""SLSQP is invariant to the scale of the observation weights — issue #137.

Multiplying every weight by ``c`` multiplies the negative log-likelihood by
``c`` and leaves the MLE where it is.  SLSQP's stopping test is absolute in
the objective (``ftol``) and its QP subproblem mixes the objective gradient
with unit-scale monotonicity rows, so a large objective used to break it:
``converged=False`` from ``w ≈ 1e4`` and a one-step exit ("Inequality
constraints incompatible") near the start point from ``w ≈ 1e6``.  The SLSQP
paths now minimise the likelihood under weights normalised to mean 1, which
has the same argmin and the scale of an unweighted fit.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mltpy import OptimizerConfig
from mltpy.basis import BernsteinBasis
from mltpy.tram import Colr, Coxph

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"
_PREFIX = "stram_bs_bs_mev_"
_HEAVY_WEIGHTS = [1e4, 1e6, 1e8]


@pytest.fixture(scope="module")
def mev() -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    paths = {
        k: REF_DIR / f"{_PREFIX}{k}.txt" for k in ("y_train", "x_train", "y_support")
    }
    if not all(p.exists() for p in paths.values()):
        pytest.skip(f"{_PREFIX}* fixtures not generated")
    a, b = np.loadtxt(paths["y_support"])
    return (
        np.loadtxt(paths["y_train"]),
        np.loadtxt(paths["x_train"]),
        (float(a), float(b)),
    )


def _slsqp() -> OptimizerConfig:
    return OptimizerConfig(solver="slsqp", random_state=0, polish=False)


def _shift(
    model: type[Coxph] | type[Colr], support: tuple[float, float]
) -> Coxph | Colr:
    return model(support=support, order=6, optimizer_config=_slsqp())


def _interaction(support: tuple[float, float]) -> Coxph:
    return Coxph(
        support=support,
        order=2,
        interacting=BernsteinBasis(order=1, support=(0.0, 1.0)),
        optimizer_config=_slsqp(),
    )


_MODELS = {
    "coxph-shift": lambda s: _shift(Coxph, s),
    "colr-shift": lambda s: _shift(Colr, s),
    "coxph-interaction": _interaction,
}


def _x(name: str, x: np.ndarray) -> np.ndarray:
    return x if name.endswith("interaction") else x.reshape(-1, 1)


@pytest.fixture(scope="module")
def unit_fits(
    mev: tuple[np.ndarray, np.ndarray, tuple[float, float]],
) -> dict[str, Coxph | Colr]:
    y, x, support = mev
    fits = {}
    for name, make in _MODELS.items():
        m = make(support)
        m.fit(y, X=_x(name, x))
        fits[name] = m
    return fits


@pytest.mark.parametrize("w", _HEAVY_WEIGHTS)
@pytest.mark.parametrize("name", list(_MODELS))
def test_uniform_heavy_weights_reach_unit_weight_mle(
    mev: tuple[np.ndarray, np.ndarray, tuple[float, float]],
    unit_fits: dict[str, Coxph | Colr],
    name: str,
    w: float,
) -> None:
    y, x, support = mev
    unit = unit_fits[name]
    heavy = _MODELS[name](support)
    heavy.fit(y, X=_x(name, x), weights=np.full(y.shape[0], w))
    assert heavy.result_ is not None and unit.result_ is not None
    assert heavy.theta_ is not None and unit.theta_ is not None
    assert heavy.result_.converged, heavy.result_.solver_message
    assert heavy.result_.n_restarts == 0
    assert_allclose(heavy.theta_, unit.theta_, rtol=1e-6, atol=1e-10)
    assert_allclose(
        heavy.result_.log_likelihood, w * unit.result_.log_likelihood, rtol=1e-8
    )


@pytest.mark.parametrize("c", [1e-6, 1e6])
def test_nonuniform_weights_are_scale_invariant(
    mev: tuple[np.ndarray, np.ndarray, tuple[float, float]], c: float
) -> None:
    """Rescaling non-uniform weights by a constant reproduces the fit, and the
    reported log-likelihood is the one under the caller's weights."""
    y, x, support = mev
    weights = np.asarray(np.random.default_rng(1).uniform(0.5, 2.0, y.shape[0]))
    base = _shift(Coxph, support)
    base.fit(y, X=x.reshape(-1, 1), weights=weights)
    scaled = _shift(Coxph, support)
    scaled.fit(y, X=x.reshape(-1, 1), weights=c * weights)
    assert base.result_ is not None and scaled.result_ is not None
    assert base.theta_ is not None and scaled.theta_ is not None
    assert base.result_.converged and scaled.result_.converged
    assert_allclose(scaled.theta_, base.theta_, rtol=1e-6, atol=1e-10)
    assert_allclose(
        scaled.result_.log_likelihood, c * base.result_.log_likelihood, rtol=1e-8
    )
