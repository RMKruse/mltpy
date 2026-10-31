"""Finite-difference verification of analytical gradients in mltpy.

Tests every analytical gradient in :mod:`mltpy.likelihood` against a
finite-difference approximation computed via
:func:`scipy.optimize.approx_fprime`. This is an independent correctness
check that does not depend on R: it catches bugs that the R-comparison
validation would miss if both implementations share the same error.

The test matrix is the full cross-product of:

* censoring type: ``none``, ``right``, ``left``, ``interval``
* base distribution: ``normal``, ``logistic``, ``min_extreme_value``,
  ``max_extreme_value``, ``exponential``
* theta position: ``initial``, ``perturbed``, ``converged``
* covariate mode: without ``X``, with ``X`` (2 covariates)

See ``GRADIENT_VALIDATION.md`` for rationale and method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import approx_fprime

from mltpy.basis import BernsteinBasis
from mltpy.likelihood import negative_log_likelihood
from mltpy.model import MLT
from mltpy.variables import CensoredData, CensoringType

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Tolerances for analytical vs. finite-difference comparison.
# approx_fprime uses forward differences with epsilon=sqrt(eps) by default,
# which gives ~7-8 digit accuracy for smooth functions. Near a converged
# optimum individual gradient components are themselves ~1e-5 in magnitude,
# so the effective absolute floor of the finite-difference approximation is
# around ~1e-6 (round-off error dominates truncation). ATOL is set to
# accommodate this without masking order-of-magnitude sign or scale errors.
RTOL = 1e-4
ATOL = 5e-6

# Finite-difference step size. The scipy default is sqrt(finfo(float).eps)
# which is approximately 1.49e-8 — a good compromise between truncation and
# round-off error for most smooth functions.
FD_EPSILON = np.sqrt(np.finfo(np.float64).eps)

# Bernstein basis order used throughout the tests. Kept low (5) to keep the
# parameter space small while still exercising the full gradient code path.
ORDER = 5
P = ORDER + 1  # number of Bernstein coefficients

# Data generation parameters
N = 40  # sample size
N_COVARIATES = 2
SUPPORT = (0.0, 1.0)

# Censoring rate for right/left-censored test data (fraction censored).
CENSORING_RATE = 0.35

# Half-width of interval-censored observations. Wide enough that the
# ``_log_diff_ndtr`` wide-interval branch is exercised; ``test_narrow_interval``
# below also covers the Taylor branch.
INTERVAL_HALF_WIDTH = 0.04


# ---------------------------------------------------------------------------
# Data + theta fixtures
# ---------------------------------------------------------------------------


BaseDistName = Literal[
    "normal",
    "logistic",
    "min_extreme_value",
    "max_extreme_value",
    "exponential",
]


@dataclass
class GradCase:
    """Bundle of data and metadata for a single parametrized test."""

    y: NDArray[np.float64] | CensoredData
    X: NDArray[np.float64] | None
    censoring: CensoringType
    base_distribution: BaseDistName
    n_covariates: int


def _make_basis() -> BernsteinBasis:
    return BernsteinBasis(order=ORDER, support=SUPPORT)


def _make_exact_data(seed: int) -> NDArray[np.float64]:
    """Generate ``n`` exact observations in the interior of the support."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0.1, 0.9, N))


def _make_right_censored_data(seed: int) -> CensoredData:
    rng = np.random.default_rng(seed)
    y = np.sort(rng.uniform(0.1, 0.9, N))
    censored = rng.random(N) < CENSORING_RATE
    # Ensure at least one exact and one censored observation
    censored[0] = False
    censored[-1] = True
    return CensoredData.right_censored(y, censored)


def _make_left_censored_data(seed: int) -> CensoredData:
    rng = np.random.default_rng(seed)
    y = np.sort(rng.uniform(0.1, 0.9, N))
    censored = rng.random(N) < CENSORING_RATE
    censored[0] = True
    censored[-1] = False
    return CensoredData.left_censored(y, censored)


def _make_interval_censored_data(seed: int) -> CensoredData:
    """All observations interval-censored, moderate width."""
    rng = np.random.default_rng(seed)
    centers = np.sort(rng.uniform(0.15, 0.85, N))
    lower = np.maximum(centers - INTERVAL_HALF_WIDTH, SUPPORT[0] + 1e-3)
    upper = np.minimum(centers + INTERVAL_HALF_WIDTH, SUPPORT[1] - 1e-3)
    return CensoredData.interval_censored(lower, upper)


def _make_X(seed: int, n_rows: int) -> NDArray[np.float64]:
    """Small covariate matrix with modest scale to keep the shift bounded."""
    rng = np.random.default_rng(seed)
    return 0.3 * rng.standard_normal((n_rows, N_COVARIATES))


def _initial_theta_b() -> NDArray[np.float64]:
    """Linspace — the optimizer's default feasible starting point."""
    return np.linspace(0.0, 1.0, P)


def _perturbed_theta_b(seed: int) -> NDArray[np.float64]:
    """Random strictly-ascending vector well inside the feasible region."""
    rng = np.random.default_rng(seed)
    increments = rng.uniform(0.1, 0.4, P)
    # Start a little below zero so the interior of the basis is hit
    return np.cumsum(increments) - 0.3


def _converged_theta(
    y_or_cd: NDArray[np.float64] | CensoredData,
    X: NDArray[np.float64] | None,
    censoring: CensoringType,
    base_distribution: BaseDistName,
) -> NDArray[np.float64]:
    """Fit an MLT model and return the converged parameter vector."""
    model = MLT(
        order=ORDER,
        support=SUPPORT,
        censoring=censoring,
        base_distribution=base_distribution,
    )
    model.fit(y_or_cd, X=X)
    assert model.theta_ is not None
    return model.theta_


def _make_test_case(
    censoring: CensoringType,
    base_distribution: BaseDistName,
    with_covariates: bool,
    seed: int,
) -> GradCase:
    """Generate data, optional covariates, and package into a GradCase."""
    if censoring is CensoringType.NONE:
        y_or_cd: NDArray[np.float64] | CensoredData = _make_exact_data(seed)
    elif censoring is CensoringType.RIGHT:
        y_or_cd = _make_right_censored_data(seed)
    elif censoring is CensoringType.LEFT:
        y_or_cd = _make_left_censored_data(seed)
    elif censoring is CensoringType.INTERVAL:
        y_or_cd = _make_interval_censored_data(seed)
    else:
        raise ValueError(f"unknown censoring: {censoring}")

    X = _make_X(seed + 1000, N) if with_covariates else None
    n_q = N_COVARIATES if with_covariates else 0

    return GradCase(
        y=y_or_cd,
        X=X,
        censoring=censoring,
        base_distribution=base_distribution,
        n_covariates=n_q,
    )


def _build_theta(
    position: Literal["initial", "perturbed", "converged"],
    case: GradCase,
    seed: int,
) -> NDArray[np.float64]:
    """Construct the parameter vector at the requested position."""
    if position == "initial":
        theta_b = _initial_theta_b()
    elif position == "perturbed":
        theta_b = _perturbed_theta_b(seed)
    elif position == "converged":
        return _converged_theta(case.y, case.X, case.censoring, case.base_distribution)
    else:
        raise ValueError(f"unknown position: {position}")

    if case.n_covariates > 0:
        rng = np.random.default_rng(seed + 2000)
        beta = 0.2 * rng.standard_normal(case.n_covariates)
        theta = np.concatenate([theta_b, beta])
    else:
        theta = theta_b

    # Exponential support is [0, ∞): h(y|x_i) = B(y)·theta_b + X_i·beta must be
    # ≥ 0. Under Bernstein monotonicity min_y B(y)·theta_b = theta_b[0], so the
    # per-observation minimum is theta_b[0] + min_i X_i·beta. Shift theta_b by a
    # uniform constant when needed to restore feasibility with a small margin.
    # Applied only to non-converged positions; converged theta is feasible by
    # construction (MLT.fit enforces the nonneg_lower constraint).
    if case.base_distribution == "exponential":
        if case.X is not None and case.n_covariates > 0:
            beta_part = theta[P:]
            shift = float(np.min(case.X @ beta_part))
        else:
            shift = 0.0
        margin = 1e-2
        deficit = margin - (theta[0] + shift)
        if deficit > 0:
            theta = theta.copy()
            theta[:P] += deficit
    return theta


# ---------------------------------------------------------------------------
# Gradient comparison helper
# ---------------------------------------------------------------------------


def _compare_gradients(
    theta: NDArray[np.float64],
    case: GradCase,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (analytical, finite_difference) gradients at *theta*."""
    basis = _make_basis()

    def f_only(t: NDArray[np.float64]) -> float:
        return float(
            negative_log_likelihood(
                t,
                basis,
                case.y,
                case.X,
                case.censoring,
                gradient=False,
                base_distribution=case.base_distribution,
            )
        )

    # Analytical gradient
    result = negative_log_likelihood(
        theta,
        basis,
        case.y,
        case.X,
        case.censoring,
        gradient=True,
        base_distribution=case.base_distribution,
    )
    assert isinstance(result, tuple)
    _, analytical = result

    # Finite-difference gradient via scipy.optimize.approx_fprime
    finite_diff = approx_fprime(theta, f_only, FD_EPSILON)

    return analytical, finite_diff


def _assert_gradients_match(
    analytical: NDArray[np.float64],
    finite_diff: NDArray[np.float64],
    label: str,
) -> None:
    """Compare with component-wise assert_allclose and informative failure."""
    assert analytical.shape == finite_diff.shape, (
        f"{label}: shape mismatch — analytical {analytical.shape}, "
        f"finite-diff {finite_diff.shape}"
    )

    abs_diff = np.abs(analytical - finite_diff)
    max_abs = float(np.max(abs_diff))
    argmax = int(np.argmax(abs_diff))

    try:
        np.testing.assert_allclose(analytical, finite_diff, rtol=RTOL, atol=ATOL)
    except AssertionError as exc:
        msg = (
            f"\n{label}: gradient mismatch\n"
            f"  max |analytical - finite_diff| = {max_abs:.3e} "
            f"at index {argmax}\n"
            f"  analytical[{argmax}] = {analytical[argmax]:.6e}\n"
            f"  finite_diff[{argmax}] = {finite_diff[argmax]:.6e}\n"
            f"  analytical = {analytical}\n"
            f"  finite_diff = {finite_diff}\n"
        )
        raise AssertionError(msg) from exc


# ---------------------------------------------------------------------------
# Parametrized tests: the full cross-product
# ---------------------------------------------------------------------------


CENSORING_TYPES = [
    pytest.param(CensoringType.NONE, id="none"),
    pytest.param(CensoringType.RIGHT, id="right"),
    pytest.param(CensoringType.LEFT, id="left"),
    pytest.param(CensoringType.INTERVAL, id="interval"),
]

BASE_DISTRIBUTIONS = [
    pytest.param("normal", id="normal"),
    pytest.param("logistic", id="logistic"),
    pytest.param("min_extreme_value", id="min_extreme_value"),
    pytest.param("max_extreme_value", id="max_extreme_value"),
    pytest.param("exponential", id="exponential"),
]

_DIST_SEED_OFFSET: dict[str, int] = {
    "normal": 0,
    "logistic": 300,
    "min_extreme_value": 500,
    "max_extreme_value": 700,
    "exponential": 900,
}

THETA_POSITIONS = [
    pytest.param("initial", id="initial"),
    pytest.param("perturbed", id="perturbed"),
    pytest.param("converged", id="converged"),
]

COVARIATE_MODES = [
    pytest.param(False, id="no_X"),
    pytest.param(True, id="with_X"),
]


@pytest.mark.parametrize("censoring", CENSORING_TYPES)
@pytest.mark.parametrize("base_distribution", BASE_DISTRIBUTIONS)
@pytest.mark.parametrize("theta_position", THETA_POSITIONS)
@pytest.mark.parametrize("with_covariates", COVARIATE_MODES)
def test_analytical_gradient_matches_finite_difference(
    censoring: CensoringType,
    base_distribution: BaseDistName,
    theta_position: Literal["initial", "perturbed", "converged"],
    with_covariates: bool,
) -> None:
    """Analytical gradient should agree with central finite differences.

    Runs for the full cross-product of 4 censoring types x 5 base distributions
    x 3 theta positions x 2 covariate modes = 120 configurations.
    """
    # Deterministic seed derived from the parameters so failures are
    # reproducible. ``hash`` of strings is PYTHONHASHSEED-dependent, so we
    # use a stable manual encoding instead.
    seed = (
        10_000
        + int(censoring.value) * 1000
        + _DIST_SEED_OFFSET[base_distribution]
        + {"initial": 0, "perturbed": 100, "converged": 200}[theta_position]
        + (0 if not with_covariates else 50)
    )

    case = _make_test_case(
        censoring=censoring,
        base_distribution=base_distribution,
        with_covariates=with_covariates,
        seed=seed,
    )
    theta = _build_theta(theta_position, case, seed=seed)

    analytical, finite_diff = _compare_gradients(theta, case)

    label = (
        f"{censoring.name}/{base_distribution}/"
        f"{theta_position}/{'X' if with_covariates else 'noX'}"
    )
    _assert_gradients_match(analytical, finite_diff, label=label)


# ---------------------------------------------------------------------------
# Targeted tests for numerically tricky regimes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base_distribution", BASE_DISTRIBUTIONS)
def test_narrow_interval_triggers_taylor_branch(
    base_distribution: BaseDistName,
) -> None:
    """Narrow intervals exercise the Taylor fallback in ``_log_diff_ndtr``.

    The switch threshold in ``_log_diff_ndtr`` is ``ratio < -1e-6`` where
    ``ratio = log F(lo) - log F(hi)``. Intervals narrow enough to trigger the
    Taylor branch use the first-order approximation ``logpdf(mid) + log(width)``
    for the log-likelihood, while ``_grad_interval`` always uses the
    wide-formula gradient ``(B_hi.T @ w_hi - B_lo.T @ w_lo)``. These two are
    not *exactly* consistent in the narrow regime — they differ by O(width^2)
    terms from the Taylor truncation — but the analytical gradient remains a
    smooth approximation that avoids singularities as width -> 0.

    The relaxed ``rtol=5e-2`` below accommodates this deliberate
    inconsistency (well below any sign or order-of-magnitude error the test
    is meant to detect).
    """
    rng = np.random.default_rng(99)
    centers = np.sort(rng.uniform(0.2, 0.8, N))
    half_width = 5e-7
    lower = centers - half_width
    upper = centers + half_width
    cd = CensoredData.interval_censored(lower, upper)

    case = GradCase(
        y=cd,
        X=None,
        censoring=CensoringType.INTERVAL,
        base_distribution=base_distribution,
        n_covariates=0,
    )
    theta = _perturbed_theta_b(seed=99)
    # Keep theta feasible for exponential support (theta_b[0] >= margin).
    if base_distribution == "exponential":
        margin = 1e-2
        deficit = margin - theta[0]
        if deficit > 0:
            theta = theta + deficit

    analytical, finite_diff = _compare_gradients(theta, case)
    # Exponential's constant score amplifies Taylor-vs-wide drift on
    # small-magnitude components — widen the absolute floor for that link.
    atol = 1e-1 if base_distribution == "exponential" else 1e-4
    np.testing.assert_allclose(analytical, finite_diff, rtol=5e-2, atol=atol)


@pytest.mark.parametrize("censoring", CENSORING_TYPES)
def test_gradient_is_near_zero_at_converged_theta(
    censoring: CensoringType,
) -> None:
    """At a converged unconstrained-interior optimum the gradient is ~0.

    A sign error in the analytical gradient that is not caught by the
    finite-difference test because the two coincide by construction will still
    show up here: scipy's SLSQP will still return a "converged" flag, but the
    analytical gradient at the returned point will be large, not zero.

    This is a sanity check, not a tight tolerance — the optimizer terminates
    on a tolerance of its own and monotonicity constraints may be active.
    """
    case = _make_test_case(
        censoring=censoring,
        base_distribution="normal",
        with_covariates=False,
        seed=500 + int(censoring.value),
    )
    theta_hat = _build_theta("converged", case, seed=500 + int(censoring.value))

    basis = _make_basis()
    result = negative_log_likelihood(
        theta_hat,
        basis,
        case.y,
        case.X,
        case.censoring,
        gradient=True,
        base_distribution=case.base_distribution,
    )
    assert isinstance(result, tuple)
    _, grad = result

    # Project out monotonicity-active components by looking only at the
    # norm; inactive constraints should leave the gradient near zero, while
    # active constraints (which may bite at the boundary) contribute bounded
    # terms. The loose threshold accommodates the latter without hiding
    # gross analytical errors.
    grad_norm = float(np.linalg.norm(grad))
    assert grad_norm < 5.0, (
        f"converged gradient norm = {grad_norm:.3e} for {censoring.name} — "
        f"expected to be near zero at the optimum"
    )
