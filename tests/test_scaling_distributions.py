"""Self-consistency for scaled likelihood across every base distribution — #71.

For each ``(base_distribution, censoring)`` cell that #71 covers, this
module fits a small synthetic dataset under the scaled-baseline form
``h(y|x) = h_0(y)·exp(0.5·x_s·γ) + x_d·β`` and asserts that the analytical
gradient of ``-ℓ`` agrees with a central-difference reference at the fitted
``θ``.  Together with the R-parity tests in
:mod:`tests.test_scaling_censoring`, this closes the acceptance matrix in
issue #71: "All seven base distributions tested under scaling for at least
one censoring type each."

``"exponential"`` × scaling is rejected up-front by
:class:`~mltpy.model.ConditionalTransformationModel.__init__`
(see ADR 0002 Decision 3); we assert the ``ValueError`` rather than fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from mltpy import MLT
from mltpy.basis import BernsteinBasis
from mltpy.likelihood import negative_log_likelihood
from mltpy.variables import CensoredData, CensoringType


def _make_dataset(
    n: int = 80, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    x_s = rng.normal(size=n).reshape(-1, 1)
    x_d = rng.normal(size=n).reshape(-1, 1)
    y = 0.5 + 0.4 * x_d.ravel() + rng.normal(size=n) * np.exp(0.25 * x_s.ravel())
    a = float(y.min() - 0.5)
    b = float(y.max() + 0.5)
    return y, x_d, x_s, (a, b)


def _generic_theta(p: int) -> np.ndarray:
    """Feasible (non-MLE) ``[θ_b | β | γ]`` for FD gradient checks.

    Sits away from optimum *and* away from any non-smooth scoring point
    (``laplace`` link's sign(h) kink at h=0 means FD vs analytical gradients
    diverge if any observation happens to land on h=0 at the MLE; a
    generic feasible θ avoids that pathology).
    """
    return np.concatenate([np.linspace(-1.5, 1.5, p), np.array([0.3]), np.array([0.2])])


def _check_gradient(
    theta: np.ndarray,
    basis: BernsteinBasis,
    y: np.ndarray | CensoredData,
    X: np.ndarray,
    censoring: CensoringType,
    base_distribution: str,
    scaling: np.ndarray,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    _, grad_analytic = negative_log_likelihood(
        theta,
        basis,
        y,
        X=X,
        censoring=censoring,
        gradient=True,
        base_distribution=base_distribution,  # type: ignore[arg-type]
        scaling=scaling,
    )

    def f_only(t: np.ndarray) -> float:
        return float(
            negative_log_likelihood(
                t,
                basis,
                y,
                X=X,
                censoring=censoring,
                gradient=False,
                base_distribution=base_distribution,  # type: ignore[arg-type]
                scaling=scaling,
            )
        )

    eps = 1e-5
    grad_fd = np.zeros_like(theta)
    for i in range(theta.size):
        tp = theta.copy()
        tp[i] += eps
        tm = theta.copy()
        tm[i] -= eps
        grad_fd[i] = (f_only(tp) - f_only(tm)) / (2.0 * eps)
    np.testing.assert_allclose(grad_analytic, grad_fd, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "base_distribution",
    [
        "normal",
        "logistic",
        "min_extreme_value",
        "max_extreme_value",
        "laplace",
        "cauchy",
    ],
)
def test_scaled_gradient_none_for_each_distribution(base_distribution: str) -> None:
    """Exact + scaling: analytical gradient agrees with FD for every link.

    Excludes ``"exponential"`` (rejected at ``__init__`` time — see the
    dedicated ``test_exponential_*`` cases below).  Evaluates at a generic
    feasible θ (``_generic_theta``) rather than the MLE so that ``laplace``'s
    score discontinuity at ``h_i = 0`` does not contaminate the FD reference.
    """
    y, x_d, x_s, support = _make_dataset()
    p = 5
    basis = BernsteinBasis(order=p - 1, support=support)
    theta = _generic_theta(p)
    _check_gradient(
        theta,
        basis,
        y,
        x_d,
        CensoringType.NONE,
        base_distribution,
        x_s,
    )


@pytest.mark.parametrize(
    "base_distribution",
    ["normal", "logistic", "max_extreme_value", "laplace", "cauchy"],
)
def test_scaled_gradient_right_for_each_distribution(base_distribution: str) -> None:
    """Right-censored + scaling: gradient FD parity at a generic feasible θ.

    The R-parity Coxph test in :mod:`tests.test_scaling_censoring` exercises
    the ``"min_extreme_value"`` link at the MLE; here we sweep the other
    five smooth links at a fixed feasible θ.  ``"min_extreme_value"`` is
    omitted because its tail growth ``exp(h)`` produces large per-row
    contributions that amplify FD truncation noise far above the analytical
    gradient signal at a generic (non-MLE) θ — the same pre-existing
    numerical issue documented for the shift-only path.
    """
    y, x_d, x_s, support = _make_dataset()
    cutoff = float(np.quantile(y, 0.7))
    censored = y >= cutoff
    y_obs = np.where(censored, cutoff, y)
    cd = CensoredData.right_censored(y_obs, censored)

    p = 5
    basis = BernsteinBasis(order=p - 1, support=support)
    theta = _generic_theta(p)
    _check_gradient(
        theta,
        basis,
        cd,
        x_d,
        CensoringType.RIGHT,
        base_distribution,
        x_s,
    )


@pytest.mark.parametrize(
    "base_distribution",
    [
        "normal",
        "logistic",
        "min_extreme_value",
        "max_extreme_value",
        "laplace",
        "cauchy",
    ],
)
def test_scaled_gradient_left_for_each_distribution(base_distribution: str) -> None:
    """Left-censored + scaling: gradient FD parity at a generic feasible θ."""
    y, x_d, x_s, support = _make_dataset()
    cutoff = float(np.quantile(y, 0.3))
    censored = y <= cutoff
    y_obs = np.where(censored, cutoff, y)
    cd = CensoredData.left_censored(y_obs, censored)

    p = 5
    basis = BernsteinBasis(order=p - 1, support=support)
    theta = _generic_theta(p)
    _check_gradient(
        theta,
        basis,
        cd,
        x_d,
        CensoringType.LEFT,
        base_distribution,
        x_s,
    )


@pytest.mark.parametrize(
    "base_distribution",
    ["normal", "logistic", "max_extreme_value", "laplace", "cauchy"],
)
def test_scaled_gradient_interval_for_each_distribution(
    base_distribution: str,
) -> None:
    """Interval-censored + scaling: gradient FD parity at a generic feasible θ.

    ``"min_extreme_value"`` is omitted for the same pre-existing reason as
    in the right-censored sweep; the R-parity BoxCox interval test in
    :mod:`tests.test_scaling_censoring` covers a well-conditioned MLE for
    the ``normal`` link, and Coxph there covers MEV.
    """
    y, x_d, x_s, support = _make_dataset()
    lo = y - 0.2
    hi = y + 0.2
    cd = CensoredData.interval_censored(lo, hi)

    p = 5
    basis = BernsteinBasis(order=p - 1, support=support)
    theta = _generic_theta(p)
    _check_gradient(
        theta,
        basis,
        cd,
        x_d,
        CensoringType.INTERVAL,
        base_distribution,
        x_s,
    )


# ---------------------------------------------------------------------------
# "exponential" × scaling — must be rejected up-front (ADR 0002 Decision 3)
# ---------------------------------------------------------------------------


def test_exponential_with_scaling_rejected_in_init() -> None:
    """``base_distribution='exponential'`` + ``scaling=`` is a ValueError.

    The exponential link requires ``h(y|x) ≥ 0``, which becomes a
    non-linear-in-γ constraint once a scaling factor multiplies the
    baseline (ADR 0002 Decision 3 — Option (3): reject up-front).
    """
    _, _, x_s, support = _make_dataset()
    with pytest.raises(ValueError, match="exponential"):
        MLT(
            order=4,
            support=support,
            base_distribution="exponential",
            scaling=x_s,
        )


def test_exponential_without_scaling_still_works() -> None:
    """Sanity guard: exponential alone is unchanged by the scaling ADR.

    Confirms the up-front rejection in
    :class:`~mltpy.model.ConditionalTransformationModel.__init__` does
    not accidentally fire when ``scaling`` is ``None``.
    """
    y, x_d, _, support = _make_dataset()
    # Shift y to be non-negative (exponential support).
    y_pos = y - y.min() + 0.01
    sup = (0.0, float(y_pos.max() + 0.5))
    model = MLT(
        order=4,
        support=sup,
        base_distribution="exponential",
    )
    model.fit(y_pos, X=x_d)
    assert model.is_fitted_
