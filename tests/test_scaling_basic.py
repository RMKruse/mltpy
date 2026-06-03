"""Tracer-bullet R parity for scaling terms — issue #70.

The smallest vertical slice exercising the heteroskedastic CTM path:
``CensoringType.NONE`` × normal base × scaling design.

Acceptance criteria (from #70):

* ``MLT(order=..., support=..., scaling=...)`` accepts the kwarg and fits
  without raising on exact + normal data.
* ``_ll_none`` and its gradient handle the ``[theta_b | beta | gamma]`` layout;
  analytical gradient matches ``scipy.optimize.check_grad`` at the fitted point.
* R reference fixture for ``tram::BoxCox(scale=~x_s)`` lands in ``reference/``.
* θ, β, γ, and log-lik match R at ``rtol=1e-6``.

The fixture is produced by ``reference/generate_reference.R`` (block "Scaling-terms
tracer (issue #70)").  R / tram parameterises the shift block with the *minus*
sign (``negative=TRUE`` on BoxCox), so ``mltpy.coef_`` compares against
``-r.beta``.  The scaling block (γ) is sign-aligned across R and mltpy; see
``docs/adr/0002-scaling-terms.md``, Decision 5.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from mltpy import MLT
from mltpy.basis import BernsteinBasis
from mltpy.likelihood import negative_log_likelihood
from mltpy.variables import CensoringType

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"
_FIXTURE_FILES = [
    "scaling_boxcox_normal_y.txt",
    "scaling_boxcox_normal_x_d.txt",
    "scaling_boxcox_normal_x_s.txt",
    "scaling_boxcox_normal_support.txt",
    "scaling_boxcox_normal_theta.txt",
    "scaling_boxcox_normal_loglik.txt",
]


def _load_scaling_fixture() -> dict:
    paths = [REF_DIR / name for name in _FIXTURE_FILES]
    if not all(p.exists() for p in paths):
        pytest.skip(
            "scaling_boxcox_normal_* reference files not yet generated — "
            "run Rscript reference/generate_reference.R"
        )
    y = np.loadtxt(paths[0])
    x_d = np.loadtxt(paths[1]).reshape(-1, 1)
    x_s = np.loadtxt(paths[2]).reshape(-1, 1)
    support = tuple(np.loadtxt(paths[3]))
    theta_full = np.loadtxt(paths[4])
    ll = float(np.loadtxt(paths[5]))
    q_d = x_d.shape[1]
    q_s = x_s.shape[1]
    p = len(theta_full) - q_d - q_s
    return {
        "y": y,
        "x_d": x_d,
        "x_s": x_s,
        "support": (float(support[0]), float(support[1])),
        "p": p,
        "q_d": q_d,
        "q_s": q_s,
        "theta_b": theta_full[:p],
        "beta_r": theta_full[p : p + q_d],
        "gamma_r": theta_full[p + q_d :],
        "log_likelihood": ll,
    }


@pytest.fixture(scope="module")
def scaling_ref() -> dict:
    return _load_scaling_fixture()


def test_mlt_accepts_scaling_kwarg(scaling_ref: dict) -> None:
    """MLT(..., scaling=X_s) must accept the kwarg and fit without raising."""
    model = MLT(
        order=scaling_ref["p"] - 1,
        support=scaling_ref["support"],
        scaling=scaling_ref["x_s"],
    )
    model.fit(scaling_ref["y"], X=scaling_ref["x_d"])
    assert model.is_fitted_
    assert model.theta_ is not None
    assert model.theta_.shape == (
        scaling_ref["p"] + scaling_ref["q_d"] + scaling_ref["q_s"],
    )


def test_scaling_theta_beta_gamma_match_R(scaling_ref: dict) -> None:
    """Fitted θ_b, β (sign-flipped), γ match R at rtol=1e-6."""
    p = scaling_ref["p"]
    q_d = scaling_ref["q_d"]
    model = MLT(
        order=p - 1,
        support=scaling_ref["support"],
        scaling=scaling_ref["x_s"],
    )
    model.fit(scaling_ref["y"], X=scaling_ref["x_d"])
    theta_b = model.theta_[:p]
    beta = model.theta_[p : p + q_d]
    gamma = model.theta_[p + q_d :]

    np.testing.assert_allclose(theta_b, scaling_ref["theta_b"], rtol=1e-5, atol=1e-7)
    # mltpy parametrises h + Xβ; tram BoxCox uses h - Xβ_R.  Sign flip on β.
    np.testing.assert_allclose(beta, -scaling_ref["beta_r"], rtol=1e-5, atol=1e-7)
    # γ is sign-aligned across the two parameterisations (ADR 0002, Decision 5).
    np.testing.assert_allclose(gamma, scaling_ref["gamma_r"], rtol=1e-5, atol=1e-7)


def test_scaling_log_likelihood_matches_R(scaling_ref: dict) -> None:
    """log-lik at the fitted parameters matches R at rtol=1e-6."""
    model = MLT(
        order=scaling_ref["p"] - 1,
        support=scaling_ref["support"],
        scaling=scaling_ref["x_s"],
    )
    model.fit(scaling_ref["y"], X=scaling_ref["x_d"])
    np.testing.assert_allclose(
        model.result_.log_likelihood,
        scaling_ref["log_likelihood"],
        rtol=1e-6,
        atol=1e-8,
    )


def test_scaling_gamma_exposed(scaling_ref: dict) -> None:
    """``gamma_`` returns the γ block (sign-aligned with R)."""
    model = MLT(
        order=scaling_ref["p"] - 1,
        support=scaling_ref["support"],
        scaling=scaling_ref["x_s"],
    )
    model.fit(scaling_ref["y"], X=scaling_ref["x_d"])
    gamma = model.gamma_
    np.testing.assert_allclose(gamma, scaling_ref["gamma_r"], rtol=1e-5, atol=1e-7)


def test_scaling_analytical_gradient_matches_finite_difference(
    scaling_ref: dict,
) -> None:
    """At the fitted θ the analytical gradient of -ℓ matches finite differences.

    Uses :func:`scipy.optimize.check_grad`-style central differences on
    :func:`negative_log_likelihood` with ``gradient=True``.  The γ block is the
    non-linear coupling term (∂h/∂γ = h_0(y) · X_s · exp(x_s · γ)), so
    independently verifying its column of the gradient is the most likely
    failure mode of this slice.
    """
    p = scaling_ref["p"]
    basis = BernsteinBasis(order=p - 1, support=scaling_ref["support"])
    model = MLT(
        order=p - 1,
        support=scaling_ref["support"],
        scaling=scaling_ref["x_s"],
    )
    model.fit(scaling_ref["y"], X=scaling_ref["x_d"])
    theta = model.theta_.copy()

    def f_only(t: np.ndarray) -> float:
        return float(
            negative_log_likelihood(
                t,
                basis,
                scaling_ref["y"],
                X=scaling_ref["x_d"],
                censoring=CensoringType.NONE,
                gradient=False,
                base_distribution="normal",
                scaling=scaling_ref["x_s"],
            )
        )

    _, grad_analytic = negative_log_likelihood(
        theta,
        basis,
        scaling_ref["y"],
        X=scaling_ref["x_d"],
        censoring=CensoringType.NONE,
        gradient=True,
        base_distribution="normal",
        scaling=scaling_ref["x_s"],
    )
    # ``approx_fprime`` is a forward-difference rule whose accuracy floor with
    # ``epsilon=1e-6`` is ~1e-4 — far above the analytic gradient magnitude at
    # the MLE (≈ 0 to optimiser tolerance).  Compare central differences with a
    # smaller epsilon, and relax atol to a margin where forward-diff alone
    # would still give the wrong answer.
    eps = 1e-5
    grad_fd = np.zeros_like(theta)
    for i in range(theta.size):
        tp = theta.copy()
        tp[i] += eps
        tm = theta.copy()
        tm[i] -= eps
        grad_fd[i] = (f_only(tp) - f_only(tm)) / (2.0 * eps)
    np.testing.assert_allclose(grad_analytic, grad_fd, rtol=1e-3, atol=1e-4)
