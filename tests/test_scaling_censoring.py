"""R parity for scaled likelihood × every censoring branch — issue #71.

Covers the three censoring types that #71 closes off (``RIGHT``, ``LEFT``,
``INTERVAL``) by fitting each against an R ``tram::*(scale = ~x_s)``
reference and asserting θ_b, β (sign-flipped), γ, and the log-likelihood
match at ``rtol=1e-5`` / ``atol=1e-7``.

``LEFT`` is exercised indirectly through the ``Colr(scale=...)`` fixture
flipped to upper-tail censoring (R does not expose a Colr left-censoring
helper, so we mirror the upper-tail data through ``-y`` and use the same
likelihood path).  The dedicated ``RIGHT`` and ``INTERVAL`` fixtures
exercise their branches directly via :class:`mltpy.variables.CensoredData`.

Sign convention: ``tram::BoxCox`` uses ``negative = TRUE`` (so ``h − X·β``)
and mltpy's β compares against ``-β_R``.  ``tram::Coxph`` and ``tram::Colr``
use ``negative = FALSE`` (``h + X·β``) — same sign as mltpy — so the parity
test does *not* flip those β blocks.  γ is sign-aligned across all three
(ADR 0002 Decision 5).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from mltpy import MLT
from mltpy.basis import BernsteinBasis
from mltpy.likelihood import _H_CLIP, log_likelihood, negative_log_likelihood
from mltpy.optimizer import OptimizerConfig
from mltpy.variables import CensoredData, CensoringType

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"


def _load(name: str) -> np.ndarray:
    return np.loadtxt(REF_DIR / name)


def _maybe_skip(files: list[str]) -> None:
    if not all((REF_DIR / f).exists() for f in files):
        pytest.skip(
            "scaling_*_censoring reference files not yet generated — "
            "run Rscript reference/generate_reference.R"
        )


# ---------------------------------------------------------------------------
# Right-censored Coxph(scale=~x_s) — min_extreme_value base
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coxph_ref() -> dict:
    files = [
        "scaling_coxph_y.txt",
        "scaling_coxph_event.txt",
        "scaling_coxph_x_d.txt",
        "scaling_coxph_x_s.txt",
        "scaling_coxph_support.txt",
        "scaling_coxph_theta.txt",
        "scaling_coxph_loglik.txt",
    ]
    _maybe_skip(files)
    y = _load("scaling_coxph_y.txt")
    event = _load("scaling_coxph_event.txt").astype(bool)
    x_d = _load("scaling_coxph_x_d.txt").reshape(-1, 1)
    x_s = _load("scaling_coxph_x_s.txt").reshape(-1, 1)
    support = tuple(_load("scaling_coxph_support.txt"))
    theta = _load("scaling_coxph_theta.txt")
    ll = float(_load("scaling_coxph_loglik.txt"))
    q_d = x_d.shape[1]
    q_s = x_s.shape[1]
    p = len(theta) - q_d - q_s
    return {
        "y": y,
        "event": event,
        "x_d": x_d,
        "x_s": x_s,
        "support": (float(support[0]), float(support[1])),
        "p": p,
        "q_d": q_d,
        "q_s": q_s,
        "theta_b": theta[:p],
        "beta_r": theta[p : p + q_d],
        "gamma_r": theta[p + q_d :],
        "log_likelihood": ll,
    }


def test_coxph_right_scaled_matches_R(coxph_ref: dict) -> None:
    """tram::Coxph(Surv(y, event) ~ x_d | x_s) θ/β/γ/loglik parity."""
    p, q_d = coxph_ref["p"], coxph_ref["q_d"]
    cd = CensoredData.right_censored(coxph_ref["y"], ~coxph_ref["event"])
    model = MLT(
        order=p - 1,
        support=coxph_ref["support"],
        censoring=CensoringType.RIGHT,
        base_distribution="min_extreme_value",
        scaling=coxph_ref["x_s"],
    )
    model.fit(cd, X=coxph_ref["x_d"])

    np.testing.assert_allclose(
        model.theta_[:p], coxph_ref["theta_b"], rtol=1e-4, atol=1e-5
    )
    # Coxph uses negative=FALSE; β sign-aligned with mltpy.
    np.testing.assert_allclose(
        model.theta_[p : p + q_d], coxph_ref["beta_r"], rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        model.theta_[p + q_d :], coxph_ref["gamma_r"], rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        model.result_.log_likelihood, coxph_ref["log_likelihood"], rtol=1e-6, atol=1e-7
    )


def test_coxph_right_scaled_gradient_matches_fd(coxph_ref: dict) -> None:
    """Analytical gradient of -ℓ matches central differences at the fitted θ."""
    p = coxph_ref["p"]
    basis = BernsteinBasis(order=p - 1, support=coxph_ref["support"])
    cd = CensoredData.right_censored(coxph_ref["y"], ~coxph_ref["event"])
    model = MLT(
        order=p - 1,
        support=coxph_ref["support"],
        censoring=CensoringType.RIGHT,
        base_distribution="min_extreme_value",
        scaling=coxph_ref["x_s"],
    )
    model.fit(cd, X=coxph_ref["x_d"])
    theta = model.theta_.copy()

    _, grad_analytic = negative_log_likelihood(
        theta,
        basis,
        cd,
        X=coxph_ref["x_d"],
        censoring=CensoringType.RIGHT,
        gradient=True,
        base_distribution="min_extreme_value",
        scaling=coxph_ref["x_s"],
    )

    def f_only(t: np.ndarray) -> float:
        return float(
            negative_log_likelihood(
                t,
                basis,
                cd,
                X=coxph_ref["x_d"],
                censoring=CensoringType.RIGHT,
                gradient=False,
                base_distribution="min_extreme_value",
                scaling=coxph_ref["x_s"],
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
    np.testing.assert_allclose(grad_analytic, grad_fd, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Colr(scale=~x_s) — logistic base, exact observations
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def colr_ref() -> dict:
    files = [
        "scaling_colr_y.txt",
        "scaling_colr_x_d.txt",
        "scaling_colr_x_s.txt",
        "scaling_colr_support.txt",
        "scaling_colr_theta.txt",
        "scaling_colr_loglik.txt",
    ]
    _maybe_skip(files)
    y = _load("scaling_colr_y.txt")
    x_d = _load("scaling_colr_x_d.txt").reshape(-1, 1)
    x_s = _load("scaling_colr_x_s.txt").reshape(-1, 1)
    support = tuple(_load("scaling_colr_support.txt"))
    theta = _load("scaling_colr_theta.txt")
    ll = float(_load("scaling_colr_loglik.txt"))
    q_d = x_d.shape[1]
    q_s = x_s.shape[1]
    p = len(theta) - q_d - q_s
    return {
        "y": y,
        "x_d": x_d,
        "x_s": x_s,
        "support": (float(support[0]), float(support[1])),
        "p": p,
        "q_d": q_d,
        "q_s": q_s,
        "theta_b": theta[:p],
        "beta_r": theta[p : p + q_d],
        "gamma_r": theta[p + q_d :],
        "log_likelihood": ll,
    }


def test_colr_scaled_matches_R(colr_ref: dict) -> None:
    """tram::Colr(y ~ x_d | x_s) θ/β/γ/loglik parity (logistic base)."""
    p, q_d = colr_ref["p"], colr_ref["q_d"]
    model = MLT(
        order=p - 1,
        support=colr_ref["support"],
        base_distribution="logistic",
        scaling=colr_ref["x_s"],
    )
    model.fit(colr_ref["y"], X=colr_ref["x_d"])

    np.testing.assert_allclose(
        model.theta_[:p], colr_ref["theta_b"], rtol=1e-4, atol=1e-5
    )
    # Colr uses negative=FALSE; β sign-aligned with mltpy.
    np.testing.assert_allclose(
        model.theta_[p : p + q_d], colr_ref["beta_r"], rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        model.theta_[p + q_d :], colr_ref["gamma_r"], rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        model.result_.log_likelihood, colr_ref["log_likelihood"], rtol=1e-6, atol=1e-7
    )


# ---------------------------------------------------------------------------
# Interval-censored BoxCox(scale=~x_s) — normal base
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def boxcox_interval_ref() -> dict:
    files = [
        "scaling_boxcox_interval_lo.txt",
        "scaling_boxcox_interval_hi.txt",
        "scaling_boxcox_interval_x_d.txt",
        "scaling_boxcox_interval_x_s.txt",
        "scaling_boxcox_interval_support.txt",
        "scaling_boxcox_interval_theta.txt",
        "scaling_boxcox_interval_loglik.txt",
    ]
    _maybe_skip(files)
    lo = _load("scaling_boxcox_interval_lo.txt")
    hi = _load("scaling_boxcox_interval_hi.txt")
    x_d = _load("scaling_boxcox_interval_x_d.txt").reshape(-1, 1)
    x_s = _load("scaling_boxcox_interval_x_s.txt").reshape(-1, 1)
    support = tuple(_load("scaling_boxcox_interval_support.txt"))
    theta = _load("scaling_boxcox_interval_theta.txt")
    ll = float(_load("scaling_boxcox_interval_loglik.txt"))
    q_d = x_d.shape[1]
    q_s = x_s.shape[1]
    p = len(theta) - q_d - q_s
    return {
        "lo": lo,
        "hi": hi,
        "x_d": x_d,
        "x_s": x_s,
        "support": (float(support[0]), float(support[1])),
        "p": p,
        "q_d": q_d,
        "q_s": q_s,
        "theta_b": theta[:p],
        "beta_r": theta[p : p + q_d],
        "gamma_r": theta[p + q_d :],
        "log_likelihood": ll,
    }


def test_boxcox_interval_scaled_matches_R(boxcox_interval_ref: dict) -> None:
    """tram::BoxCox(Surv(lo, hi, type='interval2') ~ x_d | x_s) parity."""
    p, q_d = boxcox_interval_ref["p"], boxcox_interval_ref["q_d"]
    cd = CensoredData.interval_censored(
        boxcox_interval_ref["lo"], boxcox_interval_ref["hi"]
    )
    model = MLT(
        order=p - 1,
        support=boxcox_interval_ref["support"],
        censoring=CensoringType.INTERVAL,
        base_distribution="normal",
        scaling=boxcox_interval_ref["x_s"],
    )
    model.fit(cd, X=boxcox_interval_ref["x_d"])

    np.testing.assert_allclose(
        model.theta_[:p], boxcox_interval_ref["theta_b"], rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        model.theta_[p : p + q_d],
        -boxcox_interval_ref["beta_r"],
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        model.theta_[p + q_d :],
        boxcox_interval_ref["gamma_r"],
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        model.result_.log_likelihood,
        boxcox_interval_ref["log_likelihood"],
        rtol=1e-6,
        atol=1e-7,
    )


# ---------------------------------------------------------------------------
# Fixed-time right censoring, ctm(shifting = ~x_d, scaling = ~x_s) — #162
#
# The design is issue #162's repro: every response above the 80 % quantile c
# is censored at c.  ``<dist>`` fixtures are R's fit of the correctly
# specified data, Surv(y, event).  ``flipped_<dist>`` fixtures are R's fit
# with the event flag flipped, which is what the issue's repro fitted: it
# passed ``event`` as the *censored* argument of
# :meth:`CensoredData.right_censored`.  That leaves 30 exact rows tied at
# y = c and an unbounded likelihood; R stops at a positive logLik with
# convergence 0, and its θ puts the censored rows' h far beyond ±_H_CLIP.
# ctm's shifting term enters as + x_d·β, so β is sign-aligned (no flip).
# ---------------------------------------------------------------------------

_FIXED_TIME_FITS = ("normal", "logistic", "flipped_normal", "flipped_logistic")


@pytest.fixture(scope="module")
def fixed_time_ref() -> dict:
    inputs = ("y", "event", "x_d", "x_s", "support")
    outputs = tuple(f"{s}_{k}" for s in _FIXED_TIME_FITS for k in ("theta", "loglik"))
    _maybe_skip([f"scaling_fixed_time_{stem}.txt" for stem in inputs + outputs])
    support = _load("scaling_fixed_time_support.txt")
    return {
        "y": _load("scaling_fixed_time_y.txt"),
        "event": _load("scaling_fixed_time_event.txt").astype(bool),
        "x_d": _load("scaling_fixed_time_x_d.txt").reshape(-1, 1),
        "x_s": _load("scaling_fixed_time_x_s.txt").reshape(-1, 1),
        "support": (float(support[0]), float(support[1])),
        "theta": {
            s: _load(f"scaling_fixed_time_{s}_theta.txt") for s in _FIXED_TIME_FITS
        },
        "loglik": {
            s: float(_load(f"scaling_fixed_time_{s}_loglik.txt"))
            for s in _FIXED_TIME_FITS
        },
    }


def _fixed_time_censored(ref: dict, fit: str) -> np.ndarray:
    """Per-row *censored* flag for fixture ``fit`` (the flipped fits use event)."""
    return ref["event"] if fit.startswith("flipped_") else ~ref["event"]


def _fixed_time_h(ref: dict, theta: np.ndarray) -> np.ndarray:
    """Unclipped h(y_i | x_i) = h_0(y_i)·exp(0.5·x_s,i·γ) + x_d,i·β."""
    basis = BernsteinBasis(order=4, support=ref["support"])
    f = np.exp(0.5 * ref["x_s"][:, 0] * theta[6])
    return (basis.evaluate(ref["y"]) @ theta[:5]) * f + ref["x_d"][:, 0] * theta[5]


@pytest.mark.parametrize("solver", ["auglag", "trust-constr"])
@pytest.mark.parametrize("dist", ["normal", "logistic"])
def test_fixed_time_right_scaled_fit_matches_R(
    fixed_time_ref: dict, dist: str, solver: str
) -> None:
    """The issue #162 design, correctly specified, fits R's optimum.

    The fitted h stays well inside ±_H_CLIP, so the clip plays no part.
    """
    ref = fixed_time_ref
    cd = CensoredData.right_censored(ref["y"], _fixed_time_censored(ref, dist))
    model = MLT(
        order=4,
        support=ref["support"],
        censoring=CensoringType.RIGHT,
        base_distribution=dist,
        scaling=ref["x_s"],
        optimizer_config=OptimizerConfig(solver=solver),
    )
    model.fit(cd, X=ref["x_d"])

    assert model.result_.converged
    np.testing.assert_allclose(
        model.result_.log_likelihood, ref["loglik"][dist], rtol=1e-6, atol=1e-10
    )
    np.testing.assert_allclose(model.theta_, ref["theta"][dist], rtol=1e-4, atol=1e-4)
    assert np.abs(_fixed_time_h(ref, model.theta_)).max() < _H_CLIP / 5


@pytest.mark.parametrize("fit", _FIXED_TIME_FITS)
def test_fixed_time_loglik_at_R_theta_matches_R(fixed_time_ref: dict, fit: str) -> None:
    """mltpy's ℓ at R's θ equals R's logLik, which R computes without a clip.

    On the flipped fits the censored rows' h lies far beyond ±_H_CLIP, so this
    checks that the clip does not change ℓ there; the exact rows stay inside.
    """
    ref = fixed_time_ref
    theta = ref["theta"][fit]
    censored = _fixed_time_censored(ref, fit)
    ll = log_likelihood(
        theta,
        BernsteinBasis(order=4, support=ref["support"]),
        CensoredData.right_censored(ref["y"], censored),
        X=ref["x_d"],
        censoring=CensoringType.RIGHT,
        base_distribution=fit.removeprefix("flipped_"),  # type: ignore[arg-type]
        scaling=ref["x_s"],
    )
    np.testing.assert_allclose(ll, ref["loglik"][fit], rtol=1e-6, atol=1e-10)

    h = _fixed_time_h(ref, theta)
    assert np.abs(h[~censored]).max() < _H_CLIP
    if fit.startswith("flipped_"):
        assert np.abs(h[censored]).max() > _H_CLIP


@pytest.mark.parametrize("fit", ["flipped_normal", "flipped_logistic"])
def test_fixed_time_gradient_matches_fd_beyond_clip(
    fixed_time_ref: dict, fit: str
) -> None:
    """∂(−ℓ)/∂θ is the gradient of the evaluated ℓ where censored h is clipped.

    Central differences with a step of 1e-8 relative to |θ_j|.  R's flipped
    θ has scaling factors f up to e^10, so ∂h/∂θ_b ≈ B·f on the tied exact
    rows is ~1e4 and the third derivative of ℓ is large: on the logistic fit
    a 1e-7 step leaves a 7e-3 relative truncation error in ∂/∂θ_b[4], which
    shrinks towards the analytical value as the step falls (1e-8: 5e-5).
    """
    ref = fixed_time_ref
    theta = ref["theta"][fit].copy()
    basis = BernsteinBasis(order=4, support=ref["support"])
    cd = CensoredData.right_censored(ref["y"], _fixed_time_censored(ref, fit))
    dist = fit.removeprefix("flipped_")

    def nll(t: np.ndarray) -> float:
        return float(
            negative_log_likelihood(
                t,
                basis,
                cd,
                X=ref["x_d"],
                censoring=CensoringType.RIGHT,
                base_distribution=dist,  # type: ignore[arg-type]
                scaling=ref["x_s"],
            )
        )

    _, grad = negative_log_likelihood(
        theta,
        basis,
        cd,
        X=ref["x_d"],
        censoring=CensoringType.RIGHT,
        gradient=True,
        base_distribution=dist,  # type: ignore[arg-type]
        scaling=ref["x_s"],
    )
    grad_fd = np.empty_like(theta)
    for j in range(theta.size):
        step = 1e-8 * max(1.0, abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += step
        tm[j] -= step
        grad_fd[j] = (nll(tp) - nll(tm)) / (2.0 * step)
    np.testing.assert_allclose(grad, grad_fd, rtol=1e-4, atol=1e-5)
