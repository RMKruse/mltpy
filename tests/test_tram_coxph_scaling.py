"""R parity for ``Coxph(scaling=X_s)`` — issue #74.

End-to-end coverage of the ``tram::Coxph(Surv(y, event) ~ x_d | x_s, data,
support, order)`` convenience surface: the kwarg must thread through to the
scaled-baseline likelihood (#71) and the scaled-predict path (#72) without
callers having to reach for :class:`mltpy.MLT` directly.

Reference data lives in ``reference/scaling_coxph_*`` and is produced by
``reference/generate_reference.R``.

Sign conventions:

* ``tram::Coxph`` uses ``negative = FALSE`` (so ``h + X·β``).  mltpy
  parametrises ``h + X·β`` identically, so β is sign-aligned across the
  two parameterisations.
* γ is sign-aligned across the two parameterisations (ADR 0002, Decision 5).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from mltpy.basis import BernsteinBasis
from mltpy.tram import Coxph
from mltpy.variables import CensoredData

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"


_FIT_FILES = [
    "scaling_coxph_y.txt",
    "scaling_coxph_event.txt",
    "scaling_coxph_x_d.txt",
    "scaling_coxph_x_s.txt",
    "scaling_coxph_support.txt",
    "scaling_coxph_theta.txt",
    "scaling_coxph_loglik.txt",
]


def _load(name: str) -> np.ndarray:
    return np.loadtxt(REF_DIR / name)


def _maybe_skip(files: list[str]) -> None:
    if not all((REF_DIR / f).exists() for f in files):
        pytest.skip(
            "scaling_coxph_* reference files not yet generated — "
            "run Rscript reference/generate_reference.R"
        )


@pytest.fixture(scope="module")
def coxph_scaling_ref() -> dict:
    _maybe_skip(_FIT_FILES)
    y = _load("scaling_coxph_y.txt")
    event = _load("scaling_coxph_event.txt").astype(bool)
    x_d = _load("scaling_coxph_x_d.txt").reshape(-1, 1)
    x_s = _load("scaling_coxph_x_s.txt").reshape(-1, 1)
    support = tuple(_load("scaling_coxph_support.txt"))
    theta_full = _load("scaling_coxph_theta.txt")
    ll = float(_load("scaling_coxph_loglik.txt"))
    q_d = x_d.shape[1]
    q_s = x_s.shape[1]
    p = len(theta_full) - q_d - q_s
    return {
        "y": y,
        "event": event,
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
def fitted_coxph(coxph_scaling_ref: dict) -> Coxph:
    p = coxph_scaling_ref["p"]
    model = Coxph(
        support=coxph_scaling_ref["support"],
        order=p - 1,
        scaling=coxph_scaling_ref["x_s"],
    )
    cd = CensoredData.right_censored(
        coxph_scaling_ref["y"], ~coxph_scaling_ref["event"]
    )
    model.fit(cd, X=coxph_scaling_ref["x_d"])
    return model


def test_coxph_accepts_scaling_kwarg(coxph_scaling_ref: dict) -> None:
    """Coxph(support=..., scaling=X_s).fit(cd, X=X_d) runs without raising."""
    p = coxph_scaling_ref["p"]
    model = Coxph(
        support=coxph_scaling_ref["support"],
        order=p - 1,
        scaling=coxph_scaling_ref["x_s"],
    )
    cd = CensoredData.right_censored(
        coxph_scaling_ref["y"], ~coxph_scaling_ref["event"]
    )
    model.fit(cd, X=coxph_scaling_ref["x_d"])
    assert model.is_fitted_
    assert model.theta_ is not None
    assert model.theta_.shape == (
        coxph_scaling_ref["p"] + coxph_scaling_ref["q_d"] + coxph_scaling_ref["q_s"],
    )


def test_coxph_theta_beta_gamma_match_R(
    fitted_coxph: Coxph, coxph_scaling_ref: dict
) -> None:
    """Fitted θ_b, β, γ match R at rtol=1e-4/atol=1e-5 (right-censored fit)."""
    p = coxph_scaling_ref["p"]
    q_d = coxph_scaling_ref["q_d"]
    assert fitted_coxph.theta_ is not None
    theta_b = fitted_coxph.theta_[:p]
    beta = fitted_coxph.theta_[p : p + q_d]
    gamma = fitted_coxph.theta_[p + q_d :]

    np.testing.assert_allclose(
        theta_b, coxph_scaling_ref["theta_b"], rtol=1e-4, atol=1e-5
    )
    # tram::Coxph uses negative=FALSE; β sign-aligned with mltpy.
    np.testing.assert_allclose(beta, coxph_scaling_ref["beta_r"], rtol=1e-4, atol=1e-5)
    # γ is sign-aligned across the two parameterisations.
    np.testing.assert_allclose(
        gamma, coxph_scaling_ref["gamma_r"], rtol=1e-4, atol=1e-5
    )


def test_coxph_log_likelihood_matches_R(
    fitted_coxph: Coxph, coxph_scaling_ref: dict
) -> None:
    """log-lik at the fitted parameters matches R at rtol=1e-6."""
    assert fitted_coxph.result_ is not None
    np.testing.assert_allclose(
        fitted_coxph.result_.log_likelihood,
        coxph_scaling_ref["log_likelihood"],
        rtol=1e-6,
        atol=1e-7,
    )


def test_coxph_gamma_property(fitted_coxph: Coxph, coxph_scaling_ref: dict) -> None:
    """``gamma_`` returns the γ block (sign-aligned with R)."""
    np.testing.assert_allclose(
        fitted_coxph.gamma_, coxph_scaling_ref["gamma_r"], rtol=1e-4, atol=1e-5
    )


def test_coxph_feature_names_scaling_default(fitted_coxph: Coxph) -> None:
    """``feature_names_scaling_`` falls back to ``X1, X2, ...`` for ndarray input."""
    assert fitted_coxph.feature_names_scaling_ == ["X1"]


def test_coxph_feature_names_scaling_from_dataframe(
    coxph_scaling_ref: dict,
) -> None:
    """A pandas DataFrame's column names propagate to ``feature_names_scaling_``."""
    pd = pytest.importorskip("pandas")
    p = coxph_scaling_ref["p"]
    x_s_df = pd.DataFrame(coxph_scaling_ref["x_s"], columns=["het_x"])
    model = Coxph(
        support=coxph_scaling_ref["support"],
        order=p - 1,
        scaling=x_s_df,
    )
    cd = CensoredData.right_censored(
        coxph_scaling_ref["y"], ~coxph_scaling_ref["event"]
    )
    model.fit(cd, X=coxph_scaling_ref["x_d"])
    assert model.feature_names_scaling_ == ["het_x"]


def test_coxph_gamma_raises_without_scaling(coxph_scaling_ref: dict) -> None:
    """Accessing ``gamma_`` on a non-scaling fit raises ValueError."""
    p = coxph_scaling_ref["p"]
    model = Coxph(support=coxph_scaling_ref["support"], order=p - 1)
    cd = CensoredData.right_censored(
        coxph_scaling_ref["y"], ~coxph_scaling_ref["event"]
    )
    model.fit(cd, X=coxph_scaling_ref["x_d"])
    with pytest.raises(ValueError, match="scaling="):
        _ = model.gamma_
    with pytest.raises(ValueError, match="scaling="):
        _ = model.feature_names_scaling_


def test_coxph_survival_demonstrates_non_PH(
    fitted_coxph: Coxph, coxph_scaling_ref: dict
) -> None:
    """``survival()`` at distinct ``x_s`` values diverges from the PH baseline.

    With ``scaling=x_s``, the conditional survival curves cannot be uniformly
    rescaled copies of one another — the hazard ratio between two ``x_s``
    values varies with ``t``.  We verify this by comparing the log-survival
    ratio ``log S(t | x_s_high) / log S(t | x_s_low)`` at two well-separated
    time points; under proportional hazards this ratio is constant in ``t``.
    """
    a, b = coxph_scaling_ref["support"]
    # Two well-separated interior grid points; replicate covariates per row.
    t_grid = np.array([a + 0.2 * (b - a), a + 0.8 * (b - a)])
    x_d_zero = np.zeros((t_grid.size, coxph_scaling_ref["q_d"]))
    x_s_low = np.full((t_grid.size, coxph_scaling_ref["q_s"]), -1.5)
    x_s_high = np.full((t_grid.size, coxph_scaling_ref["q_s"]), 1.5)

    s_low = fitted_coxph.survival(t_grid, X=x_d_zero, X_scale=x_s_low).ravel()
    s_high = fitted_coxph.survival(t_grid, X=x_d_zero, X_scale=x_s_high).ravel()

    # Both finite and in (0, 1) — otherwise the ratio test is uninformative.
    assert np.all((s_low > 0.0) & (s_low < 1.0))
    assert np.all((s_high > 0.0) & (s_high < 1.0))
    # log-survival ratios at the two time points should differ — non-PH.
    ratio = np.log(s_high) / np.log(s_low)
    assert abs(ratio[0] - ratio[1]) > 0.01, (
        "log-survival ratio constant across t — PH not violated by scaling"
    )


def test_coxph_survival_matches_R(fitted_coxph: Coxph, coxph_scaling_ref: dict) -> None:
    """``survival()`` matches ``1 - predict(..., what='distribution')``."""
    # New data: two contrasting scaling-covariate rows at fixed x_d.
    x_d_new = np.zeros((2, coxph_scaling_ref["q_d"]))
    x_s_new = np.array([[-1.0], [1.0]])
    a, b = coxph_scaling_ref["support"]
    t_grid = np.linspace(a + 0.05 * (b - a), a + 0.95 * (b - a), 4)

    # Broadcast (k, m) -> (k*m,) per existing predict convention.
    m = x_d_new.shape[0]
    k = t_grid.shape[0]
    y_rep = np.repeat(t_grid[:, None], m, axis=1).ravel()
    X_d_rep = np.tile(x_d_new, (k, 1))
    X_s_rep = np.tile(x_s_new, (k, 1))

    surv = fitted_coxph.survival(y_rep, X=X_d_rep, X_scale=X_s_rep)
    cdf = fitted_coxph.predict(
        y_rep, X_new=X_d_rep, X_scale_new=X_s_rep, what="distribution"
    )
    np.testing.assert_allclose(surv, 1.0 - cdf, rtol=1e-10, atol=1e-12)


def test_coxph_survival_requires_X_scale(
    fitted_coxph: Coxph, coxph_scaling_ref: dict
) -> None:
    """``survival()`` errors when X_scale is omitted on a scaling fit."""
    a, b = coxph_scaling_ref["support"]
    t_grid = np.linspace(a + 0.1 * (b - a), a + 0.9 * (b - a), 3)
    x_d_new = np.zeros((3, coxph_scaling_ref["q_d"]))
    with pytest.raises(ValueError, match="X_scale_new must be"):
        fitted_coxph.survival(t_grid, X=x_d_new)


def test_coxph_scaling_with_interacting_rejected(
    coxph_scaling_ref: dict,
) -> None:
    """``Coxph(scaling=..., interacting=...)`` is rejected (ADR 0002 Dec. 2)."""
    p = coxph_scaling_ref["p"]
    interacting = BernsteinBasis(order=2, support=(-3.0, 3.0))
    with pytest.raises(ValueError, match="scaling=.*interacting="):
        Coxph(
            support=coxph_scaling_ref["support"],
            order=p - 1,
            scaling=coxph_scaling_ref["x_s"],
            interacting=interacting,
        )
