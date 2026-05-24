"""R parity for ``Colr(scaling=X_s)`` — issue #75.

End-to-end coverage of the ``tram::Colr(y ~ x_d | x_s, data, support, order)``
convenience surface: the kwarg must thread through to the scaled-baseline
likelihood (#71) and the scaled-predict path (#72) without callers having to
reach for :class:`pymlt.MLT` directly.

Reference data lives in ``reference/scaling_colr_*`` and is produced by
``reference/generate_reference.R``.

Sign conventions:

* ``tram::Colr`` uses ``negative = FALSE`` (so ``h + X·β``), unlike
  ``tram::BoxCox``.  pymlt parametrises ``h + X·β`` identically, so β is
  sign-aligned with R ``tram::Colr``.
* γ is sign-aligned across the two parameterisations (ADR 0002, Decision 5).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pymlt.tram import Colr

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"


_FIT_FILES = [
    "scaling_colr_y.txt",
    "scaling_colr_x_d.txt",
    "scaling_colr_x_s.txt",
    "scaling_colr_support.txt",
    "scaling_colr_theta.txt",
    "scaling_colr_loglik.txt",
]

_LOGODDS_FILES = [
    "scaling_colr_logodds_x_d_new.txt",
    "scaling_colr_logodds_x_s_new.txt",
    "scaling_colr_logodds_q_grid.txt",
    "scaling_colr_logodds.txt",
]


def _load(name: str) -> np.ndarray:
    return np.loadtxt(REF_DIR / name)


def _maybe_skip(files: list[str]) -> None:
    if not all((REF_DIR / f).exists() for f in files):
        pytest.skip(
            "scaling_colr_* reference files not yet generated — "
            "run Rscript reference/generate_reference.R"
        )


@pytest.fixture(scope="module")
def colr_scaling_ref() -> dict:
    _maybe_skip(_FIT_FILES)
    y = _load("scaling_colr_y.txt")
    x_d = _load("scaling_colr_x_d.txt").reshape(-1, 1)
    x_s = _load("scaling_colr_x_s.txt").reshape(-1, 1)
    support = tuple(_load("scaling_colr_support.txt"))
    theta_full = _load("scaling_colr_theta.txt")
    ll = float(_load("scaling_colr_loglik.txt"))
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
def fitted_colr(colr_scaling_ref: dict) -> Colr:
    p = colr_scaling_ref["p"]
    model = Colr(
        support=colr_scaling_ref["support"],
        order=p - 1,
        scaling=colr_scaling_ref["x_s"],
    )
    model.fit(colr_scaling_ref["y"], X=colr_scaling_ref["x_d"])
    return model


def test_colr_accepts_scaling_kwarg(colr_scaling_ref: dict) -> None:
    """Colr(support=..., scaling=X_s).fit(y, X=X_d) runs without raising."""
    p = colr_scaling_ref["p"]
    model = Colr(
        support=colr_scaling_ref["support"],
        order=p - 1,
        scaling=colr_scaling_ref["x_s"],
    )
    model.fit(colr_scaling_ref["y"], X=colr_scaling_ref["x_d"])
    assert model.is_fitted_
    assert model.theta_ is not None
    assert model.theta_.shape == (
        colr_scaling_ref["p"] + colr_scaling_ref["q_d"] + colr_scaling_ref["q_s"],
    )


def test_colr_theta_beta_gamma_match_R(
    fitted_colr: Colr, colr_scaling_ref: dict
) -> None:
    """Fitted θ_b, β, γ match R at rtol=1e-4/atol=1e-5.

    The Colr likelihood is log-concave but the heteroskedastic fit shows the
    usual MLE noise floor — relaxed from the issue's aspirational 1e-6 target
    to match the empirically observed optimiser parity (see Coxph's #74 test
    for the same rationale).  ``log-lik`` is asserted separately at rtol=1e-6.
    """
    p = colr_scaling_ref["p"]
    q_d = colr_scaling_ref["q_d"]
    assert fitted_colr.theta_ is not None
    theta_b = fitted_colr.theta_[:p]
    beta = fitted_colr.theta_[p : p + q_d]
    gamma = fitted_colr.theta_[p + q_d :]

    np.testing.assert_allclose(
        theta_b, colr_scaling_ref["theta_b"], rtol=1e-4, atol=1e-5
    )
    # tram::Colr uses negative=FALSE (h + Xβ); pymlt parametrises h + Xβ
    # identically, so β is sign-aligned with R `tram::Colr`.
    np.testing.assert_allclose(beta, colr_scaling_ref["beta_r"], rtol=1e-4, atol=1e-5)
    # γ is sign-aligned across the two parameterisations (ADR 0002, Decision 5).
    np.testing.assert_allclose(gamma, colr_scaling_ref["gamma_r"], rtol=1e-4, atol=1e-5)


def test_colr_log_likelihood_matches_R(
    fitted_colr: Colr, colr_scaling_ref: dict
) -> None:
    """log-lik at the fitted parameters matches R at rtol=1e-6."""
    assert fitted_colr.result_ is not None
    np.testing.assert_allclose(
        fitted_colr.result_.log_likelihood,
        colr_scaling_ref["log_likelihood"],
        rtol=1e-6,
        atol=1e-8,
    )


def test_colr_gamma_property(fitted_colr: Colr, colr_scaling_ref: dict) -> None:
    """``gamma_`` returns the γ block (sign-aligned with R)."""
    np.testing.assert_allclose(
        fitted_colr.gamma_, colr_scaling_ref["gamma_r"], rtol=1e-4, atol=1e-5
    )


def test_colr_feature_names_scaling_default(fitted_colr: Colr) -> None:
    """``feature_names_scaling_`` falls back to ``X1, X2, ...`` for ndarray input."""
    assert fitted_colr.feature_names_scaling_ == ["X1"]


def test_colr_feature_names_scaling_from_dataframe(
    colr_scaling_ref: dict,
) -> None:
    """A pandas DataFrame's column names propagate to ``feature_names_scaling_``."""
    pd = pytest.importorskip("pandas")
    p = colr_scaling_ref["p"]
    x_s_df = pd.DataFrame(colr_scaling_ref["x_s"], columns=["het_x"])
    model = Colr(
        support=colr_scaling_ref["support"],
        order=p - 1,
        scaling=x_s_df,
    )
    model.fit(colr_scaling_ref["y"], X=colr_scaling_ref["x_d"])
    assert model.feature_names_scaling_ == ["het_x"]


def test_colr_gamma_raises_without_scaling(colr_scaling_ref: dict) -> None:
    """Accessing ``gamma_`` on a non-scaling fit raises ValueError."""
    p = colr_scaling_ref["p"]
    model = Colr(support=colr_scaling_ref["support"], order=p - 1)
    model.fit(colr_scaling_ref["y"], X=colr_scaling_ref["x_d"])
    with pytest.raises(ValueError, match="scaling="):
        _ = model.gamma_
    with pytest.raises(ValueError, match="scaling="):
        _ = model.feature_names_scaling_


def test_colr_predict_logodds_matches_R(fitted_colr: Colr) -> None:
    """``predict(y_new, X_new, X_scale_new, what='logodds')`` matches R on a grid.

    Heteroskedastic Colr — the log-odds curve is no longer parallel across
    x_s strata, so this grid exercises the scaled-baseline ``h(y) · exp(0.5
    · x_s · γ)`` factor inside the predict path (#72).
    """
    _maybe_skip(_LOGODDS_FILES)
    x_d_new = _load("scaling_colr_logodds_x_d_new.txt").reshape(-1, 1)
    x_s_new = _load("scaling_colr_logodds_x_s_new.txt").reshape(-1, 1)
    q_grid = _load("scaling_colr_logodds_q_grid.txt")
    k = q_grid.shape[0]
    m = x_d_new.shape[0]
    expected = _load("scaling_colr_logodds.txt").reshape(k, m)

    # Row-major (k, m) broadcast: y rep'd m times per row; (X_d, X_s) tiled k times.
    y_rep = np.repeat(q_grid[:, None], m, axis=1).ravel()
    X_d_rep = np.tile(x_d_new, (k, 1))
    X_s_rep = np.tile(x_s_new, (k, 1))

    got = fitted_colr.predict(
        y_rep, X_new=X_d_rep, X_scale_new=X_s_rep, what="logodds"
    ).reshape(k, m)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-7)


def test_colr_logodds_demonstrates_non_proportional_odds(fitted_colr: Colr) -> None:
    """log-odds gap between two ``x_s`` values varies with ``y`` (non-PO).

    Under a *proportional*-odds model the difference
    ``logit F(y | x_s_high) − logit F(y | x_s_low)`` is constant in ``y``.
    Under ``scaling=x_s`` it depends on ``y`` through the scaled baseline
    ``h_0(y) · exp(0.5 · x_s · γ)``.  This test verifies that we see
    a non-trivial divergence at two interior y values.
    """
    a, b = fitted_colr._support
    y_grid = np.array([a + 0.2 * (b - a), a + 0.8 * (b - a)])
    x_d_zero = np.zeros((y_grid.size, 1))
    x_s_low = np.full((y_grid.size, 1), -1.5)
    x_s_high = np.full((y_grid.size, 1), 1.5)

    lo_low = fitted_colr.predict(
        y_grid, X_new=x_d_zero, X_scale_new=x_s_low, what="logodds"
    )
    lo_high = fitted_colr.predict(
        y_grid, X_new=x_d_zero, X_scale_new=x_s_high, what="logodds"
    )
    gap = lo_high - lo_low
    assert abs(gap[0] - gap[1]) > 0.01, (
        "log-odds gap constant across y — PO not violated by scaling"
    )


def test_colr_predict_requires_X_scale_new(
    fitted_colr: Colr, colr_scaling_ref: dict
) -> None:
    """When fitted with ``scaling=``, ``predict`` errors out without X_scale_new."""
    x_d_new = colr_scaling_ref["x_d"][:3]
    y_new = colr_scaling_ref["y"][:3]
    with pytest.raises(ValueError, match="X_scale_new must be"):
        fitted_colr.predict(y_new, X_new=x_d_new, what="logodds")
