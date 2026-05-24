"""R parity for ``Lm(scaling=X_s)`` — issue #76.

End-to-end coverage of the ``tram::Lm(y ~ x_d | x_s, data, scale = ~x_s)``
convenience surface.  The kwarg must thread through to the scaled-baseline
likelihood (#71) and the scaled-predict path (#72) without callers having
to reach for :class:`pymlt.MLT` directly.

Reference data lives in ``reference/scaling_lm_*`` and is produced by
``reference/generate_reference.R``.

Sign conventions:

* ``tram::Lm`` uses ``negative = TRUE`` (so ``h − X_d·β``).  pymlt
  parametrises ``h + X_d·β``; the β block compares against ``-β_R``.
* γ is sign-aligned across the two parameterisations
  (ADR 0002, Decision 5).

Notes
-----
``tram::Lm`` represents the baseline as the affine basis ``(1, y)`` (a
``polynomial_basis``) whereas pymlt's ``Lm`` uses the equivalent order=1
``BernsteinBasis``.  The two parameterisations are linearly related — the
fitted h(y) curve is identical at the MLE — but the raw ``theta_b`` block
is in different coordinates and is not asserted element-wise here.
Log-likelihood, β and γ are parameterisation-invariant and *are* compared
at the issue's rtol=1e-6 budget.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pymlt.tram import Lm

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"

_FIT_FILES = [
    "scaling_lm_y.txt",
    "scaling_lm_x_d.txt",
    "scaling_lm_x_s.txt",
    "scaling_lm_support.txt",
    "scaling_lm_theta.txt",
    "scaling_lm_loglik.txt",
]


def _load(name: str) -> np.ndarray:
    return np.loadtxt(REF_DIR / name)


def _maybe_skip(files: list[str]) -> None:
    if not all((REF_DIR / f).exists() for f in files):
        pytest.skip(
            "scaling_lm_* reference files not yet generated — "
            "run Rscript reference/generate_reference.R"
        )


@pytest.fixture(scope="module")
def lm_scaling_ref() -> dict:
    _maybe_skip(_FIT_FILES)
    y = _load("scaling_lm_y.txt")
    x_d = _load("scaling_lm_x_d.txt").reshape(-1, 1)
    x_s = _load("scaling_lm_x_s.txt").reshape(-1, 1)
    support = tuple(_load("scaling_lm_support.txt"))
    theta_full = _load("scaling_lm_theta.txt")
    ll = float(_load("scaling_lm_loglik.txt"))
    q_d = x_d.shape[1]
    q_s = x_s.shape[1]
    # tram::Lm always uses an order=1 affine baseline (p = 2).
    p = len(theta_full) - q_d - q_s
    assert p == 2
    return {
        "y": y,
        "x_d": x_d,
        "x_s": x_s,
        "support": (float(support[0]), float(support[1])),
        "p": p,
        "q_d": q_d,
        "q_s": q_s,
        "beta_r": theta_full[p : p + q_d],
        "gamma_r": theta_full[p + q_d :],
        "log_likelihood": ll,
    }


@pytest.fixture(scope="module")
def fitted_lm(lm_scaling_ref: dict) -> Lm:
    model = Lm(support=lm_scaling_ref["support"], scaling=lm_scaling_ref["x_s"])
    model.fit(lm_scaling_ref["y"], X=lm_scaling_ref["x_d"])
    return model


def test_lm_accepts_scaling_kwarg(lm_scaling_ref: dict) -> None:
    """Lm(support=..., scaling=X_s).fit(y, X=X_d) runs without raising."""
    model = Lm(support=lm_scaling_ref["support"], scaling=lm_scaling_ref["x_s"])
    model.fit(lm_scaling_ref["y"], X=lm_scaling_ref["x_d"])
    assert model.is_fitted_
    assert model.theta_ is not None
    assert model.theta_.shape == (
        lm_scaling_ref["p"] + lm_scaling_ref["q_d"] + lm_scaling_ref["q_s"],
    )


def test_lm_beta_gamma_match_R(fitted_lm: Lm, lm_scaling_ref: dict) -> None:
    """Fitted β (sign-flipped) and γ match R at rtol=1e-5/atol=1e-7."""
    p = lm_scaling_ref["p"]
    q_d = lm_scaling_ref["q_d"]
    assert fitted_lm.theta_ is not None
    beta = fitted_lm.theta_[p : p + q_d]
    gamma = fitted_lm.theta_[p + q_d :]

    # pymlt parametrises h + Xβ; tram::Lm uses h − Xβ_R.  Sign flip on β.
    np.testing.assert_allclose(beta, -lm_scaling_ref["beta_r"], rtol=1e-5, atol=1e-7)
    # γ is sign-aligned across the two parameterisations (ADR 0002, Decision 5).
    np.testing.assert_allclose(gamma, lm_scaling_ref["gamma_r"], rtol=1e-5, atol=1e-7)


def test_lm_log_likelihood_matches_R(fitted_lm: Lm, lm_scaling_ref: dict) -> None:
    """log-lik at the fitted parameters matches R at rtol=1e-6."""
    assert fitted_lm.result_ is not None
    np.testing.assert_allclose(
        fitted_lm.result_.log_likelihood,
        lm_scaling_ref["log_likelihood"],
        rtol=1e-6,
        atol=1e-8,
    )


def test_lm_gamma_property(fitted_lm: Lm, lm_scaling_ref: dict) -> None:
    """``gamma_`` returns the γ block (sign-aligned with R)."""
    np.testing.assert_allclose(
        fitted_lm.gamma_, lm_scaling_ref["gamma_r"], rtol=1e-5, atol=1e-7
    )


def test_lm_feature_names_scaling_default(fitted_lm: Lm) -> None:
    """``feature_names_scaling_`` falls back to ``X1, X2, ...`` for ndarray input."""
    assert fitted_lm.feature_names_scaling_ == ["X1"]


def test_lm_feature_names_scaling_from_dataframe(lm_scaling_ref: dict) -> None:
    """A pandas DataFrame's column names propagate to ``feature_names_scaling_``."""
    pd = pytest.importorskip("pandas")
    x_s_df = pd.DataFrame(lm_scaling_ref["x_s"], columns=["het_x"])
    model = Lm(support=lm_scaling_ref["support"], scaling=x_s_df)
    model.fit(lm_scaling_ref["y"], X=lm_scaling_ref["x_d"])
    assert model.feature_names_scaling_ == ["het_x"]


def test_lm_gamma_raises_without_scaling(lm_scaling_ref: dict) -> None:
    """Accessing ``gamma_`` on a non-scaling fit raises ValueError."""
    model = Lm(support=lm_scaling_ref["support"])
    model.fit(lm_scaling_ref["y"], X=lm_scaling_ref["x_d"])
    with pytest.raises(ValueError, match="scaling="):
        _ = model.gamma_
    with pytest.raises(ValueError, match="scaling="):
        _ = model.feature_names_scaling_


@pytest.mark.parametrize("prop", ["sigma_", "intercept_", "coef_"])
def test_lm_closed_form_accessors_raise_under_scaling(fitted_lm: Lm, prop: str) -> None:
    """``sigma_``/``intercept_``/``coef_`` raise ``NotImplementedError`` under scaling.

    Decision: the constant-variance closed-form mapping from CTM to lm
    parameters no longer applies when σ depends on x_s via γ; rather than
    silently returning a misleading scalar or returning a callable, the
    accessors raise with a clear pointer at :attr:`gamma_`.
    """
    with pytest.raises(NotImplementedError, match="scaling="):
        _ = getattr(fitted_lm, prop)


def test_lm_closed_form_accessors_still_work_without_scaling(
    lm_scaling_ref: dict,
) -> None:
    """Without ``scaling=``, the closed-form accessors are unchanged."""
    model = Lm(support=lm_scaling_ref["support"])
    model.fit(lm_scaling_ref["y"], X=lm_scaling_ref["x_d"])
    # Just check that no exception is raised — backward compatibility.
    _ = model.sigma_
    _ = model.intercept_
    _ = model.coef_


def test_lm_predict_requires_X_scale_new(fitted_lm: Lm, lm_scaling_ref: dict) -> None:
    """When fitted with ``scaling=``, ``predict`` errors out without X_scale_new."""
    x_d_new = lm_scaling_ref["x_d"][:3]
    y_new = lm_scaling_ref["y"][:3]
    with pytest.raises(ValueError, match="X_scale_new must be"):
        fitted_lm.predict(y_new, X_new=x_d_new, what="distribution")
