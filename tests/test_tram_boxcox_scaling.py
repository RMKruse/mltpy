"""R parity for ``BoxCox(scaling=X_s)`` — issue #73.

End-to-end coverage of the ``tram::BoxCox(y ~ x_d, data, scale=~x_s)``
convenience surface: the kwarg must thread through to the scaled-baseline
likelihood (#71) and the scaled-predict path (#72) without callers having to
reach for :class:`pymlt.MLT` directly.

Reference data is reused from the #70 fit fixture
(``scaling_boxcox_normal_*``) and the #72 predict fixtures
(``scaling_predict_{distribution,density}.txt``), all produced by
``reference/generate_reference.R``.

Sign conventions:

* ``tram::BoxCox`` uses ``negative = TRUE`` (so ``h − X·β``).  pymlt
  parametrises ``h + X·β``; the β block compares against ``-β_R``.
* γ is sign-aligned across the two parameterisations
  (ADR 0002, Decision 5).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pymlt.tram import BoxCox

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"

_FIT_FILES = [
    "scaling_boxcox_normal_y.txt",
    "scaling_boxcox_normal_x_d.txt",
    "scaling_boxcox_normal_x_s.txt",
    "scaling_boxcox_normal_support.txt",
    "scaling_boxcox_normal_theta.txt",
    "scaling_boxcox_normal_loglik.txt",
]

_PREDICT_FILES = [
    "scaling_predict_x_d_new.txt",
    "scaling_predict_x_s_new.txt",
    "scaling_predict_q_grid.txt",
    "scaling_predict_distribution.txt",
    "scaling_predict_density.txt",
]


def _load(name: str) -> np.ndarray:
    return np.loadtxt(REF_DIR / name)


def _maybe_skip(files: list[str]) -> None:
    if not all((REF_DIR / f).exists() for f in files):
        pytest.skip(
            "scaling reference files not yet generated — "
            "run Rscript reference/generate_reference.R"
        )


@pytest.fixture(scope="module")
def boxcox_scaling_ref() -> dict:
    _maybe_skip(_FIT_FILES)
    y = _load("scaling_boxcox_normal_y.txt")
    x_d = _load("scaling_boxcox_normal_x_d.txt").reshape(-1, 1)
    x_s = _load("scaling_boxcox_normal_x_s.txt").reshape(-1, 1)
    support = tuple(_load("scaling_boxcox_normal_support.txt"))
    theta_full = _load("scaling_boxcox_normal_theta.txt")
    ll = float(_load("scaling_boxcox_normal_loglik.txt"))
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
def fitted_boxcox(boxcox_scaling_ref: dict) -> BoxCox:
    p = boxcox_scaling_ref["p"]
    model = BoxCox(
        support=boxcox_scaling_ref["support"],
        order=p - 1,
        scaling=boxcox_scaling_ref["x_s"],
    )
    model.fit(boxcox_scaling_ref["y"], X=boxcox_scaling_ref["x_d"])
    return model


def test_boxcox_accepts_scaling_kwarg(boxcox_scaling_ref: dict) -> None:
    """BoxCox(support=..., scaling=X_s).fit(y, X=X_d) runs without raising."""
    p = boxcox_scaling_ref["p"]
    model = BoxCox(
        support=boxcox_scaling_ref["support"],
        order=p - 1,
        scaling=boxcox_scaling_ref["x_s"],
    )
    model.fit(boxcox_scaling_ref["y"], X=boxcox_scaling_ref["x_d"])
    assert model.is_fitted_
    assert model.theta_ is not None
    assert model.theta_.shape == (
        boxcox_scaling_ref["p"] + boxcox_scaling_ref["q_d"] + boxcox_scaling_ref["q_s"],
    )


def test_boxcox_theta_beta_gamma_match_R(
    fitted_boxcox: BoxCox, boxcox_scaling_ref: dict
) -> None:
    """Fitted θ_b, β (sign-flipped), γ match R at rtol=1e-5/atol=1e-7."""
    p = boxcox_scaling_ref["p"]
    q_d = boxcox_scaling_ref["q_d"]
    assert fitted_boxcox.theta_ is not None
    theta_b = fitted_boxcox.theta_[:p]
    beta = fitted_boxcox.theta_[p : p + q_d]
    gamma = fitted_boxcox.theta_[p + q_d :]

    np.testing.assert_allclose(
        theta_b, boxcox_scaling_ref["theta_b"], rtol=1e-5, atol=1e-7
    )
    # pymlt parametrises h + Xβ; tram::BoxCox uses h - Xβ_R.  Sign flip on β.
    np.testing.assert_allclose(
        beta, -boxcox_scaling_ref["beta_r"], rtol=1e-5, atol=1e-7
    )
    # γ is sign-aligned across the two parameterisations (ADR 0002, Decision 5).
    np.testing.assert_allclose(
        gamma, boxcox_scaling_ref["gamma_r"], rtol=1e-5, atol=1e-7
    )


def test_boxcox_log_likelihood_matches_R(
    fitted_boxcox: BoxCox, boxcox_scaling_ref: dict
) -> None:
    """log-lik at the fitted parameters matches R at rtol=1e-6."""
    assert fitted_boxcox.result_ is not None
    np.testing.assert_allclose(
        fitted_boxcox.result_.log_likelihood,
        boxcox_scaling_ref["log_likelihood"],
        rtol=1e-6,
        atol=1e-8,
    )


def test_boxcox_gamma_property(fitted_boxcox: BoxCox, boxcox_scaling_ref: dict) -> None:
    """``gamma_`` returns the γ block (sign-aligned with R)."""
    np.testing.assert_allclose(
        fitted_boxcox.gamma_, boxcox_scaling_ref["gamma_r"], rtol=1e-5, atol=1e-7
    )


def test_boxcox_feature_names_scaling_default(fitted_boxcox: BoxCox) -> None:
    """``feature_names_scaling_`` falls back to ``X1, X2, ...`` for ndarray input."""
    assert fitted_boxcox.feature_names_scaling_ == ["X1"]


def test_boxcox_feature_names_scaling_from_dataframe(
    boxcox_scaling_ref: dict,
) -> None:
    """A pandas DataFrame's column names propagate to ``feature_names_scaling_``."""
    pd = pytest.importorskip("pandas")
    p = boxcox_scaling_ref["p"]
    x_s_df = pd.DataFrame(boxcox_scaling_ref["x_s"], columns=["het_x"])
    model = BoxCox(
        support=boxcox_scaling_ref["support"],
        order=p - 1,
        scaling=x_s_df,
    )
    model.fit(boxcox_scaling_ref["y"], X=boxcox_scaling_ref["x_d"])
    assert model.feature_names_scaling_ == ["het_x"]


def test_boxcox_gamma_raises_without_scaling(boxcox_scaling_ref: dict) -> None:
    """Accessing ``gamma_`` on a non-scaling fit raises ValueError."""
    p = boxcox_scaling_ref["p"]
    model = BoxCox(support=boxcox_scaling_ref["support"], order=p - 1)
    model.fit(boxcox_scaling_ref["y"], X=boxcox_scaling_ref["x_d"])
    with pytest.raises(ValueError, match="scaling="):
        _ = model.gamma_
    with pytest.raises(ValueError, match="scaling="):
        _ = model.feature_names_scaling_


@pytest.mark.parametrize("what", ["distribution", "density"])
def test_boxcox_predict_matches_R(
    fitted_boxcox: BoxCox, boxcox_scaling_ref: dict, what: str
) -> None:
    """``predict(y_new, X_new, X_scale_new, what=...)`` matches R on a grid."""
    _maybe_skip(_PREDICT_FILES)
    x_d_new = _load("scaling_predict_x_d_new.txt").reshape(-1, 1)
    x_s_new = _load("scaling_predict_x_s_new.txt").reshape(-1, 1)
    q_grid = _load("scaling_predict_q_grid.txt")
    k = q_grid.shape[0]
    m = x_d_new.shape[0]
    expected = _load(f"scaling_predict_{what}.txt").reshape(k, m)

    # Row-major (k, m) broadcast: y rep'd m times per row; (X_d, X_s) tiled k times.
    y_rep = np.repeat(q_grid[:, None], m, axis=1).ravel()
    X_d_rep = np.tile(x_d_new, (k, 1))
    X_s_rep = np.tile(x_s_new, (k, 1))

    got = fitted_boxcox.predict(
        y_rep, X_new=X_d_rep, X_scale_new=X_s_rep, what=what
    ).reshape(k, m)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-7)


def test_boxcox_predict_requires_X_scale_new(
    fitted_boxcox: BoxCox,
    boxcox_scaling_ref: dict,
) -> None:
    """When fitted with ``scaling=``, ``predict`` errors out without X_scale_new."""
    x_d_new = boxcox_scaling_ref["x_d"][:3]
    y_new = boxcox_scaling_ref["y"][:3]
    with pytest.raises(ValueError, match="X_scale_new must be"):
        fitted_boxcox.predict(y_new, X_new=x_d_new, what="distribution")
