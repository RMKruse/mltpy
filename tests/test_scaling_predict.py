"""R parity for predict() with scaling — issue #72.

End-to-end checks for ``ConditionalTransformationModel.predict`` when the model
was fit with ``scaling=`` (and therefore carries a ``γ`` block).  The base
fit re-used here is the BoxCox + ``scale = ~ x_s`` reference from #70
(normal base, exact observations).  All R-parity values are produced by
``reference/generate_reference.R``; see the ``Scaling-terms predict path
(issue #72)`` block for the fixture layout.

The full 13-value ``what`` surface is exercised here.  ``"trafo"`` and the
``log*`` mirrors are checked for *self-consistency* against the
``"distribution"`` / ``"survivor"`` / ``"density"`` paths (their log/non-log
relationship is purely algebraic and does not need an R reference).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from scipy.stats import kstest

from mltpy import MLT

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"

_FIT_FILES = [
    "scaling_boxcox_normal_y.txt",
    "scaling_boxcox_normal_x_d.txt",
    "scaling_boxcox_normal_x_s.txt",
    "scaling_boxcox_normal_support.txt",
    "scaling_boxcox_normal_theta.txt",
]

_PREDICT_FILES = [
    "scaling_predict_x_d_new.txt",
    "scaling_predict_x_s_new.txt",
    "scaling_predict_q_grid.txt",
    "scaling_predict_prob_grid.txt",
    "scaling_predict_distribution.txt",
    "scaling_predict_density.txt",
    "scaling_predict_hazard.txt",
    "scaling_predict_survivor.txt",
    "scaling_predict_quantile.txt",
]


def _load(name: str) -> np.ndarray:
    return np.loadtxt(REF_DIR / name)


def _maybe_skip(files: list[str]) -> None:
    if not all((REF_DIR / f).exists() for f in files):
        pytest.skip(
            "scaling_predict_* reference files not yet generated — "
            "run Rscript reference/generate_reference.R"
        )


@pytest.fixture(scope="module")
def fitted_scaling_model() -> tuple[MLT, dict]:
    """Refit the BoxCox + scale fixture and return (model, fixture-data)."""
    _maybe_skip(_FIT_FILES + _PREDICT_FILES)
    y = _load("scaling_boxcox_normal_y.txt")
    x_d = _load("scaling_boxcox_normal_x_d.txt").reshape(-1, 1)
    x_s = _load("scaling_boxcox_normal_x_s.txt").reshape(-1, 1)
    support = tuple(_load("scaling_boxcox_normal_support.txt"))
    theta = _load("scaling_boxcox_normal_theta.txt")
    p = len(theta) - x_d.shape[1] - x_s.shape[1]

    model = MLT(
        order=p - 1,
        support=(float(support[0]), float(support[1])),
        scaling=x_s,
    )
    model.fit(y, X=x_d)

    x_d_new = _load("scaling_predict_x_d_new.txt").reshape(-1, 1)
    x_s_new = _load("scaling_predict_x_s_new.txt").reshape(-1, 1)
    q_grid = _load("scaling_predict_q_grid.txt")
    prob_grid = _load("scaling_predict_prob_grid.txt")
    m = x_d_new.shape[0]
    k = q_grid.shape[0]
    k_q = prob_grid.shape[0]
    return model, {
        "x_d_new": x_d_new,
        "x_s_new": x_s_new,
        "q_grid": q_grid,
        "prob_grid": prob_grid,
        "m": m,
        "k": k,
        "k_q": k_q,
        "p": p,
    }


def _broadcast_grid(
    q_grid: np.ndarray, x_d_new: np.ndarray, x_s_new: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cartesian product helper: returns (y_rep, X_d_rep, X_s_rep) shape (k*m, *)."""
    k = q_grid.shape[0]
    m = x_d_new.shape[0]
    y_rep = np.repeat(q_grid[:, None], m, axis=1).ravel()  # row-major (k, m)
    X_d_rep = np.tile(x_d_new, (k, 1))
    X_s_rep = np.tile(x_s_new, (k, 1))
    return y_rep, X_d_rep, X_s_rep


@pytest.mark.parametrize("what", ["distribution", "survivor", "density", "hazard"])
def test_predict_what_matches_R(
    fitted_scaling_model: tuple[MLT, dict], what: str
) -> None:
    """``predict(y_new, X_new, X_scale_new, what=...)`` matches R fixture."""
    model, ref = fitted_scaling_model
    expected = _load(f"scaling_predict_{what}.txt").reshape(ref["k"], ref["m"])
    y_rep, X_d_rep, X_s_rep = _broadcast_grid(
        ref["q_grid"], ref["x_d_new"], ref["x_s_new"]
    )
    got = model.predict(y_rep, X_new=X_d_rep, X_scale_new=X_s_rep, what=what).reshape(
        ref["k"], ref["m"]
    )
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-7)


def test_predict_quantile_matches_R(fitted_scaling_model: tuple[MLT, dict]) -> None:
    """``predict(..., what='quantile')`` matches R fixture.

    Quantile inversion is the downstream of the fitted θ; per
    ``test_scaling_basic``, ``model.theta_`` matches R only at ``rtol=1e-5``,
    so quantile precision is bounded by that — use the same ``rtol=1e-3``
    band that the interaction-basis quantile parity test (#67) settled on.
    """
    model, ref = fitted_scaling_model
    expected = _load("scaling_predict_quantile.txt").reshape(ref["k_q"], ref["m"])
    p_rep, X_d_rep, X_s_rep = _broadcast_grid(
        ref["prob_grid"], ref["x_d_new"], ref["x_s_new"]
    )
    got = model.predict(
        p_rep, X_new=X_d_rep, X_scale_new=X_s_rep, what="quantile"
    ).reshape(ref["k_q"], ref["m"])
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Self-consistency for the `log*` / `trafo` / `odds` / `cumhazard` variants —
# these are algebraic mirrors of distribution / survivor / density and do not
# need a separate R fixture.  Spot-checks across all 13 `what` values.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def predict_full_set(fitted_scaling_model: tuple[MLT, dict]) -> dict:
    """Compute every ``what`` value once on the same broadcast grid."""
    model, ref = fitted_scaling_model
    y_rep, X_d_rep, X_s_rep = _broadcast_grid(
        ref["q_grid"], ref["x_d_new"], ref["x_s_new"]
    )
    out = {}
    for what in (
        "trafo",
        "distribution",
        "logdistribution",
        "survivor",
        "logsurvivor",
        "density",
        "logdensity",
        "hazard",
        "loghazard",
        "cumhazard",
        "logcumhazard",
        "odds",
        "logodds",
    ):
        out[what] = model.predict(y_rep, X_new=X_d_rep, X_scale_new=X_s_rep, what=what)
    return out


def test_predict_log_distribution_consistent(predict_full_set: dict) -> None:
    np.testing.assert_allclose(
        predict_full_set["logdistribution"],
        np.log(predict_full_set["distribution"]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_predict_log_survivor_consistent(predict_full_set: dict) -> None:
    np.testing.assert_allclose(
        predict_full_set["logsurvivor"],
        np.log(predict_full_set["survivor"]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_predict_log_density_consistent(predict_full_set: dict) -> None:
    np.testing.assert_allclose(
        predict_full_set["logdensity"],
        np.log(predict_full_set["density"]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_predict_log_hazard_consistent(predict_full_set: dict) -> None:
    np.testing.assert_allclose(
        predict_full_set["loghazard"],
        np.log(predict_full_set["hazard"]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_predict_cumhazard_consistent(predict_full_set: dict) -> None:
    np.testing.assert_allclose(
        predict_full_set["cumhazard"],
        -np.log(predict_full_set["survivor"]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_predict_log_cumhazard_consistent(predict_full_set: dict) -> None:
    np.testing.assert_allclose(
        predict_full_set["logcumhazard"],
        np.log(predict_full_set["cumhazard"]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_predict_odds_consistent(predict_full_set: dict) -> None:
    np.testing.assert_allclose(
        predict_full_set["odds"],
        predict_full_set["distribution"] / predict_full_set["survivor"],
        rtol=1e-10,
        atol=1e-12,
    )


def test_predict_log_odds_consistent(predict_full_set: dict) -> None:
    np.testing.assert_allclose(
        predict_full_set["logodds"],
        np.log(predict_full_set["odds"]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_predict_trafo_is_h(
    fitted_scaling_model: tuple[MLT, dict], predict_full_set: dict
) -> None:
    """``what='trafo'`` returns ``h(y|x) = h_0(y)·exp(0.5 x_s γ) + x_d β``."""
    model, ref = fitted_scaling_model
    y_rep, X_d_rep, X_s_rep = _broadcast_grid(
        ref["q_grid"], ref["x_d_new"], ref["x_s_new"]
    )
    p = ref["p"]
    theta_b = model.theta_[:p]
    q_d = X_d_rep.shape[1]
    beta = model.theta_[p : p + q_d]
    gamma = model.theta_[p + q_d :]
    B = model.basis.evaluate(y_rep)
    h0 = B @ theta_b
    f = np.exp(0.5 * (X_s_rep @ gamma))
    expected = h0 * f + X_d_rep @ beta
    np.testing.assert_allclose(predict_full_set["trafo"], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# X_scale_new validation
# ---------------------------------------------------------------------------


def test_predict_missing_X_scale_new_raises(
    fitted_scaling_model: tuple[MLT, dict],
) -> None:
    """Scaling model requires X_scale_new at predict time."""
    model, ref = fitted_scaling_model
    y_rep, X_d_rep, _ = _broadcast_grid(ref["q_grid"], ref["x_d_new"], ref["x_s_new"])
    with pytest.raises(ValueError, match="X_scale_new"):
        model.predict(y_rep, X_new=X_d_rep, what="distribution")


def test_predict_wrong_X_scale_shape_raises(
    fitted_scaling_model: tuple[MLT, dict],
) -> None:
    """X_scale_new column count must equal q_s."""
    model, ref = fitted_scaling_model
    y_rep, X_d_rep, X_s_rep = _broadcast_grid(
        ref["q_grid"], ref["x_d_new"], ref["x_s_new"]
    )
    X_s_bad = np.hstack([X_s_rep, X_s_rep])  # doubled columns
    with pytest.raises(ValueError, match="X_scale_new"):
        model.predict(y_rep, X_new=X_d_rep, X_scale_new=X_s_bad, what="distribution")


def test_predict_X_scale_row_mismatch_raises(
    fitted_scaling_model: tuple[MLT, dict],
) -> None:
    """X_scale_new row count must match y_new length."""
    model, ref = fitted_scaling_model
    y_rep, X_d_rep, X_s_rep = _broadcast_grid(
        ref["q_grid"], ref["x_d_new"], ref["x_s_new"]
    )
    with pytest.raises(ValueError, match="X_scale_new"):
        model.predict(
            y_rep, X_new=X_d_rep, X_scale_new=X_s_rep[:2], what="distribution"
        )


# ---------------------------------------------------------------------------
# simulate() self-consistency under scaling
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings(r"ignore:predict\(what='quantile'\)")
def test_simulate_under_scaling_matches_distribution(
    fitted_scaling_model: tuple[MLT, dict],
) -> None:
    """Empirical CDF of large simulate() output matches predict(what='distribution').

    Holds one (x_d, x_s) profile fixed across all draws, then runs a KS test
    against the model's analytical CDF.
    """
    model, ref = fitted_scaling_model
    n_sim = 5000
    x_d_fixed = float(ref["x_d_new"][0, 0])
    x_s_fixed = float(ref["x_s_new"][0, 0])
    X_d = np.full((n_sim, 1), x_d_fixed)
    X_s = np.full((n_sim, 1), x_s_fixed)
    draws = model.simulate(
        n_sim, X=X_d, X_scale=X_s, random_state=np.random.default_rng(72)
    )

    def cdf(y: np.ndarray) -> np.ndarray:
        return model.predict(
            np.asarray(y, dtype=float),
            X_new=np.full((y.size, 1), x_d_fixed),
            X_scale_new=np.full((y.size, 1), x_s_fixed),
            what="distribution",
        )

    stat, _ = kstest(draws, cdf)
    # 5000-sample KS critical value @ alpha = 1e-4 is ≈ 0.028; use a generous
    # ceiling so the test fails only on a genuine implementation bug, not on
    # the long-tailed empirical fluctuations of a single draw.
    assert stat < 0.04, f"KS statistic {stat:.4f} exceeds 0.04"
