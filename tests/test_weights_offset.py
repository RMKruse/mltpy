"""Tests for observation weights and offset in fit() / predict() / score().

Covers:
- Input validation: wrong shape, negative weights, non-finite values.
- Identity: weights=None ≡ weights=ones; offset=None ≡ offset=zeros.
- Scaling: doubling all weights leaves theta unchanged, doubles the LL.
- Replication invariance: fit on 2n with weight=0.5 ≡ fit on n.
- Offset shifts transformation: predict(y, offset_new=c) ≡ trafo + c.
- Stored attributes weights_ and offset_ after fit.
- Score residuals at weighted MLE sum to zero.
- R reference parity (skipped when reference files are absent).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pymlt import MLT, CensoredData
from pymlt.likelihood import (
    _validate_weights_offset,
    log_likelihood,
    negative_log_likelihood,
    score_matrix,
)
from pymlt.tram import BoxCox, Colr, Coxph
from pymlt.variables import CensoringType

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_fit(n: int = 60, seed: int = 0) -> tuple[MLT, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.05, 0.95, n)
    model = MLT(order=4, support=(0.0, 1.0)).fit(y)
    return model, y


def _simple_fit_with_X(
    n: int = 80, seed: int = 1
) -> tuple[MLT, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = 0.5 + 0.3 * x + rng.normal(0, 0.5, n)
    a, b = y.min() - 0.1, y.max() + 0.1
    X = x[:, None]
    model = MLT(order=4, support=(a, b)).fit(y, X)
    return model, y, X


# ---------------------------------------------------------------------------
# _validate_weights_offset
# ---------------------------------------------------------------------------


class TestValidateWeightsOffset:
    def test_none_inputs_pass(self) -> None:
        w, o = _validate_weights_offset(None, None, 10)
        assert w is None and o is None

    def test_valid_weights_coerced(self) -> None:
        w, _ = _validate_weights_offset([1, 2, 3], None, 3)
        assert w is not None
        assert w.dtype == np.float64
        np.testing.assert_array_equal(w, [1.0, 2.0, 3.0])

    def test_valid_offset_coerced(self) -> None:
        _, o = _validate_weights_offset(None, [0.1, -0.2, 0.0], 3)
        assert o is not None
        assert o.dtype == np.float64

    def test_wrong_shape_weights(self) -> None:
        with pytest.raises(ValueError, match="weights must have shape"):
            _validate_weights_offset(np.ones(5), None, 10)

    def test_wrong_shape_offset(self) -> None:
        with pytest.raises(ValueError, match="offset must have shape"):
            _validate_weights_offset(None, np.zeros(3), 10)

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _validate_weights_offset(np.array([1.0, -0.5, 2.0]), None, 3)

    def test_nan_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            _validate_weights_offset(np.array([1.0, np.nan, 2.0]), None, 3)

    def test_inf_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            _validate_weights_offset(np.array([1.0, np.inf, 2.0]), None, 3)

    def test_nan_offset_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            _validate_weights_offset(None, np.array([0.0, np.nan, 0.0]), 3)

    def test_zero_weight_allowed(self) -> None:
        w, _ = _validate_weights_offset(np.array([1.0, 0.0, 2.0]), None, 3)
        assert w is not None
        assert w[1] == 0.0

    def test_all_zero_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one positive"):
            _validate_weights_offset(np.zeros(5), None, 5)


# ---------------------------------------------------------------------------
# fit() validation integration
# ---------------------------------------------------------------------------


class TestFitValidation:
    def test_negative_weight_in_fit(self) -> None:
        y = np.linspace(0.1, 0.9, 10)
        w = np.ones(10)
        w[3] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            MLT(order=3, support=(0.0, 1.0)).fit(y, weights=w)

    def test_wrong_weight_shape_in_fit(self) -> None:
        y = np.linspace(0.1, 0.9, 10)
        with pytest.raises(ValueError, match="weights must have shape"):
            MLT(order=3, support=(0.0, 1.0)).fit(y, weights=np.ones(5))

    def test_wrong_offset_shape_in_fit(self) -> None:
        y = np.linspace(0.1, 0.9, 10)
        with pytest.raises(ValueError, match="offset must have shape"):
            MLT(order=3, support=(0.0, 1.0)).fit(y, offset=np.zeros(5))

    def test_all_zero_weights_in_fit(self) -> None:
        y = np.linspace(0.1, 0.9, 10)
        with pytest.raises(ValueError, match="at least one positive"):
            MLT(order=3, support=(0.0, 1.0)).fit(y, weights=np.zeros(10))


# ---------------------------------------------------------------------------
# Identity tests
# ---------------------------------------------------------------------------


class TestWeightsIdentity:
    """weights=None ≡ weights=1 · 𝟏."""

    def test_no_weights_vs_ones(self) -> None:
        rng = np.random.default_rng(10)
        y = rng.uniform(0.05, 0.95, 50)
        m1 = MLT(order=4, support=(0.0, 1.0)).fit(y)
        m2 = MLT(order=4, support=(0.0, 1.0)).fit(y, weights=np.ones(50))
        np.testing.assert_allclose(m1.theta_, m2.theta_, rtol=1e-5, atol=1e-8)
        assert abs(m1.result_.log_likelihood - m2.result_.log_likelihood) < 1e-6  # type: ignore[union-attr]

    def test_no_weights_vs_ones_with_X(self) -> None:
        rng = np.random.default_rng(11)
        x = rng.normal(0, 1, 60)
        y = 0.5 + 0.4 * x + rng.normal(0, 0.5, 60)
        a, b = y.min() - 0.1, y.max() + 0.1
        X = x[:, None]
        m1 = MLT(order=4, support=(a, b)).fit(y, X)
        m2 = MLT(order=4, support=(a, b)).fit(y, X, weights=np.ones(60))
        np.testing.assert_allclose(m1.theta_, m2.theta_, rtol=1e-5, atol=1e-8)

    def test_weights_stored_after_fit(self) -> None:
        y = np.linspace(0.05, 0.95, 20)
        w = np.ones(20) * 2.0
        m = MLT(order=3, support=(0.0, 1.0)).fit(y, weights=w)
        assert m.weights_ is not None
        np.testing.assert_array_equal(m.weights_, w)

    def test_no_weights_stored_none(self) -> None:
        y = np.linspace(0.05, 0.95, 20)
        m = MLT(order=3, support=(0.0, 1.0)).fit(y)
        assert m.weights_ is None


class TestOffsetIdentity:
    """offset=None ≡ offset=0 · 𝟏."""

    def test_no_offset_vs_zeros(self) -> None:
        rng = np.random.default_rng(20)
        y = rng.uniform(0.05, 0.95, 50)
        m1 = MLT(order=4, support=(0.0, 1.0)).fit(y)
        m2 = MLT(order=4, support=(0.0, 1.0)).fit(y, offset=np.zeros(50))
        np.testing.assert_allclose(m1.theta_, m2.theta_, rtol=1e-5, atol=1e-8)

    def test_no_offset_vs_zeros_with_X(self) -> None:
        rng = np.random.default_rng(21)
        x = rng.normal(0, 1, 60)
        y = 0.5 + 0.4 * x + rng.normal(0, 0.5, 60)
        a, b = y.min() - 0.1, y.max() + 0.1
        X = x[:, None]
        m1 = MLT(order=4, support=(a, b)).fit(y, X)
        m2 = MLT(order=4, support=(a, b)).fit(y, X, offset=np.zeros(60))
        np.testing.assert_allclose(m1.theta_, m2.theta_, rtol=1e-5, atol=1e-8)

    def test_offset_stored_after_fit(self) -> None:
        y = np.linspace(0.05, 0.95, 20)
        o = np.zeros(20)
        m = MLT(order=3, support=(0.0, 1.0)).fit(y, offset=o)
        assert m.offset_ is not None
        np.testing.assert_array_equal(m.offset_, o)

    def test_no_offset_stored_none(self) -> None:
        y = np.linspace(0.05, 0.95, 20)
        m = MLT(order=3, support=(0.0, 1.0)).fit(y)
        assert m.offset_ is None


# ---------------------------------------------------------------------------
# Weight scaling tests
# ---------------------------------------------------------------------------


class TestWeightScaling:
    def test_uniform_doubling_leaves_theta_unchanged(self) -> None:
        """Doubling all weights leaves theta unchanged."""
        rng = np.random.default_rng(30)
        y = rng.uniform(0.05, 0.95, 60)
        m1 = MLT(order=4, support=(0.0, 1.0)).fit(y)
        m2 = MLT(order=4, support=(0.0, 1.0)).fit(y, weights=2.0 * np.ones(60))
        np.testing.assert_allclose(m1.theta_, m2.theta_, rtol=1e-4, atol=1e-6)

    def test_uniform_doubling_scales_loglik(self) -> None:
        """Doubling all weights scales log-likelihood by 2."""
        rng = np.random.default_rng(31)
        y = rng.uniform(0.05, 0.95, 60)
        m1 = MLT(order=4, support=(0.0, 1.0)).fit(y)
        m2 = MLT(order=4, support=(0.0, 1.0)).fit(y, weights=2.0 * np.ones(60))
        assert m1.result_ is not None and m2.result_ is not None  # type: ignore[union-attr]
        np.testing.assert_allclose(
            m2.result_.log_likelihood,
            2.0 * m1.result_.log_likelihood,
            rtol=1e-5,
        )

    def test_replication_invariance(self) -> None:
        """Fit on 2n with weight 0.5 gives same theta as fit on n."""
        rng = np.random.default_rng(32)
        y = rng.uniform(0.05, 0.95, 40)
        y2 = np.tile(y, 2)  # 80 obs, each appearing twice
        m1 = MLT(order=4, support=(0.0, 1.0)).fit(y)
        m2 = MLT(order=4, support=(0.0, 1.0)).fit(y2, weights=0.5 * np.ones(80))
        np.testing.assert_allclose(m1.theta_, m2.theta_, rtol=1e-4, atol=1e-6)

    def test_replication_invariance_with_X(self) -> None:
        """Replication invariance with covariates."""
        rng = np.random.default_rng(33)
        x = rng.normal(0, 1, 40)
        y = 0.5 + 0.3 * x + rng.normal(0, 0.5, 40)
        a, b = y.min() - 0.1, y.max() + 0.1
        X = x[:, None]
        y2, X2 = np.tile(y, 2), np.tile(X, (2, 1))
        m1 = MLT(order=4, support=(a, b)).fit(y, X)
        m2 = MLT(order=4, support=(a, b)).fit(y2, X2, weights=0.5 * np.ones(80))
        np.testing.assert_allclose(m1.theta_, m2.theta_, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Offset tests
# ---------------------------------------------------------------------------


class TestOffsetPredict:
    def test_offset_shifts_trafo(self) -> None:
        """predict(y, offset_new=c) ≡ predict(y, what='trafo') + c."""
        rng = np.random.default_rng(40)
        y = rng.uniform(0.05, 0.95, 30)
        m, _ = _simple_fit(n=60)
        c = 0.5
        h_base = m.predict(y, what="trafo")
        h_off = m.predict(y, what="trafo", offset_new=c * np.ones(30))
        np.testing.assert_allclose(h_off, h_base + c, rtol=1e-12)

    def test_offset_shifts_distribution(self) -> None:
        """Distribution at offset h is the CDF evaluated at h+c."""
        from scipy.stats import norm as _norm

        rng = np.random.default_rng(41)
        y = rng.uniform(0.05, 0.95, 20)
        m, _ = _simple_fit(n=60)
        c = 0.3
        h_base = m.predict(y, what="trafo")
        cdf_base = _norm.cdf(h_base)
        cdf_off = m.predict(y, what="distribution", offset_new=c * np.ones(20))
        np.testing.assert_allclose(cdf_off, _norm.cdf(h_base + c), rtol=1e-12)
        assert not np.allclose(cdf_off, cdf_base)  # actually shifted

    def test_offset_per_observation(self) -> None:
        """Per-observation offset correctly shifts each h_i independently."""
        rng = np.random.default_rng(42)
        y = rng.uniform(0.05, 0.95, 20)
        m, _ = _simple_fit(n=60)
        off = rng.normal(0, 0.2, 20)
        h_base = m.predict(y, what="trafo")
        h_off = m.predict(y, what="trafo", offset_new=off)
        np.testing.assert_allclose(h_off, h_base + off, rtol=1e-12)

    def test_offset_in_quantile_prediction(self) -> None:
        """predict(p, offset_new=c, what='quantile') inverts h(q) = ppf(p) - c."""
        from scipy.stats import norm as _norm

        m, _ = _simple_fit(n=60)
        probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        c = 0.3
        q_base = m.predict(probs, what="quantile")
        q_off = m.predict(probs, what="quantile", offset_new=c * np.ones(5))
        # With positive offset, the quantile should shift such that h(q) + c = ppf(p)
        # → h(q_off) = ppf(p) - c < ppf(p) → q_off < q_base
        assert np.all(q_off < q_base)
        # Verify: h(q_off) + c should ≈ ppf(p)
        h_off = m.predict(q_off, what="trafo")
        np.testing.assert_allclose(h_off + c, _norm.ppf(probs), atol=1e-4)

    def test_zero_offset_matches_no_offset_predict(self) -> None:
        """predict(y, offset_new=0) ≡ predict(y)."""
        rng = np.random.default_rng(44)
        y = rng.uniform(0.05, 0.95, 20)
        m, _ = _simple_fit(n=60)
        for what in ("trafo", "distribution", "density"):
            base = m.predict(y, what=what)
            with_zero = m.predict(y, what=what, offset_new=np.zeros(20))
            np.testing.assert_allclose(with_zero, base, rtol=1e-12)

    def test_wrong_shape_offset_predict_raises(self) -> None:
        rng = np.random.default_rng(45)
        y = rng.uniform(0.05, 0.95, 20)
        m, _ = _simple_fit(n=60)
        with pytest.raises(ValueError, match="offset must have shape"):
            m.predict(y, offset_new=np.zeros(5))

    def test_wrong_shape_offset_quantile_raises(self) -> None:
        m, _ = _simple_fit(n=60)
        probs = np.array([0.1, 0.5, 0.9])
        with pytest.raises(ValueError, match="offset must have shape"):
            m.predict(probs, what="quantile", offset_new=np.zeros(10))

    def test_nonfinite_offset_predict_raises(self) -> None:
        rng = np.random.default_rng(46)
        y = rng.uniform(0.05, 0.95, 20)
        m, _ = _simple_fit(n=60)
        with pytest.raises(ValueError, match="finite"):
            m.predict(y, offset_new=np.full(20, np.nan))


class TestOffsetFit:
    def test_constant_offset_shifts_theta_b(self) -> None:
        """A constant offset c is absorbed into theta_b shift."""
        rng = np.random.default_rng(50)
        y = rng.uniform(0.05, 0.95, 60)
        c = 0.5
        m0 = MLT(order=4, support=(0.0, 1.0)).fit(y)
        m1 = MLT(order=4, support=(0.0, 1.0)).fit(y, offset=c * np.ones(60))
        # theta_b should shift by approximately -c (offset is added to h,
        # so the basis coefficients shift down to compensate).
        h0 = m0.predict(y, what="trafo")
        h1 = m1.predict(y, what="trafo", offset_new=c * np.ones(60))
        np.testing.assert_allclose(h0, h1, atol=0.05)

    def test_offset_residuals_sum_near_zero(self) -> None:
        """Score residuals at the MLE sum to zero regardless of offset."""
        rng = np.random.default_rng(51)
        y = rng.uniform(0.05, 0.95, 60)
        off = rng.normal(0, 0.1, 60)
        m = MLT(order=4, support=(0.0, 1.0)).fit(y, offset=off)
        r = m.residuals(type="score")
        assert abs(r.sum()) < 1e-4


# ---------------------------------------------------------------------------
# score() with weights/offset
# ---------------------------------------------------------------------------


class TestScoreMethod:
    def test_score_with_ones_matches_no_weights(self) -> None:
        m, y = _simple_fit()
        s1 = m.score(y)
        s2 = m.score(y, weights=np.ones(len(y)))
        np.testing.assert_allclose(s1, s2, rtol=1e-10)

    def test_score_with_zeros_offset_matches_no_offset(self) -> None:
        m, y = _simple_fit()
        s1 = m.score(y)
        s2 = m.score(y, offset=np.zeros(len(y)))
        np.testing.assert_allclose(s1, s2, rtol=1e-10)

    def test_score_weights_scale_loglik(self) -> None:
        m, y = _simple_fit()
        s1 = m.score(y)
        s2 = m.score(y, weights=2.0 * np.ones(len(y)))
        np.testing.assert_allclose(s2, 2.0 * s1, rtol=1e-10)


# ---------------------------------------------------------------------------
# Residuals with weights
# ---------------------------------------------------------------------------


class TestResidualsSumToZero:
    def test_score_residuals_sum_zero_weighted(self) -> None:
        """Weighted score residuals sum ≈ 0 at the MLE."""
        rng = np.random.default_rng(60)
        y = rng.uniform(0.05, 0.95, 60)
        w = rng.integers(1, 5, 60).astype(float)
        m = MLT(order=4, support=(0.0, 1.0)).fit(y, weights=w)
        r = m.residuals(type="score")
        assert abs(r.sum()) < 1e-4

    def test_score_residuals_unweighted_sum_zero(self) -> None:
        """Score residuals sum ≈ 0 at the (unweighted) MLE."""
        m, y = _simple_fit()
        r = m.residuals(type="score")
        assert abs(r.sum()) < 1e-4


# ---------------------------------------------------------------------------
# confband() with offset
# ---------------------------------------------------------------------------


class TestConfbandOffset:
    def test_zero_offset_matches_no_offset(self) -> None:
        m, y = _simple_fit()
        g = np.linspace(0.05, 0.95, 20)
        band0 = m.confband(g)
        band_z = m.confband(g, offset=np.zeros(20))
        np.testing.assert_allclose(band0, band_z, rtol=1e-12)

    def test_constant_offset_shifts_distribution_band(self) -> None:
        from scipy.stats import norm as _norm

        m, y = _simple_fit()
        g = np.linspace(0.1, 0.9, 20)
        c = 0.5
        band0 = m.confband(g, what="distribution")
        band1 = m.confband(g, what="distribution", offset=c * np.ones(20))
        h0 = m.predict(g, what="trafo")
        h1 = h0 + c
        est0_expected = _norm.cdf(h0)
        est1_expected = _norm.cdf(h1)
        np.testing.assert_allclose(band0[:, 0], est0_expected, rtol=1e-10)
        np.testing.assert_allclose(band1[:, 0], est1_expected, rtol=1e-10)

    def test_wrong_shape_offset_confband_raises(self) -> None:
        m, _ = _simple_fit()
        g = np.linspace(0.05, 0.95, 20)
        with pytest.raises(ValueError, match="offset must have shape"):
            m.confband(g, offset=np.zeros(5))

    def test_nonfinite_offset_confband_raises(self) -> None:
        m, _ = _simple_fit()
        g = np.linspace(0.05, 0.95, 20)
        with pytest.raises(ValueError, match="finite"):
            m.confband(g, offset=np.full(20, np.inf))


# ---------------------------------------------------------------------------
# tram convenience layer
# ---------------------------------------------------------------------------


class TestTramWeightsOffset:
    def test_coxph_survival_offset_shifts_result(self) -> None:
        rng = np.random.default_rng(70)
        t = rng.exponential(1.0, 80)
        c = rng.exponential(3.0, 80)
        y = np.minimum(t, c)
        event = (t <= c).astype(float)
        cd = CensoredData.right_censored(y, censored=event == 0)
        a, b = 1e-3, y.max() + 0.1
        m = Coxph(support=(a, b), order=4).fit(cd)
        g = np.linspace(a + 0.01, b - 0.01, 10)
        s0 = m.survival(g)
        off = 0.5 * np.ones(10)
        s1 = m.survival(g, offset=off)
        assert not np.allclose(s0, s1), "offset should change survivor values"

    def test_coxph_survival_zero_offset_identity(self) -> None:
        rng = np.random.default_rng(71)
        t = rng.exponential(1.0, 60)
        c = rng.exponential(3.0, 60)
        y = np.minimum(t, c)
        event = (t <= c).astype(float)
        cd = CensoredData.right_censored(y, censored=event == 0)
        a, b = 1e-3, y.max() + 0.1
        m = Coxph(support=(a, b), order=4).fit(cd)
        g = np.linspace(a + 0.01, b - 0.01, 10)
        s0 = m.survival(g)
        s1 = m.survival(g, offset=np.zeros(10))
        np.testing.assert_allclose(s0, s1, rtol=1e-12)


# ---------------------------------------------------------------------------
# R reference parity
# ---------------------------------------------------------------------------


def _load_weights_reference(model_name: str) -> dict | None:
    """Load R-generated weights reference; return None if files not present."""
    needed = [
        REF_DIR / f"weights_{model_name}_w.txt",
        REF_DIR / f"weights_{model_name}_theta.txt",
        REF_DIR / f"weights_{model_name}_ll.txt",
        REF_DIR / f"weights_{model_name}_estfun.txt",
    ]
    if not all(p.exists() for p in needed):
        return None
    w = np.loadtxt(needed[0])
    theta = np.loadtxt(needed[1])
    ll = float(np.loadtxt(needed[2]))
    estfun_flat = np.loadtxt(needed[3])
    n = len(w)
    p_plus_q = len(theta)
    return {
        "w": w,
        "theta": theta,
        "ll": ll,
        "estfun": estfun_flat.reshape(n, p_plus_q),
    }


def _load_vcov_reference_for_weights(model_name: str) -> dict | None:
    """Load vcov reference (y, x, support) for re-fitting with weights."""
    needed = [
        REF_DIR / f"vcov_{model_name}_y.txt",
        REF_DIR / f"vcov_{model_name}_x.txt",
        REF_DIR / f"vcov_{model_name}_support.txt",
        REF_DIR / f"vcov_{model_name}_theta.txt",
    ]
    if not all(p.exists() for p in needed):
        return None
    theta = np.loadtxt(needed[3])
    data: dict = {
        "y": np.loadtxt(needed[0]),
        "x": np.loadtxt(needed[1]).reshape(-1, 1),
        "support": tuple(np.loadtxt(needed[2])),
        "p_plus_q": len(theta),
    }
    event_path = REF_DIR / f"vcov_{model_name}_event.txt"
    if event_path.exists():
        data["event"] = np.loadtxt(event_path).astype(int)
    return data


@pytest.mark.parametrize(
    "model_name, tram_cls, base_dist, censoring, beta_sign",
    [
        ("boxcox", BoxCox, "normal", CensoringType.NONE, -1.0),
        ("colr", Colr, "logistic", CensoringType.NONE, 1.0),
        ("coxph", Coxph, "min_extreme_value", CensoringType.RIGHT, 1.0),
    ],
)
def test_r_weights_parity(
    model_name: str,
    tram_cls: type,
    base_dist: str,
    censoring: CensoringType,
    beta_sign: float,
) -> None:
    """Weighted parity with R at both R theta and pymlt's fitted theta."""
    wref = _load_weights_reference(model_name)
    vref = _load_vcov_reference_for_weights(model_name)
    if wref is None or vref is None:
        pytest.skip(
            f"weights_{model_name}_* or vcov_{model_name}_* reference files "
            "not yet generated — run Rscript reference/generate_reference.R"
        )

    a, b = vref["support"]
    order = vref["p_plus_q"] - 2  # p = order + 1, q = 1 (one covariate)
    y_raw = vref["y"]
    X = vref["x"]
    w = wref["w"]

    if censoring == CensoringType.RIGHT:
        event = vref["event"]
        y_obj: np.ndarray | CensoredData = CensoredData.right_censored(
            y_raw, censored=event == 0
        )
    else:
        y_obj = y_raw

    m = MLT(
        order=order,
        support=(float(a), float(b)),
        censoring=censoring,
        base_distribution=base_dist,  # type: ignore[arg-type]
    ).fit(y_obj, X, weights=w)

    assert m.theta_ is not None

    # beta sign flip: R BoxCox uses negative = TRUE → flip beta entries
    p = order + 1
    sign_vec = np.ones(p + 1)
    sign_vec[p:] = beta_sign
    theta_r = wref["theta"] * sign_vec

    # Exact parity check at R theta (objective and weighted score matrix).
    ll_at_r_theta = log_likelihood(
        theta_r,
        m.basis,
        y_obj,
        X=X,
        censoring=censoring,
        base_distribution=base_dist,  # type: ignore[arg-type]
        weights=w,
    )
    np.testing.assert_allclose(
        ll_at_r_theta,
        wref["ll"],
        rtol=1e-8,
        atol=1e-10,
        err_msg=f"{model_name}: weighted log-likelihood mismatch at R theta",
    )

    scores_at_r_theta = score_matrix(
        theta_r,
        m.basis,
        y_obj,
        X=X,
        censoring=censoring,
        base_distribution=base_dist,  # type: ignore[arg-type]
        weights=w,
    )
    # R estfun convention here is ∂(-ℓ)/∂θ; pymlt score_matrix returns ∂ℓ/∂θ.
    # BoxCox additionally needs β-sign conversion (negative=TRUE in R).
    estfun_r_converted = -wref["estfun"] * sign_vec[None, :]
    np.testing.assert_allclose(
        scores_at_r_theta,
        estfun_r_converted,
        rtol=1e-6,
        atol=1e-8,
        err_msg=f"{model_name}: weighted score matrix mismatch at R theta",
    )

    # Fit-parity check at pymlt's own optimum: objective value should match R.
    np.testing.assert_allclose(
        m.result_.log_likelihood,  # type: ignore[union-attr]
        wref["ll"],
        rtol=1e-4,
        err_msg=f"{model_name}: fitted log-likelihood deviates from R reference",
    )

    # Constraint-robust score identity at fitted theta:
    # sum_i estfun_i == grad(ℓ) == -grad(NLL).
    estfun_pymlt = m.estfun()
    _, grad_nll = negative_log_likelihood(
        m.theta_,
        m.basis,
        y_obj,
        X=X,
        censoring=censoring,
        gradient=True,
        base_distribution=base_dist,  # type: ignore[arg-type]
        weights=w,
    )
    np.testing.assert_allclose(
        estfun_pymlt.sum(axis=0),
        -grad_nll,
        atol=1e-10,
        err_msg=f"{model_name}: estfun column sums must equal -grad(NLL)",
    )
