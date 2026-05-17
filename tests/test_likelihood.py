"""Tests for pymlt.likelihood — log-likelihood correctness, stability, gradients."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import check_grad

from pymlt.basis import BernsteinBasis
from pymlt.likelihood import (
    _VALID_BASE_DISTRIBUTIONS,
    _get_dist,
    _log_diff_ndtr,
    log_likelihood,
    negative_log_likelihood,
)
from pymlt.variables import CensoredData, CensoringType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_basis(order: int = 3, support: tuple = (0.0, 1.0)) -> BernsteinBasis:
    return BernsteinBasis(order=order, support=support)


def ascending_theta(order: int, low: float = 0.0, step: float = 0.5) -> np.ndarray:
    return np.array([low + step * i for i in range(order + 1)])


# ---------------------------------------------------------------------------
# _log_diff_ndtr — numerical stability helper
# ---------------------------------------------------------------------------


class TestLogDiffNdtr:
    def test_standard_case(self):
        """log(Φ(1) - Φ(0)) matches reference."""
        from scipy.stats import norm

        expected = np.log(norm.cdf(1.0) - norm.cdf(0.0))
        result = _log_diff_ndtr(np.array([0.0]), np.array([1.0]))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_no_inf_for_narrow_interval(self):
        """Very narrow intervals must not produce -inf."""
        a = np.array([0.0])
        b = a + 1e-8
        result = _log_diff_ndtr(a, b)
        assert np.isfinite(result), f"Expected finite, got {result}"

    def test_no_inf_for_very_narrow_interval(self):
        a = np.array([0.0])
        b = a + 1e-12
        result = _log_diff_ndtr(a, b)
        assert np.isfinite(result)

    def test_wide_interval_matches_log_ndtr(self):
        """For very wide interval, result ≈ log Φ(b)."""
        from scipy.special import log_ndtr

        b = np.array([2.0])
        a = np.array([-100.0])
        result = _log_diff_ndtr(a, b)
        np.testing.assert_allclose(result, log_ndtr(b), atol=1e-10)

    def test_symmetric_interval(self):
        """log(Φ(1) - Φ(-1)) = log(2*Φ(1)-1)."""
        from scipy.stats import norm

        expected = np.log(2 * norm.cdf(1.0) - 1.0)
        result = _log_diff_ndtr(np.array([-1.0]), np.array([1.0]))
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# NONE — exact observations
# ---------------------------------------------------------------------------


class TestLogLikelihoodNone:
    def test_known_value(self):
        """Hardcoded reference: order=2, theta=[0,1,2], y=[0.25, 0.5, 0.75].

        h = [0.5, 1.0, 1.5],  h' = [2.0, 2.0, 2.0]
        ℓ = Σ norm.logpdf(h) + Σ log(h')
          = -4.5068155996140183 + 2.0794415416798357
          = -2.4273740579341826
        """
        basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        theta = np.array([0.0, 1.0, 2.0])
        y = np.array([0.25, 0.5, 0.75])
        result = log_likelihood(theta, basis, y)
        np.testing.assert_allclose(result, -2.4273740579341826, rtol=1e-10)

    def test_manual_computation(self):
        """LL equals Σ logpdf(h) + Σ log(h') computed directly."""
        from scipy.stats import norm as _norm

        basis = make_basis(order=4)
        theta = ascending_theta(4, step=0.3)
        y = np.linspace(0.1, 0.9, 8)

        B = basis.evaluate(y)
        D = basis.derivative(y, order=1)
        h = B @ theta
        hp = D @ theta
        expected = np.sum(_norm.logpdf(h)) + np.sum(np.log(hp))

        result = log_likelihood(theta, basis, y)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_ndarray_input(self):
        basis = make_basis(order=3)
        theta = ascending_theta(3)
        y = np.array([0.3, 0.5, 0.7])
        result = log_likelihood(theta, basis, y)
        assert np.isfinite(result)

    def test_censored_data_none_equals_ndarray(self):
        """CensoredData with censoring=NONE gives same result as plain array."""
        basis = make_basis(order=3)
        theta = ascending_theta(3)
        y_arr = np.array([0.2, 0.5, 0.8])
        cd = CensoredData.from_exact(y_arr)
        result_arr = log_likelihood(theta, basis, y_arr)
        result_cd = log_likelihood(theta, basis, cd, censoring=CensoringType.NONE)
        np.testing.assert_allclose(result_arr, result_cd, rtol=1e-12)

    def test_monotonicity_violation_raises(self):
        """Descending theta → h' ≤ 0 → log(h') = -inf → ValueError."""
        basis = make_basis(order=2)
        theta = np.array([2.0, 1.0, 0.0])  # strictly descending
        y = np.array([0.5])
        with pytest.raises(ValueError, match="monoton"):
            log_likelihood(theta, basis, y)


# ---------------------------------------------------------------------------
# RIGHT — right-censored
# ---------------------------------------------------------------------------


class TestLogLikelihoodRight:
    def _make_right_data(
        self, n: int = 50, frac_censored: float = 0.3, seed: int = 0
    ) -> tuple:
        """Synthetic right-censored survival data on [0, 3]."""
        rng = np.random.default_rng(seed)
        support = (0.0, 3.0)
        basis = BernsteinBasis(order=3, support=support)
        theta = ascending_theta(3, step=0.5)
        y_true = rng.uniform(0.1, 2.9, n)
        is_censored = rng.random(n) < frac_censored
        cd = CensoredData.right_censored(y_true, is_censored)
        return basis, theta, cd

    def test_sign_and_finiteness(self):
        basis, theta, cd = self._make_right_data()
        result = log_likelihood(theta, basis, cd, censoring=CensoringType.RIGHT)
        assert np.isfinite(result)

    def test_all_censored_uses_logsf(self):
        """All censored → LL = Σ norm.logsf(h)."""
        from scipy.stats import norm as _norm

        basis = make_basis(order=3)
        theta = ascending_theta(3)
        y = np.array([0.2, 0.5, 0.8])
        cd = CensoredData.right_censored(y, np.array([True, True, True]))
        B = basis.evaluate(y)
        h = B @ theta
        expected = float(np.sum(_norm.logsf(h)))
        result = log_likelihood(theta, basis, cd, censoring=CensoringType.RIGHT)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_no_censoring_equals_none(self):
        """Right-censored with no censored obs == NONE."""
        basis = make_basis(order=3)
        theta = ascending_theta(3)
        y = np.array([0.2, 0.5, 0.8])
        cd = CensoredData.right_censored(y, np.array([False, False, False]))
        ll_none = log_likelihood(theta, basis, y)
        ll_right = log_likelihood(theta, basis, cd, censoring=CensoringType.RIGHT)
        np.testing.assert_allclose(ll_right, ll_none, rtol=1e-12)

    def test_reference_npy(self):
        """Right-censored LL matches mlt::logLik at R's fitted θ on the same data.

        Reference is generated by ``reference/generate_reference.R`` which
        fits an mlt model with a Bernstein basis of order 4 on (0, 1) to 200
        right-censored observations, then writes y, the event indicator,
        θ, and the scalar log-likelihood. This test reconstructs the same
        censored dataset in pymlt, evaluates ``log_likelihood`` at R's θ,
        and asserts exact agreement (same formula, same data, same θ).
        """
        import pathlib

        ref_dir = pathlib.Path(__file__).parent.parent / "reference"
        required = [
            ref_dir / "ll_right_y.txt",
            ref_dir / "ll_right_event.txt",
            ref_dir / "ll_right_theta.txt",
            ref_dir / "ll_right_ll.txt",
        ]
        if not all(p.exists() for p in required):
            pytest.skip(
                "ll_right_* reference files not yet generated — "
                "run Rscript reference/generate_reference.R"
            )

        y = np.loadtxt(required[0])
        event = np.loadtxt(required[1]).astype(int)
        theta = np.loadtxt(required[2])
        ll_ref = float(np.loadtxt(required[3]))

        # R's Surv(time, event) uses event=1 for observed, 0 for censored;
        # pymlt's CensoredData.right_censored takes is_censored (True ⇒ censored).
        is_censored = event == 0

        basis = BernsteinBasis(order=len(theta) - 1, support=(0.0, 1.0))
        cd = CensoredData.right_censored(y, is_censored)
        ll = log_likelihood(theta, basis, cd, censoring=CensoringType.RIGHT)

        np.testing.assert_allclose(ll, ll_ref, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# LEFT — left-censored
# ---------------------------------------------------------------------------


class TestLogLikelihoodLeft:
    def test_all_censored_uses_logcdf(self):
        """All censored → LL = Σ log_ndtr(h)."""
        from scipy.special import log_ndtr as _log_ndtr

        basis = make_basis(order=3)
        theta = ascending_theta(3)
        y = np.array([0.2, 0.5, 0.8])
        cd = CensoredData.left_censored(y, np.array([True, True, True]))
        B = basis.evaluate(y)
        h = B @ theta
        expected = float(np.sum(_log_ndtr(h)))
        result = log_likelihood(theta, basis, cd, censoring=CensoringType.LEFT)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_no_censoring_equals_none(self):
        basis = make_basis(order=3)
        theta = ascending_theta(3)
        y = np.array([0.2, 0.5, 0.8])
        cd = CensoredData.left_censored(y, np.array([False, False, False]))
        ll_none = log_likelihood(theta, basis, y)
        ll_left = log_likelihood(theta, basis, cd, censoring=CensoringType.LEFT)
        np.testing.assert_allclose(ll_left, ll_none, rtol=1e-12)


# ---------------------------------------------------------------------------
# INTERVAL — interval-censored
# ---------------------------------------------------------------------------


class TestLogLikelihoodInterval:
    def test_no_inf_for_narrow_intervals(self):
        """Narrow intervals (Δy = 1e-6) must not produce -inf."""
        basis = make_basis(order=3)
        theta = ascending_theta(3)
        centers = np.array([0.2, 0.5, 0.8])
        eps = 1e-6
        cd = CensoredData.interval_censored(centers - eps, centers + eps)
        result = log_likelihood(theta, basis, cd, censoring=CensoringType.INTERVAL)
        assert np.isfinite(result), f"Expected finite, got {result}"

    def test_very_narrow_intervals_finite(self):
        basis = make_basis(order=3)
        theta = ascending_theta(3)
        centers = np.array([0.3, 0.6])
        eps = 1e-10
        cd = CensoredData.interval_censored(centers - eps, centers + eps)
        result = log_likelihood(theta, basis, cd, censoring=CensoringType.INTERVAL)
        assert np.isfinite(result)

    def test_wide_interval_close_to_none(self):
        """Extremely wide interval → almost all probability covered → LL ≈ 0."""
        basis = BernsteinBasis(order=3, support=(-10.0, 10.0))
        theta = ascending_theta(3, low=-2.0, step=1.0)
        # Interval so wide that Φ(h_upper) - Φ(h_lower) ≈ 1
        cd = CensoredData.interval_censored(np.array([-9.9]), np.array([9.9]))
        result = log_likelihood(theta, basis, cd, censoring=CensoringType.INTERVAL)
        assert result > -1.0, f"Expected ≈ 0, got {result}"

    def test_interval_manual(self):
        """LL = Σ _log_diff_ndtr(h_lo, h_hi) computed independently."""
        basis = make_basis(order=3)
        theta = ascending_theta(3)
        lo = np.array([0.1, 0.4, 0.7])
        hi = np.array([0.3, 0.6, 0.9])
        cd = CensoredData.interval_censored(lo, hi)
        B_lo = basis.evaluate(lo)
        B_hi = basis.evaluate(hi)
        h_lo = B_lo @ theta
        h_hi = B_hi @ theta
        expected = float(np.sum(_log_diff_ndtr(h_lo, h_hi)))
        result = log_likelihood(theta, basis, cd, censoring=CensoringType.INTERVAL)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Truncation (delayed entry) — exercises the centralised correction injected
# by every public dispatcher (log_likelihood, negative_log_likelihood,
# score_matrix, hessian, intercept_score) when CensoredData carries
# trunc_lower / trunc_upper.
# ---------------------------------------------------------------------------


class TestLogLikelihoodTruncation:
    @staticmethod
    def _manual_log_p(
        theta: np.ndarray,
        basis: BernsteinBasis,
        lo: np.ndarray,
        hi: np.ndarray,
        X: np.ndarray | None = None,
        beta: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-observation ``log P_i`` computed with scipy.norm only.

        Mirrors the formula the truncation correction is supposed to
        implement.  Independent of pymlt internals.
        """
        from scipy.special import log_ndtr
        from scipy.stats import norm

        n = len(lo)
        log_p = np.zeros(n, dtype=float)
        theta_b = theta[: basis.order + 1]
        h_shift = X @ beta if (X is not None and beta is not None) else np.zeros(n)
        for i in range(n):
            shift_i = float(h_shift[i])
            li, ui = lo[i], hi[i]
            if np.isfinite(li) and np.isfinite(ui):
                h_l = float((basis.evaluate(np.array([li])) @ theta_b)[0]) + shift_i
                h_u = float((basis.evaluate(np.array([ui])) @ theta_b)[0]) + shift_i
                log_p[i] = float(np.log(norm.cdf(h_u) - norm.cdf(h_l)))
            elif np.isfinite(ui):
                h_u = float((basis.evaluate(np.array([ui])) @ theta_b)[0]) + shift_i
                log_p[i] = float(log_ndtr(h_u))
            elif np.isfinite(li):
                h_l = float((basis.evaluate(np.array([li])) @ theta_b)[0]) + shift_i
                log_p[i] = float(norm.logsf(h_l))
            # else: both infinite → 0
        return log_p

    def test_construction_rejects_observation_outside_truncation(self):
        """CensoredData must error when A_i ⊄ B_i for any row.

        The truncation correction relies on
        ``ℓ_i = log P(A_i) − log P(B_i)`` which only equals
        ``log[P(A_i ∩ B_i) / P(B_i)]`` when ``A_i ⊆ B_i``.  Inputs that
        violate this would silently produce wrong likelihoods, so the
        constraint is enforced loud at construction.
        """
        # Exact obs above the upper truncation bound.
        with pytest.raises(ValueError, match=r"contained in its truncation"):
            CensoredData(
                exact=np.array([0.5, 0.9]),
                lower=np.array([0.5, 0.9]),
                upper=np.array([0.5, 0.9]),
                trunc_lower=np.array([0.0, 0.0]),
                trunc_upper=np.array([1.0, 0.8]),
            )
        # Right-censored row paired with a finite upper truncation
        # (upper=+inf cannot fit inside any finite trunc_upper).
        with pytest.raises(ValueError, match=r"contained in its truncation"):
            CensoredData(
                exact=np.array([np.nan]),
                lower=np.array([0.3]),
                upper=np.array([np.inf]),
                trunc_lower=np.array([0.2]),
                trunc_upper=np.array([0.9]),
            )
        # Interval-censored row whose upper bound exceeds trunc_upper.
        with pytest.raises(ValueError, match=r"contained in its truncation"):
            CensoredData(
                exact=np.array([np.nan]),
                lower=np.array([0.30]),
                upper=np.array([0.70]),
                trunc_lower=np.array([0.20]),
                trunc_upper=np.array([0.60]),
            )
        # Left-censored row whose lower bound (-inf) is below trunc_lower.
        with pytest.raises(ValueError, match=r"contained in its truncation"):
            CensoredData(
                exact=np.array([np.nan]),
                lower=np.array([-np.inf]),
                upper=np.array([0.5]),
                trunc_lower=np.array([0.1]),
                trunc_upper=np.array([np.inf]),
            )

    def test_noop_when_both_bounds_none(self):
        """Truncation correction is a no-op when neither bound is set."""
        basis = make_basis(order=3)
        theta = ascending_theta(3, step=0.4)
        y = np.array([0.2, 0.5, 0.8])
        cd = CensoredData.from_exact(y)
        assert cd.trunc_lower is None and cd.trunc_upper is None
        ll_arr = log_likelihood(theta, basis, y)
        ll_cd = log_likelihood(theta, basis, cd, censoring=CensoringType.NONE)
        np.testing.assert_allclose(ll_arr, ll_cd, rtol=1e-12)

    def test_noop_when_bounds_are_infinite(self):
        """Explicit ±inf bounds reproduce the unconditional log-likelihood."""
        basis = make_basis(order=3)
        theta = ascending_theta(3, step=0.4)
        y = np.array([0.2, 0.5, 0.8])
        n = len(y)
        cd_inf = CensoredData(
            exact=y.copy(),
            lower=y.copy(),
            upper=y.copy(),
            trunc_lower=np.full(n, -np.inf),
            trunc_upper=np.full(n, np.inf),
        )
        ll_uncond = log_likelihood(theta, basis, y)
        ll_inf = log_likelihood(theta, basis, cd_inf, censoring=CensoringType.NONE)
        np.testing.assert_allclose(ll_inf, ll_uncond, rtol=1e-12)

    def test_exact_with_two_sided_truncation(self):
        """ℓ = ℓ_uncond − Σ log[Φ(h(u)) − Φ(h(l))] computed manually."""
        basis = make_basis(order=3)
        theta = ascending_theta(3, step=0.4)
        y = np.array([0.30, 0.55, 0.70, 0.85])
        lo = np.array([0.10, 0.40, 0.50, 0.60])
        hi = np.array([0.50, 0.70, 0.85, 0.95])
        cd = CensoredData(
            exact=y.copy(),
            lower=y.copy(),
            upper=y.copy(),
            trunc_lower=lo,
            trunc_upper=hi,
        )
        ll_uncond = log_likelihood(theta, basis, y)
        log_p = self._manual_log_p(theta, basis, lo, hi)
        expected = ll_uncond - log_p.sum()
        ll = log_likelihood(theta, basis, cd, censoring=CensoringType.NONE)
        np.testing.assert_allclose(ll, expected, rtol=1e-10)

    def test_one_sided_lower_truncation(self):
        """Lower-only truncation: correction = −Σ log S(h(l_i))."""
        basis = make_basis(order=3)
        theta = ascending_theta(3, step=0.4)
        y = np.array([0.30, 0.55, 0.80])
        lo = np.array([0.10, 0.40, 0.60])
        cd = CensoredData(
            exact=y.copy(),
            lower=y.copy(),
            upper=y.copy(),
            trunc_lower=lo,
        )
        ll_uncond = log_likelihood(theta, basis, y)
        hi = np.full(len(y), np.inf)
        log_p = self._manual_log_p(theta, basis, lo, hi)
        expected = ll_uncond - log_p.sum()
        ll = log_likelihood(theta, basis, cd, censoring=CensoringType.NONE)
        np.testing.assert_allclose(ll, expected, rtol=1e-10)

    def test_one_sided_upper_truncation(self):
        """Upper-only truncation: correction = −Σ log Φ(h(u_i))."""
        basis = make_basis(order=3)
        theta = ascending_theta(3, step=0.4)
        y = np.array([0.20, 0.45, 0.65])
        hi = np.array([0.40, 0.60, 0.85])
        cd = CensoredData(
            exact=y.copy(),
            lower=y.copy(),
            upper=y.copy(),
            trunc_upper=hi,
        )
        ll_uncond = log_likelihood(theta, basis, y)
        lo = np.full(len(y), -np.inf)
        log_p = self._manual_log_p(theta, basis, lo, hi)
        expected = ll_uncond - log_p.sum()
        ll = log_likelihood(theta, basis, cd, censoring=CensoringType.NONE)
        np.testing.assert_allclose(ll, expected, rtol=1e-10)

    def test_mixed_finite_and_infinite_bounds(self):
        """Per-row mix of two-sided and one-sided truncation."""
        basis = make_basis(order=3)
        theta = ascending_theta(3, step=0.4)
        y = np.array([0.30, 0.45, 0.60, 0.80])
        lo = np.array([0.10, -np.inf, 0.50, -np.inf])
        hi = np.array([0.50, 0.60, np.inf, 0.95])
        cd = CensoredData(
            exact=y.copy(),
            lower=y.copy(),
            upper=y.copy(),
            trunc_lower=lo,
            trunc_upper=hi,
        )
        ll_uncond = log_likelihood(theta, basis, y)
        log_p = self._manual_log_p(theta, basis, lo, hi)
        expected = ll_uncond - log_p.sum()
        ll = log_likelihood(theta, basis, cd, censoring=CensoringType.NONE)
        np.testing.assert_allclose(ll, expected, rtol=1e-10)

    def test_right_censoring_plus_truncation(self):
        """Right-censored survival data with delayed entry."""
        rng = np.random.default_rng(7)
        basis = make_basis(order=3)
        theta = ascending_theta(3, step=0.4)
        n = 12
        y = np.sort(rng.uniform(0.30, 0.95, n))
        is_censored = rng.random(n) < 0.4
        trunc_lo = y - rng.uniform(0.05, 0.20, n)
        cd = CensoredData.left_truncated(y, trunc_lo, censored=is_censored)
        # unconditional right-censored ll
        cd_uncond = CensoredData.right_censored(y, is_censored)
        ll_uncond = log_likelihood(
            theta, basis, cd_uncond, censoring=CensoringType.RIGHT
        )
        log_p = self._manual_log_p(theta, basis, trunc_lo, np.full(n, np.inf))
        expected = ll_uncond - log_p.sum()
        ll = log_likelihood(theta, basis, cd, censoring=CensoringType.RIGHT)
        np.testing.assert_allclose(ll, expected, rtol=1e-10)

    def test_interval_censoring_plus_truncation(self):
        """Interval-censored data with finite truncation window."""
        basis = make_basis(order=3)
        theta = ascending_theta(3, step=0.4)
        lo_obs = np.array([0.20, 0.45, 0.65])
        hi_obs = np.array([0.30, 0.55, 0.75])
        trunc_lo = np.array([0.10, 0.30, 0.55])
        trunc_hi = np.array([0.40, 0.65, 0.90])
        cd = CensoredData(
            exact=np.full(3, np.nan),
            lower=lo_obs,
            upper=hi_obs,
            trunc_lower=trunc_lo,
            trunc_upper=trunc_hi,
        )
        cd_uncond = CensoredData.interval_censored(lo_obs, hi_obs)
        ll_uncond = log_likelihood(
            theta, basis, cd_uncond, censoring=CensoringType.INTERVAL
        )
        log_p = self._manual_log_p(theta, basis, trunc_lo, trunc_hi)
        expected = ll_uncond - log_p.sum()
        ll = log_likelihood(theta, basis, cd, censoring=CensoringType.INTERVAL)
        np.testing.assert_allclose(ll, expected, rtol=1e-10)

    @pytest.mark.parametrize(
        "base_distribution",
        ["normal", "logistic", "min_extreme_value", "max_extreme_value"],
    )
    def test_gradient_check_grad(self, base_distribution):
        """check_grad < 1e-5 on negative_log_likelihood with truncation."""
        rng = np.random.default_rng(2)
        basis = make_basis(order=3, support=(0.0, 2.0))
        theta = ascending_theta(3, step=0.6)
        n = 14
        y = np.sort(rng.uniform(0.4, 1.6, n))
        is_censored = rng.random(n) < 0.3
        trunc_lo = y - rng.uniform(0.1, 0.3, n)
        cd = CensoredData.left_truncated(y, trunc_lo, censored=is_censored)

        def f(t):
            return negative_log_likelihood(
                t,
                basis,
                cd,
                None,
                CensoringType.RIGHT,
                base_distribution=base_distribution,
            )

        def g(t):
            _, grad = negative_log_likelihood(
                t,
                basis,
                cd,
                None,
                CensoringType.RIGHT,
                gradient=True,
                base_distribution=base_distribution,
            )
            return grad

        err = check_grad(f, g, theta)
        assert err < 1e-5, f"check_grad err={err:.2e} for {base_distribution}"

    def test_gradient_check_grad_with_X(self):
        """Gradient FD check, no-censoring + two-sided truncation, with X."""
        rng = np.random.default_rng(11)
        basis = make_basis(order=3)
        n, q = 12, 2
        y = np.sort(rng.uniform(0.3, 0.7, n))
        X = rng.standard_normal((n, q))
        theta = np.concatenate(
            [ascending_theta(3, step=0.3), 0.1 * rng.standard_normal(q)]
        )
        trunc_lo = y - rng.uniform(0.05, 0.15, n)
        trunc_hi = y + rng.uniform(0.05, 0.20, n)
        cd = CensoredData(
            exact=y.copy(),
            lower=y.copy(),
            upper=y.copy(),
            trunc_lower=trunc_lo,
            trunc_upper=trunc_hi,
        )

        def f(t):
            return negative_log_likelihood(t, basis, cd, X, CensoringType.NONE)

        def g(t):
            _, grad = negative_log_likelihood(
                t, basis, cd, X, CensoringType.NONE, gradient=True
            )
            return grad

        err = check_grad(f, g, theta)
        assert err < 1e-4, f"check_grad (with X) err={err:.2e}"

    def test_hessian_symmetry_and_finite_diff(self):
        """hessian(...) is symmetric and matches the FD of the analytic gradient."""
        from scipy.optimize import approx_fprime

        from pymlt.likelihood import hessian

        rng = np.random.default_rng(3)
        basis = make_basis(order=3)
        theta = ascending_theta(3, step=0.4)
        n = 10
        y = np.sort(rng.uniform(0.30, 0.85, n))
        trunc_lo = y - rng.uniform(0.05, 0.20, n)
        cd = CensoredData(
            exact=y.copy(),
            lower=y.copy(),
            upper=y.copy(),
            trunc_lower=trunc_lo,
        )
        H = hessian(theta, basis, cd, None, CensoringType.NONE)
        np.testing.assert_allclose(H, H.T, atol=1e-10)

        def grad_fn(t):
            _, g = negative_log_likelihood(
                t, basis, cd, None, CensoringType.NONE, gradient=True
            )
            return g

        H_fd = np.stack(
            [
                approx_fprime(theta, lambda t, i=i: grad_fn(t)[i], 1e-5)
                for i in range(len(theta))
            ]
        )
        H_fd = 0.5 * (H_fd + H_fd.T)
        np.testing.assert_allclose(H, H_fd, atol=5e-3, rtol=1e-3)

    def test_score_matrix_sums_to_minus_nll_gradient(self):
        """``score_matrix.sum(0) == −grad(NLL)`` under truncation."""
        from pymlt.likelihood import score_matrix

        rng = np.random.default_rng(5)
        basis = make_basis(order=3)
        theta = ascending_theta(3, step=0.4)
        n = 16
        y = np.sort(rng.uniform(0.30, 0.85, n))
        is_censored = rng.random(n) < 0.3
        trunc_lo = y - rng.uniform(0.05, 0.15, n)
        cd = CensoredData.left_truncated(y, trunc_lo, censored=is_censored)

        _, grad_nll = negative_log_likelihood(
            theta, basis, cd, None, CensoringType.RIGHT, gradient=True
        )
        scores = score_matrix(theta, basis, cd, None, CensoringType.RIGHT)
        np.testing.assert_allclose(scores.sum(axis=0), -grad_nll, atol=1e-10)

    def test_intercept_score_finite_difference(self):
        """intercept_score == d ℓ_i / d α at α=0, verified via finite differences."""
        from pymlt.likelihood import intercept_score

        rng = np.random.default_rng(9)
        basis = make_basis(order=3)
        theta = ascending_theta(3, step=0.4)
        n = 8
        y = np.sort(rng.uniform(0.30, 0.85, n))
        trunc_lo = y - rng.uniform(0.05, 0.15, n)
        cd = CensoredData(
            exact=y.copy(),
            lower=y.copy(),
            upper=y.copy(),
            trunc_lower=trunc_lo,
        )

        analytic = intercept_score(theta, basis, cd, None, CensoringType.NONE)

        eps = 1e-6
        offset_p = np.full(n, eps)
        offset_m = np.full(n, -eps)
        ll_p = log_likelihood(
            theta, basis, cd, None, CensoringType.NONE, offset=offset_p
        )
        ll_m = log_likelihood(
            theta, basis, cd, None, CensoringType.NONE, offset=offset_m
        )
        # Sum of intercept_score equals total dℓ/dα (offset shifts h uniformly)
        d_total_fd = (ll_p - ll_m) / (2 * eps)
        np.testing.assert_allclose(analytic.sum(), d_total_fd, rtol=1e-4, atol=1e-6)

    def test_reference_ll_trunc(self):
        """Pymlt LL matches mlt::logLik on left-truncated + right-censored data."""
        import pathlib

        ref_dir = pathlib.Path(__file__).parent.parent / "reference"
        required = [
            ref_dir / "ll_trunc_y.txt",
            ref_dir / "ll_trunc_event.txt",
            ref_dir / "ll_trunc_lower.txt",
            ref_dir / "ll_trunc_theta.txt",
            ref_dir / "ll_trunc_ll.txt",
        ]
        if not all(p.exists() for p in required):
            pytest.skip(
                "ll_trunc_* reference files not yet generated — "
                "run Rscript reference/generate_reference.R"
            )

        y = np.loadtxt(required[0])
        event = np.loadtxt(required[1]).astype(int)
        trunc_lo = np.loadtxt(required[2])
        theta = np.loadtxt(required[3])
        ll_ref = float(np.loadtxt(required[4]))

        is_censored = event == 0
        basis = BernsteinBasis(order=len(theta) - 1, support=(0.0, 1.0))
        cd = CensoredData.left_truncated(y, trunc_lo, censored=is_censored)
        ll = log_likelihood(theta, basis, cd, censoring=CensoringType.RIGHT)

        np.testing.assert_allclose(ll, ll_ref, rtol=1e-6, atol=1e-8)

    def test_weights_no_overflow_to_zero(self):
        # Regression: the both-finite branch of _truncation_weights used to
        # exp(logpdf - log_p) and then nan_to_num(posinf=0.0).  In a very
        # narrow truncation window logpdf - log_p exceeds log(float64.max),
        # exp(...) overflows to +inf, and the +inf -> 0 mapping silently
        # zeroed a genuinely large derivative weight, biasing gradient and
        # Hessian.  Construct a degenerate-window _TruncContext directly
        # (h_lo == h_hi forces _log_diff_ndtr -> -inf via its narrow-path
        # log(width=0) = -inf) so logpdf - log_p = +inf and exercises the
        # overflow branch.  After the fix the weight must be finite and
        # large, never zero.
        from pymlt.likelihood import (
            _LOG_FLOAT_MAX,
            _NORM_OPS,
            _truncation_weights,
            _TruncContext,
        )

        p_dim = 2
        h = np.array([0.0])
        B = np.ones((1, p_dim))
        empty_mask = np.array([False])
        ctx = _TruncContext(
            n=1,
            both=np.array([True]),
            only_hi=empty_mask,
            only_lo=empty_mask,
            B_lo_b=B,
            B_hi_b=B,
            h_lo_b=h,
            h_hi_b=h,
            X_b=None,
            B_hi_o=np.zeros((0, p_dim)),
            h_hi_o=np.zeros(0),
            X_only_hi=None,
            B_lo_o=np.zeros((0, p_dim)),
            h_lo_o=np.zeros(0),
            X_only_lo=None,
        )
        w_lo_b, w_hi_b, _, _ = _truncation_weights(ctx, _NORM_OPS)

        assert np.isfinite(w_hi_b[0]), f"w_hi_b must be finite, got {w_hi_b[0]}"
        assert np.isfinite(w_lo_b[0]), f"w_lo_b must be finite, got {w_lo_b[0]}"
        # Clipped at _LOG_FLOAT_MAX, the weight is ~exp(_LOG_FLOAT_MAX).
        # The previous buggy implementation returned exactly 0.0.
        threshold = np.exp(0.5 * _LOG_FLOAT_MAX)
        assert w_hi_b[0] > threshold, (
            f"narrow-window weight must be large, got {w_hi_b[0]} "
            "(buggy code silently returned 0.0)"
        )
        assert w_lo_b[0] > threshold


# ---------------------------------------------------------------------------
# Gradient correctness via scipy.optimize.check_grad
# ---------------------------------------------------------------------------


class TestGradients:
    def _check(self, basis, theta, y, censoring, *, X=None, atol=1e-5):
        cd_or_arr = y

        def f(t):
            return negative_log_likelihood(t, basis, cd_or_arr, X, censoring)

        def g(t):
            _, grad = negative_log_likelihood(
                t, basis, cd_or_arr, X, censoring, gradient=True
            )
            return grad

        err = check_grad(f, g, theta)
        assert err < atol, f"check_grad error = {err:.2e} for {censoring}"

    def test_gradient_none(self):
        basis = make_basis(order=4)
        theta = ascending_theta(4, step=0.4)
        y = np.linspace(0.1, 0.9, 10)
        self._check(basis, theta, y, CensoringType.NONE)

    def test_gradient_right(self):
        basis = make_basis(order=3)
        theta = ascending_theta(3, step=0.5)
        rng = np.random.default_rng(7)
        y = np.sort(rng.uniform(0.05, 0.95, 12))
        censored = rng.random(12) < 0.4
        cd = CensoredData.right_censored(y, censored)
        self._check(basis, theta, cd, CensoringType.RIGHT)

    def test_gradient_left(self):
        basis = make_basis(order=3)
        theta = ascending_theta(3, step=0.5)
        rng = np.random.default_rng(13)
        y = np.sort(rng.uniform(0.05, 0.95, 10))
        censored = rng.random(10) < 0.4
        cd = CensoredData.left_censored(y, censored)
        self._check(basis, theta, cd, CensoringType.LEFT)

    def test_gradient_interval(self):
        basis = make_basis(order=3)
        theta = ascending_theta(3, step=0.5)
        centers = np.linspace(0.2, 0.8, 8)
        cd = CensoredData.interval_censored(centers - 0.05, centers + 0.05)
        self._check(basis, theta, cd, CensoringType.INTERVAL)

    def test_gradient_none_with_regression(self):
        """Gradient with covariate X: includes beta part."""
        rng = np.random.default_rng(42)
        basis = make_basis(order=3)
        n, q = 12, 2
        y = np.sort(rng.uniform(0.1, 0.9, n))
        X = rng.standard_normal((n, q))
        theta_full = np.concatenate(
            [ascending_theta(3, step=0.4), rng.standard_normal(q)]
        )

        def f(t):
            return negative_log_likelihood(t, basis, y, X, CensoringType.NONE)

        def g(t):
            _, grad = negative_log_likelihood(
                t, basis, y, X, CensoringType.NONE, gradient=True
            )
            return grad

        err = check_grad(f, g, theta_full)
        assert err < 1e-4, f"check_grad (with X) error = {err:.2e}"

    def test_gradient_tuple_returned(self):
        basis = make_basis(order=3)
        theta = ascending_theta(3)
        y = np.array([0.3, 0.6])
        result = negative_log_likelihood(theta, basis, y, gradient=True)
        assert isinstance(result, tuple)
        nll, grad = result
        assert isinstance(nll, float)
        assert grad.shape == theta.shape

    def test_gradient_false_returns_float(self):
        basis = make_basis(order=3)
        theta = ascending_theta(3)
        y = np.array([0.3, 0.6])
        result = negative_log_likelihood(theta, basis, y, gradient=False)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# negative_log_likelihood wrapper
# ---------------------------------------------------------------------------


class TestNegativeLogLikelihood:
    def test_is_negation(self):
        basis = make_basis(order=3)
        theta = ascending_theta(3)
        y = np.array([0.2, 0.5, 0.8])
        ll = log_likelihood(theta, basis, y)
        nll = negative_log_likelihood(theta, basis, y)
        np.testing.assert_allclose(nll, -ll, rtol=1e-12)

    def test_positive_for_diffuse_data(self):
        """For well-separated data, NLL should be positive."""
        basis = make_basis(order=2)
        # theta maps [0,1] → [0, 2]: h(0.5)=1, h'=2
        theta = np.array([0.0, 1.0, 2.0])
        y = np.array([0.25, 0.5, 0.75])
        nll = negative_log_likelihood(theta, basis, y)
        assert nll > 0, f"Expected NLL > 0, got {nll}"


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(
    order=st.integers(2, 8),
    seed=st.integers(0, 2**31 - 1),
    n=st.integers(3, 20),
)
@settings(max_examples=150)
def test_ll_finite_for_valid_theta(order, seed, n):
    """For ascending theta and y in support, LL is always finite."""
    rng = np.random.default_rng(seed)
    basis = BernsteinBasis(order=order, support=(0.0, 1.0))
    theta = np.cumsum(rng.uniform(0.1, 1.0, size=order + 1))
    y = rng.uniform(0.01, 0.99, n)
    result = log_likelihood(theta, basis, y)
    assert np.isfinite(result), f"Expected finite LL, got {result}"


@given(
    order=st.integers(2, 6),
    seed=st.integers(0, 2**31 - 1),
    n=st.integers(3, 15),
)
@settings(max_examples=150)
def test_nll_equals_negation_of_ll(order, seed, n):
    """negative_log_likelihood == -log_likelihood always."""
    rng = np.random.default_rng(seed)
    basis = BernsteinBasis(order=order, support=(0.0, 1.0))
    theta = np.cumsum(rng.uniform(0.1, 0.5, size=order + 1))
    y = rng.uniform(0.01, 0.99, n)
    ll = log_likelihood(theta, basis, y)
    nll = negative_log_likelihood(theta, basis, y)
    np.testing.assert_allclose(nll, -ll, rtol=1e-12)


@given(
    order=st.integers(2, 6),
    seed=st.integers(0, 2**31 - 1),
    n=st.integers(4, 15),
    frac=st.floats(0.1, 0.9),
)
@settings(max_examples=100)
def test_ll_right_finite_for_valid_data(order, seed, n, frac):
    """RIGHT log-likelihood is finite for valid ascending theta and mixed data."""
    rng = np.random.default_rng(seed)
    basis = BernsteinBasis(order=order, support=(0.0, 1.0))
    theta = np.cumsum(rng.uniform(0.1, 0.5, size=order + 1))
    y = rng.uniform(0.01, 0.99, n)
    censored = rng.random(n) < frac
    cd = CensoredData.right_censored(y, censored)
    result = log_likelihood(theta, basis, cd, censoring=CensoringType.RIGHT)
    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# base_distribution validation
# ---------------------------------------------------------------------------


class TestGetDist:
    def test_normal_returns_norm(self):
        from scipy.stats import norm

        ops = _get_dist("normal")
        assert ops.kind == "normal"
        assert ops.scipy is norm

    def test_logistic_returns_logistic(self):
        from scipy.stats import logistic

        ops = _get_dist("logistic")
        assert ops.kind == "logistic"
        assert ops.scipy is logistic

    def test_min_extreme_value_returns_gumbel_l(self):
        from scipy.stats import gumbel_l

        ops = _get_dist("min_extreme_value")
        assert ops.kind == "min_extreme_value"
        assert ops.scipy is gumbel_l

    def test_max_extreme_value_returns_gumbel_r(self):
        from scipy.stats import gumbel_r

        ops = _get_dist("max_extreme_value")
        assert ops.kind == "max_extreme_value"
        assert ops.scipy is gumbel_r

    def test_exponential_returns_expon(self):
        from scipy.stats import expon

        ops = _get_dist("exponential")
        assert ops.kind == "exponential"
        assert ops.scipy is expon

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="base_distribution"):
            _get_dist("student-t")

    def test_laplace_returns_laplace(self):
        from scipy.stats import laplace

        ops = _get_dist("laplace")
        assert ops.kind == "laplace"
        assert ops.scipy is laplace

    def test_cauchy_returns_cauchy(self):
        from scipy.stats import cauchy

        ops = _get_dist("cauchy")
        assert ops.kind == "cauchy"
        assert ops.scipy is cauchy

    def test_error_message_contains_valid_options(self):
        with pytest.raises(ValueError, match="normal"):
            _get_dist("student-t")

    @pytest.mark.parametrize("bad", ["Normal", "NORMAL", "gauss", "", "t", "uniform"])
    def test_case_sensitive_and_rejects_aliases(self, bad):
        with pytest.raises(ValueError, match="base_distribution"):
            _get_dist(bad)

    def test_valid_distributions_constant(self):
        assert set(_VALID_BASE_DISTRIBUTIONS) == {
            "normal",
            "logistic",
            "min_extreme_value",
            "max_extreme_value",
            "exponential",
            "laplace",
            "cauchy",
        }


def test_log_likelihood_invalid_distribution_raises():
    basis = make_basis()
    theta = ascending_theta(basis.order)
    y = np.linspace(0.1, 0.9, 20)
    with pytest.raises(ValueError, match="base_distribution"):
        log_likelihood(theta, basis, y, base_distribution="student-t")


def test_negative_log_likelihood_invalid_distribution_raises():
    basis = make_basis()
    theta = ascending_theta(basis.order)
    y = np.linspace(0.1, 0.9, 20)
    with pytest.raises(ValueError, match="base_distribution"):
        negative_log_likelihood(theta, basis, y, base_distribution="student-t")


# ---------------------------------------------------------------------------
# _neg_score — analytical formulae per base distribution
# ---------------------------------------------------------------------------


class TestNegScore:
    """Verify _neg_score(h, dist) matches -∂ log f(h)/∂h for each distribution.

    Reference values are derived from the closed-form score of each density.
    A finite-difference check on ``dist.logpdf`` confirms the formula.
    """

    @staticmethod
    def _fd_neg_score(dist, h, eps=1e-6):
        # -(d/dh log f(h)) via central differences on logpdf.
        return -(dist.logpdf(h + eps) - dist.logpdf(h - eps)) / (2 * eps)

    def test_normal(self):
        from pymlt.likelihood import _NORM_OPS, _neg_score

        h = np.linspace(-2.0, 2.0, 9)
        np.testing.assert_allclose(_neg_score(h, _NORM_OPS), h, rtol=1e-12)
        np.testing.assert_allclose(
            _neg_score(h, _NORM_OPS), self._fd_neg_score(_NORM_OPS, h), rtol=1e-4
        )

    def test_min_extreme_value(self):
        from pymlt.likelihood import _MEV_OPS, _neg_score

        h = np.linspace(-1.5, 1.5, 9)
        expected = np.exp(h) - 1.0
        np.testing.assert_allclose(_neg_score(h, _MEV_OPS), expected, rtol=1e-12)
        np.testing.assert_allclose(
            _neg_score(h, _MEV_OPS), self._fd_neg_score(_MEV_OPS, h), rtol=1e-4
        )

    def test_max_extreme_value(self):
        from pymlt.likelihood import _MAXEV_OPS, _neg_score

        h = np.linspace(-1.5, 1.5, 9)
        expected = 1.0 - np.exp(-h)
        np.testing.assert_allclose(_neg_score(h, _MAXEV_OPS), expected, rtol=1e-12)
        np.testing.assert_allclose(
            _neg_score(h, _MAXEV_OPS), self._fd_neg_score(_MAXEV_OPS, h), rtol=1e-4
        )

    def test_exponential(self):
        from pymlt.likelihood import _EXPON_OPS, _neg_score

        h = np.linspace(0.1, 3.0, 9)  # strictly > 0: in support
        expected = np.ones_like(h)
        np.testing.assert_allclose(_neg_score(h, _EXPON_OPS), expected, rtol=1e-12)
        np.testing.assert_allclose(
            _neg_score(h, _EXPON_OPS), self._fd_neg_score(_EXPON_OPS, h), rtol=1e-4
        )

    def test_logistic(self):
        from pymlt.likelihood import _LOGIS_OPS, _neg_score

        h = np.linspace(-2.0, 2.0, 9)
        expected = 2.0 * _LOGIS_OPS.cdf(h) - 1.0
        np.testing.assert_allclose(_neg_score(h, _LOGIS_OPS), expected, rtol=1e-12)
        np.testing.assert_allclose(
            _neg_score(h, _LOGIS_OPS), self._fd_neg_score(_LOGIS_OPS, h), rtol=1e-4
        )

    def test_laplace(self):
        from pymlt.likelihood import _LAPLACE_OPS, _neg_score

        h = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        expected = np.sign(h)
        np.testing.assert_allclose(_neg_score(h, _LAPLACE_OPS), expected, rtol=1e-12)
        np.testing.assert_allclose(
            _neg_score(h, _LAPLACE_OPS), self._fd_neg_score(_LAPLACE_OPS, h), rtol=1e-4
        )

    def test_cauchy(self):
        from pymlt.likelihood import _CAUCHY_OPS, _neg_score

        h = np.linspace(-2.0, 2.0, 9)
        expected = 2.0 * h / (1.0 + h**2)
        np.testing.assert_allclose(_neg_score(h, _CAUCHY_OPS), expected, rtol=1e-12)
        np.testing.assert_allclose(
            _neg_score(h, _CAUCHY_OPS), self._fd_neg_score(_CAUCHY_OPS, h), rtol=1e-4
        )

    def test_unhandled_kind_raises_neg_score(self):
        """Exhaustiveness guard: an unknown kind fails loudly, no logistic fallthrough."""
        from pymlt.likelihood import DistOps, _neg_score

        bogus = DistOps(kind="not-a-real-distribution", scipy=None)  # type: ignore[arg-type]

        with pytest.raises(AssertionError, match="unhandled dist.kind"):
            _neg_score(np.array([0.0, 1.0]), bogus)

    def test_unhandled_kind_raises_d2_logpdf(self):
        """Same guard in _d2_logpdf — the other correctness-critical dispatch."""
        from pymlt.likelihood import DistOps, _d2_logpdf

        bogus = DistOps(kind="not-a-real-distribution", scipy=None)  # type: ignore[arg-type]

        with pytest.raises(AssertionError, match="unhandled dist.kind"):
            _d2_logpdf(np.array([0.0, 1.0]), bogus)


# ---------------------------------------------------------------------------
# Per-distribution log-likelihood + gradient checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_distribution",
    [
        "normal",
        "logistic",
        "min_extreme_value",
        "max_extreme_value",
        "exponential",
        "laplace",
        "cauchy",
    ],
)
class TestPerDistributionLikelihood:
    """End-to-end per-distribution coverage of log_likelihood and its gradient."""

    def _data(self, base_distribution):
        """Return (basis, theta, y) feasible for the given base distribution.

        For ``exponential`` we need h(y) >= 0, so start theta_b at 0.
        """
        basis = make_basis(order=4)
        theta = ascending_theta(4, step=0.4)  # theta_b[0] = 0, feasible for all
        y = np.linspace(0.1, 0.9, 12)
        return basis, theta, y

    def test_finite(self, base_distribution):
        basis, theta, y = self._data(base_distribution)
        ll = log_likelihood(theta, basis, y, base_distribution=base_distribution)
        assert np.isfinite(ll)

    def test_manual_matches_scipy(self, base_distribution):
        """LL = Σ dist.logpdf(h) + Σ log(h') computed directly."""
        basis, theta, y = self._data(base_distribution)
        dist = _get_dist(base_distribution)
        B = basis.evaluate(y)
        D = basis.derivative(y, order=1)
        h = B @ theta
        hp = D @ theta
        expected = float(np.sum(dist.logpdf(h)) + np.sum(np.log(hp)))
        result = log_likelihood(theta, basis, y, base_distribution=base_distribution)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_gradient_matches_fd(self, base_distribution):
        basis, theta, y = self._data(base_distribution)

        def f(t):
            return negative_log_likelihood(
                t,
                basis,
                y,
                None,
                CensoringType.NONE,
                base_distribution=base_distribution,
            )

        def g(t):
            _, grad = negative_log_likelihood(
                t,
                basis,
                y,
                None,
                CensoringType.NONE,
                gradient=True,
                base_distribution=base_distribution,
            )
            return grad

        err = check_grad(f, g, theta)
        assert err < 1e-4, f"check_grad err={err:.2e} for {base_distribution}"

    def test_gradient_right_matches_fd(self, base_distribution):
        basis, theta, _ = self._data(base_distribution)
        rng = np.random.default_rng(11)
        y = np.sort(rng.uniform(0.05, 0.95, 12))
        censored = rng.random(12) < 0.4
        cd = CensoredData.right_censored(y, censored)

        def f(t):
            return negative_log_likelihood(
                t,
                basis,
                cd,
                None,
                CensoringType.RIGHT,
                base_distribution=base_distribution,
            )

        def g(t):
            _, grad = negative_log_likelihood(
                t,
                basis,
                cd,
                None,
                CensoringType.RIGHT,
                gradient=True,
                base_distribution=base_distribution,
            )
            return grad

        err = check_grad(f, g, theta)
        assert err < 1e-4, f"check_grad err={err:.2e} for {base_distribution}"


# ---------------------------------------------------------------------------
# Regression: DistOps dispatches by kind string, not scipy object identity.
# ---------------------------------------------------------------------------


class _ScipyProxy:
    """Minimal passthrough around a scipy.stats distribution.

    A fresh proxy object has a different ``id()`` than ``scipy.stats.norm``
    etc., so any code that dispatches by ``dist is norm`` (or similar identity
    check) would silently take the wrong branch when handed one of these.
    The proxy is there to simulate the real-world scenarios that prompted the
    refactor: scipy reimports in sub-interpreters, pickle round-trips, and
    test wrappers.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.mark.parametrize(
    "base_distribution",
    ["normal", "logistic", "min_extreme_value", "max_extreme_value", "exponential"],
)
class TestDistOpsDispatchIsIdentityFree:
    """A DistOps wrapping a proxy must behave identically to the canonical one.

    This is the regression guard for the fragile ``dist is norm`` pattern that
    existed prior to DistOps.  If a future change reintroduces identity-based
    dispatch on the scipy object, these tests will fail — either with wrong
    numerics in ``_neg_score`` / ``_d2_logpdf`` (the silent correctness bug)
    or with a reduced-precision fallback in the ``log_ndtr`` path.
    """

    def _make_ops(self, base_distribution):
        from pymlt.likelihood import DistOps, _get_dist

        canonical = _get_dist(base_distribution)
        proxy = _ScipyProxy(canonical.scipy)
        wrapped = DistOps(kind=canonical.kind, scipy=proxy)
        # Sanity: identity against the module-level singleton is broken on purpose.
        assert wrapped.scipy is not canonical.scipy
        return canonical, wrapped

    @staticmethod
    def _h_grid(base_distribution):
        # exponential has support [0, inf); stay strictly > 0.
        if base_distribution == "exponential":
            return np.linspace(0.1, 2.5, 9)
        return np.linspace(-2.0, 2.0, 9)

    def test_neg_score_matches(self, base_distribution):
        from pymlt.likelihood import _neg_score

        canonical, wrapped = self._make_ops(base_distribution)
        h = self._h_grid(base_distribution)
        np.testing.assert_array_equal(_neg_score(h, wrapped), _neg_score(h, canonical))

    def test_d2_logpdf_matches(self, base_distribution):
        from pymlt.likelihood import _d2_logpdf

        canonical, wrapped = self._make_ops(base_distribution)
        h = self._h_grid(base_distribution)
        np.testing.assert_array_equal(_d2_logpdf(h, wrapped), _d2_logpdf(h, canonical))

    def test_log_likelihood_none_matches(self, base_distribution):
        from pymlt.likelihood import _log_likelihood_from_dist

        canonical, wrapped = self._make_ops(base_distribution)
        basis = make_basis(order=4)
        theta = ascending_theta(4, step=0.4)
        y = np.linspace(0.1, 0.9, 12)
        ref = _log_likelihood_from_dist(
            theta, basis, y, None, CensoringType.NONE, canonical
        )
        got = _log_likelihood_from_dist(
            theta, basis, y, None, CensoringType.NONE, wrapped
        )
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)

    def test_nll_with_gradient_matches(self, base_distribution):
        from pymlt.likelihood import _negative_log_likelihood_from_dist

        canonical, wrapped = self._make_ops(base_distribution)
        basis = make_basis(order=4)
        theta = ascending_theta(4, step=0.4)
        y = np.linspace(0.1, 0.9, 12)

        nll_ref, grad_ref = _negative_log_likelihood_from_dist(
            theta, basis, y, None, CensoringType.NONE, True, canonical
        )
        nll_w, grad_w = _negative_log_likelihood_from_dist(
            theta, basis, y, None, CensoringType.NONE, True, wrapped
        )
        np.testing.assert_allclose(nll_w, nll_ref, rtol=1e-12, atol=0.0)
        np.testing.assert_array_equal(grad_w, grad_ref)

    def test_log_likelihood_left_censored_matches(self, base_distribution):
        """Covers the ``log_ndtr if dist.kind == 'normal'`` fast-path sites."""
        from pymlt.likelihood import _log_likelihood_from_dist

        canonical, wrapped = self._make_ops(base_distribution)
        basis = make_basis(order=4)
        theta = ascending_theta(4, step=0.4)
        rng = np.random.default_rng(7)
        y = np.sort(rng.uniform(0.1, 0.9, 12))
        censored = rng.random(12) < 0.4
        cd = CensoredData.left_censored(y, censored)

        ref = _log_likelihood_from_dist(
            theta, basis, cd, None, CensoringType.LEFT, canonical
        )
        got = _log_likelihood_from_dist(
            theta, basis, cd, None, CensoringType.LEFT, wrapped
        )
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)
