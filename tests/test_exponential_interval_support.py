"""Exponential link: both-finite interval rows below the support (issue #164).

Under ``base_distribution="exponential"`` the base CDF is R's ``pexp``::

    F(h) = 1 − e^{−h}   (h > 0),        F(h) = 0   (h ≤ 0).

A both-finite interval row ``(l, u]`` with ``h_l = h(l|x_i)`` and
``h_u = h(u|x_i)`` therefore scores::

    log P = log(F(h_u) − F(h_l))     if h_l ≥ 0,
    log P = log F(h_u)               if h_l < 0 < h_u,
    log P = −inf                     if h_u ≤ 0.

The value is continuous at ``h_l = 0`` (``F(0) = 0``).  Below the support
``∂ log P / ∂h_l = 0``, so the lower-endpoint Mills weight ``f(h_l)/P`` is 0
there; at ``h_l = 0`` it keeps the feasible-side value ``1/P`` (R's
``dexp(0) = 1``).  Exact rows keep the smooth extension ``log f = −h`` on the
whole line (ADR 0005, #114).  ADR 0005 records the decision in its #164
amendment.

R reference (``reference/mlt_exponential_interval_out_*.txt``, generated in
``reference/generate_reference.R``): an unfitted ``mlt(ctm(Bernstein, shifting
= ~x_d, scaling = ~x_s, todistr = "Exponential"))`` evaluated at a θ with
``θ_b[0] < 0``.  Its per-row ``logliki`` and ``scorei`` cover all three row
classes.  The raw ``ctm`` coefficient order is mltpy's ``[θ_b | β | γ]``
(``ctm``'s shift sign is positive, as in mltpy's likelihood).
"""

from __future__ import annotations

import pathlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from mltpy.basis import BernsteinBasis
from mltpy.likelihood import (
    InfeasibleParameterError,
    _get_dist,
    _ll_interval,
    _log_diff_ndtr,
    _pair_density_weights,
    hessian,
    log_likelihood,
    negative_log_likelihood,
    score_matrix,
)
from mltpy.variables import CensoredData, CensoringType

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"
_EXP = _get_dist("exponential")
_ORDER = 4
_P = _ORDER + 1

# CLAUDE.md's R-comparison tolerance.
_R_TOL = {"rtol": 1e-6, "atol": 1e-10}
# Central differences (step 1e-6) against the analytic derivatives.
_FD_STEP = 1e-6
_FD_TOL = {"rtol": 1e-5, "atol": 1e-6}


def _central_gradient(
    fun: Callable[[NDArray[np.float64]], float],
    theta: NDArray[np.float64],
    eps: float = _FD_STEP,
) -> NDArray[np.float64]:
    """Central finite-difference gradient (test reference only)."""
    out = np.empty_like(theta)
    for j in range(theta.size):
        e = np.zeros_like(theta)
        e[j] = eps
        out[j] = (fun(theta + e) - fun(theta - e)) / (2.0 * eps)
    return out


def _central_jacobian(
    fun: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    theta: NDArray[np.float64],
    eps: float = _FD_STEP,
) -> NDArray[np.float64]:
    """Central finite-difference Jacobian of a vector function (columns = θ_j)."""
    cols = []
    for j in range(theta.size):
        e = np.zeros_like(theta)
        e[j] = eps
        cols.append((fun(theta + e) - fun(theta - e)) / (2.0 * eps))
    return np.column_stack(cols)


def _pexp_log_p(
    h_lo: NDArray[np.float64], h_hi: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``log(pexp(h_hi) − pexp(h_lo))`` written out with ``F(h ≤ 0) = 0``."""
    with np.errstate(divide="ignore"):
        diff = np.exp(-np.maximum(h_lo, 0.0)) - np.exp(-np.maximum(h_hi, 0.0))
        return np.asarray(np.log(diff), dtype=np.float64)


def _interval_h(
    basis: BernsteinBasis,
    cd: CensoredData,
    theta_b: NDArray[np.float64],
    f: NDArray[np.float64],
    shift: NDArray[np.float64] | float = 0.0,
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """``(both, h_lo, h_hi)`` written out: ``h = h_0(y)·f + shift``.

    ``both`` marks the both-finite rows; an open upper end is evaluated at the
    support's upper end (its ``h_hi`` is unused).
    """
    both = np.isfinite(cd.upper)
    up = np.where(both, cd.upper, basis.support[1])
    h_lo = basis.evaluate(cd.lower) @ theta_b * f + shift
    h_hi = basis.evaluate(up) @ theta_b * f + shift
    return both, h_lo, h_hi


# ---------------------------------------------------------------------------
# The two helpers every both-finite interval block shares
# ---------------------------------------------------------------------------


class TestLogDiffNdtrBelowSupport:
    """``_log_diff_ndtr`` with the exponential ``pexp`` semantics."""

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            # h_l < 0 < h_u: F(h_l) = 0, so log P = log F(h_u).
            (-0.5, 1.0, np.log(-np.expm1(-1.0))),
            (-1e-12, 1e-3, np.log(-np.expm1(-1e-3))),
            (-30.0, 30.0, np.log(-np.expm1(-30.0))),
            (-2.0, 1e-300, np.log(1e-300)),
            # h_u ≤ 0: P = 0.
            (-1.0, -0.2, -np.inf),
            (-1.0, 0.0, -np.inf),
            (-2.0, -2.0, -np.inf),
            (-1e-12, -1e-13, -np.inf),
            # In the support: unchanged.
            (0.0, 0.0, -np.inf),
            (0.0, 1.0, np.log(-np.expm1(-1.0))),
            (0.5, 1.0, np.log(np.exp(-0.5) - np.exp(-1.0))),
        ],
    )
    def test_value(self, a: float, b: float, expected: float) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = _log_diff_ndtr(np.array([a]), np.array([b]), dist=_EXP)
        np.testing.assert_allclose(out, [expected], rtol=1e-12)

    def test_vectorised_mix(self) -> None:
        """All three classes in one call match the written-out ``pexp`` form."""
        a = np.array([0.3, -0.3, -0.3, 0.0, -1.0, 1.0])
        b = np.array([0.9, 0.9, -0.1, 0.4, 0.0, 1.0 + 1e-9])
        out = _log_diff_ndtr(a, b, dist=_EXP)
        expected = _pexp_log_p(a[:5], b[:5])
        np.testing.assert_allclose(out[:5], expected, rtol=1e-12)
        # Narrow in-support interval: Taylor branch, unchanged.
        np.testing.assert_allclose(out[5], -1.0 + np.log(1e-9), rtol=1e-6)

    def test_continuous_at_lower_support_end(self) -> None:
        """``h_l → 0`` from either side gives the same ``log F(h_u)``."""
        a = np.array([-1e-10, 0.0, 1e-10])
        out = _log_diff_ndtr(a, np.full(3, 0.7), dist=_EXP)
        np.testing.assert_allclose(out, np.log(-np.expm1(-0.7)), rtol=1e-9)

    def test_nan_propagates(self) -> None:
        out = _log_diff_ndtr(
            np.array([np.nan, -1.0]), np.array([1.0, np.nan]), dist=_EXP
        )
        assert np.all(np.isnan(out))

    def test_empty(self) -> None:
        out = _log_diff_ndtr(np.zeros(0), np.zeros(0), dist=_EXP)
        assert out.shape == (0,)

    @pytest.mark.parametrize("kind", ["normal", "logistic", "min_extreme_value"])
    def test_other_links_untouched(self, kind: str) -> None:
        """Negative ``h`` is inside every other link's support."""
        dist = _get_dist(kind)
        a, b = np.array([-0.5]), np.array([1.0])
        expected = np.log(dist.cdf(b) - dist.cdf(a))
        np.testing.assert_allclose(_log_diff_ndtr(a, b, dist=dist), expected, 1e-12)


class TestPairDensityWeightsBelowSupport:
    """Mills weights ``(f(h_u)/P, f(h_l)/P)`` with ``f = dexp``."""

    def test_weights(self) -> None:
        h_lo = np.array([0.3, -0.3, 0.0, -1.0])
        h_hi = np.array([0.9, 0.9, 0.9, -0.2])
        log_p = _log_diff_ndtr(h_lo, h_hi, dist=_EXP)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            w_hi, w_lo = _pair_density_weights(h_lo, h_hi, log_p, _EXP)
        p_in = np.exp(-0.3) - np.exp(-0.9)
        f_u = -np.expm1(-0.9)
        np.testing.assert_allclose(
            w_hi, [np.exp(-0.9) / p_in, np.exp(-0.9) / f_u, np.exp(-0.9) / f_u, 0.0]
        )
        # Lower weight: in support / zero below / feasible side at h_l = 0
        # (R's dexp(0) = 1) / zero (P = 0, ℓ = −inf raises anyway).
        np.testing.assert_allclose(w_lo, [np.exp(-0.3) / p_in, 0.0, 1.0 / f_u, 0.0])

    @pytest.mark.parametrize(
        ("h_lo", "feasible_side"),
        [(-1e-12, True), (-0.99e-7, True), (-1.01e-7, False), (-1e-3, False)],
    )
    def test_support_tolerance_band(self, h_lo: float, feasible_side: bool) -> None:
        """Within ``_EXPON_SUPPORT_TOL`` of 0 the endpoint counts as on the support.

        The value is ``pexp`` throughout; only the weight takes the
        feasible-side derivative ``e^{−h}/P`` there.
        """
        lo, hi = np.array([h_lo]), np.array([0.9])
        log_p = _log_diff_ndtr(lo, hi, dist=_EXP)
        np.testing.assert_allclose(log_p, np.log(-np.expm1(-0.9)), rtol=1e-12)
        _, w_lo = _pair_density_weights(lo, hi, log_p, _EXP)
        expected = np.exp(-h_lo) / -np.expm1(-0.9) if feasible_side else 0.0
        np.testing.assert_allclose(w_lo, [expected], rtol=1e-12)

    def test_normal_lower_weight_kept(self) -> None:
        dist = _get_dist("normal")
        h_lo, h_hi = np.array([-0.3]), np.array([0.9])
        log_p = _log_diff_ndtr(h_lo, h_hi, dist=dist)
        _, w_lo = _pair_density_weights(h_lo, h_hi, log_p, dist)
        np.testing.assert_allclose(w_lo, dist.pdf(h_lo) / np.exp(log_p), 1e-12)


# ---------------------------------------------------------------------------
# R parity at a θ outside the support
# ---------------------------------------------------------------------------


def _load(name: str) -> NDArray[np.float64]:
    path = REF_DIR / f"mlt_exponential_interval_out_{name}.txt"
    return np.loadtxt(path, dtype=np.float64)


@pytest.fixture(scope="module")
def ref() -> dict[str, NDArray[np.float64]]:
    names = ("lower", "upper", "xd", "xs", "support", "theta", "logliki", "scorei")
    return {name: _load(name) for name in names}


def _ref_problem(
    ref: dict[str, NDArray[np.float64]], rows: NDArray[np.bool_] | None = None
) -> tuple[BernsteinBasis, CensoredData, NDArray[np.float64], NDArray[np.float64]]:
    keep = np.ones(ref["lower"].size, bool) if rows is None else rows
    basis = BernsteinBasis(order=_ORDER, support=tuple(ref["support"]))
    cd = CensoredData.interval_censored(ref["lower"][keep], ref["upper"][keep])
    return basis, cd, ref["xd"][keep, None], ref["xs"][keep, None]


class TestRParity:
    """mltpy against ``mlt``'s ``logliki`` / ``scorei`` at an infeasible θ."""

    def test_fixture_covers_every_row_class(
        self, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        basis, cd, xd, xs = _ref_problem(ref)
        theta = ref["theta"]
        f = np.exp(0.5 * xs[:, 0] * theta[_P + 1])
        both, h_lo, h_hi = _interval_h(basis, cd, theta[:_P], f, xd[:, 0] * theta[_P])
        assert np.sum(both & (h_lo >= 0)) >= 3
        assert np.sum(both & (h_lo < 0) & (h_hi > 0)) >= 3
        assert np.sum(both & (h_hi <= 0)) >= 3
        np.testing.assert_array_equal(np.isneginf(ref["logliki"]), both & (h_hi <= 0))

    def test_per_row_loglik(self, ref: dict[str, NDArray[np.float64]]) -> None:
        """Every row's ``log P``, ``−inf`` rows included."""
        basis, cd, xd, xs = _ref_problem(ref)
        out = np.array(
            [
                _ll_interval(
                    CensoredData.interval_censored(cd.lower[[i]], cd.upper[[i]]),
                    ref["theta"],
                    basis,
                    xd[[i]],
                    dist=_EXP,
                    scaling=xs[[i]],
                )
                for i in range(cd.n)
            ]
        )
        np.testing.assert_allclose(out, ref["logliki"], **_R_TOL)

    def test_loglik_of_finite_rows(self, ref: dict[str, NDArray[np.float64]]) -> None:
        finite = np.isfinite(ref["logliki"])
        basis, cd, xd, xs = _ref_problem(ref, finite)
        ll = log_likelihood(
            ref["theta"],
            basis,
            cd,
            xd,
            CensoringType.INTERVAL,
            base_distribution="exponential",
            scaling=xs,
        )
        np.testing.assert_allclose(ll, ref["logliki"][finite].sum(), **_R_TOL)

    def test_neg_inf_rows_raise(self, ref: dict[str, NDArray[np.float64]]) -> None:
        basis, cd, xd, xs = _ref_problem(ref)
        with pytest.raises(InfeasibleParameterError):
            log_likelihood(
                ref["theta"],
                basis,
                cd,
                xd,
                CensoringType.INTERVAL,
                base_distribution="exponential",
                scaling=xs,
            )

    def test_score(self, ref: dict[str, NDArray[np.float64]]) -> None:
        """``score_matrix`` rows equal R's ``scorei``; the gradient their sum."""
        finite = np.isfinite(ref["logliki"])
        basis, cd, xd, xs = _ref_problem(ref, finite)
        problem = _Problem(basis, cd, xd, xs)
        np.testing.assert_allclose(
            problem.scores(ref["theta"]), ref["scorei"], **_R_TOL
        )
        np.testing.assert_allclose(
            -problem.grad(ref["theta"]), ref["scorei"].sum(axis=0), **_R_TOL
        )


# ---------------------------------------------------------------------------
# Analytic derivatives on both sides of h_l = 0
# ---------------------------------------------------------------------------


def _simulated(seed: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The #161 / #163 design: Exp(rate 2·exp(0.3·x_s)) responses, n = 150."""
    rng = np.random.default_rng(seed)
    x_s = rng.normal(size=(150, 1))
    y = rng.exponential(size=150) / (2.0 * np.exp(0.5 * 0.6 * x_s[:, 0]))
    return y, x_s


def _inspection_grid(
    y: NDArray[np.float64], q: float
) -> tuple[CensoredData, tuple[float, float]]:
    """Four common inspection times; first interval (0, t_1], last (t_4, ∞)."""
    grid = np.quantile(y, np.linspace(0.0, q, 5)[1:])
    k = np.searchsorted(grid, y)
    lower = np.concatenate([[0.0], grid])[k]
    upper = np.concatenate([grid, [np.inf]])[k]
    return CensoredData.interval_censored(lower, upper), (0.0, 1.01 * float(grid[-1]))


@dataclass(frozen=True)
class _Problem:
    """An exponential-link likelihood problem: value, gradient, Hessian, scores."""

    basis: BernsteinBasis
    cd: CensoredData
    X: NDArray[np.float64] | None
    scaling: NDArray[np.float64] | None
    censoring: CensoringType = CensoringType.INTERVAL

    def nll(self, theta: NDArray[np.float64]) -> float:
        out = negative_log_likelihood(
            theta,
            self.basis,
            self.cd,
            self.X,
            self.censoring,
            base_distribution="exponential",
            scaling=self.scaling,
        )
        assert isinstance(out, float)
        return out

    def grad(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        out = negative_log_likelihood(
            theta,
            self.basis,
            self.cd,
            self.X,
            self.censoring,
            gradient=True,
            base_distribution="exponential",
            scaling=self.scaling,
        )
        assert isinstance(out, tuple)
        return out[1]

    def hess(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return hessian(
            theta,
            self.basis,
            self.cd,
            self.X,
            self.censoring,
            base_distribution="exponential",
            scaling=self.scaling,
        )

    def scores(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return score_matrix(
            theta,
            self.basis,
            self.cd,
            self.X,
            self.censoring,
            base_distribution="exponential",
            scaling=self.scaling,
        )

    def check_derivatives(self, theta: NDArray[np.float64]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            grad = self.grad(theta)
            hess = self.hess(theta)
            scores = self.scores(theta)
        np.testing.assert_allclose(grad, _central_gradient(self.nll, theta), **_FD_TOL)
        np.testing.assert_allclose(hess, _central_jacobian(self.grad, theta), **_FD_TOL)
        np.testing.assert_allclose(hess, hess.T, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scores.sum(axis=0), -grad, rtol=1e-12, atol=1e-12)


@pytest.fixture(scope="module")
def grid_problem() -> _Problem:
    """Seed 0, ``q = 0.8``: the #163 evidence design (scale only)."""
    y, x_s = _simulated(0)
    cd, support = _inspection_grid(y, 0.8)
    return _Problem(BernsteinBasis(order=_ORDER, support=support), cd, None, x_s)


class TestDerivativesBothSides:
    """Gradient, Hessian and scores against central differences."""

    @pytest.mark.parametrize("theta_b0", [-0.5, -0.3, -1e-4, 1e-4, 0.3])
    def test_inspection_grid(self, grid_problem: _Problem, theta_b0: float) -> None:
        """The ``(0, t_1]`` rows sit at ``h_l = θ_b[0] · f_i``: every sign."""
        theta = np.array([theta_b0, 0.5, 1.0, 1.5, 2.0, 0.3])
        grid_problem.check_derivatives(theta)

    def test_feasible_side_within_support_tolerance(
        self, grid_problem: _Problem
    ) -> None:
        """At ``θ_b[0] = ∓1e-10`` (an active row met to round-off) the
        gradient and scores agree: both use the feasible-side derivative.

        SLSQP ends at such points (``θ_b[0] ≈ −1e-14 … −5e-10``); with ``dexp``'s
        zero weight there it stopped early on 3 of the 9 inspection-grid
        designs, and the score / Hessian at its fit would drop every lower
        endpoint of a ``(0, t]`` row.
        """
        theta_lo = np.array([-1e-10, 0.5, 1.0, 1.5, 2.0, 0.3])
        theta_hi = np.array([1e-10, 0.5, 1.0, 1.5, 2.0, 0.3])
        np.testing.assert_allclose(
            grid_problem.grad(theta_lo), grid_problem.grad(theta_hi), rtol=1e-8
        )
        np.testing.assert_allclose(
            grid_problem.scores(theta_lo),
            grid_problem.scores(theta_hi),
            rtol=1e-8,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            grid_problem.hess(theta_lo), grid_problem.hess(theta_hi), rtol=1e-8
        )
        # Below the band the lower endpoints of the (0, t_1] rows, each
        # contributing f_i · f(h_l)/P > 0 to ∂(−ℓ)/∂θ_b[0], drop out.
        theta_below = np.array([-1e-6, 0.5, 1.0, 1.5, 2.0, 0.3])
        jump = grid_problem.grad(theta_hi)[0] - grid_problem.grad(theta_below)[0]
        assert jump > 10.0

    def test_r_fixture_rows(self, ref: dict[str, NDArray[np.float64]]) -> None:
        """Shift + scale, rows on both sides of ``h_l = 0`` at once."""
        basis, cd, xd, xs = _ref_problem(ref, np.isfinite(ref["logliki"]))
        _Problem(basis, cd, xd, xs).check_derivatives(ref["theta"])

    def test_mixed_exact_and_interval(self, grid_problem: _Problem) -> None:
        """Exact rows keep ``log f = −h``; interval rows use ``pexp``; ℓ adds up."""
        basis, cd, x_s = grid_problem.basis, grid_problem.cd, grid_problem.scaling
        assert x_s is not None
        y_exact = np.linspace(0.05, 0.9, 20) * basis.support[1]
        n_int = cd.n
        mixed = CensoredData(
            exact=np.concatenate([np.full(n_int, np.nan), y_exact]),
            lower=np.concatenate([cd.lower, y_exact]),
            upper=np.concatenate([cd.upper, y_exact]),
        )
        xs_exact = np.linspace(-1.0, 1.0, 20)[:, None]
        xs_mixed = np.vstack([x_s, xs_exact])
        problem = _Problem(basis, mixed, None, xs_mixed)
        theta = np.array([-0.3, 0.5, 1.0, 1.5, 2.0, 0.3])
        problem.check_derivatives(theta)

        ll_int = -grid_problem.nll(theta)
        ll_exact = log_likelihood(
            theta,
            basis,
            y_exact,
            base_distribution="exponential",
            scaling=xs_exact,
        )
        np.testing.assert_allclose(-problem.nll(theta), ll_int + ll_exact, rtol=1e-12)

    @pytest.mark.parametrize("theta_b0", [-0.3, 0.3])
    def test_truncation_correction(self, theta_b0: float) -> None:
        """Both-finite truncation ``[0, b]``: ``log P_trunc = log F(h(b))`` below."""
        y = np.linspace(0.1, 0.9, 15)
        basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        cd = CensoredData(
            exact=y,
            lower=y,
            upper=y,
            trunc_lower=np.zeros_like(y),
            trunc_upper=np.ones_like(y),
        )
        problem = _Problem(basis, cd, None, None, CensoringType.NONE)
        theta = np.array([theta_b0, 0.4, 0.9, 1.5])
        problem.check_derivatives(theta)
        h = basis.evaluate(y) @ theta
        h_prime = basis.derivative(y) @ theta
        log_p_trunc = _pexp_log_p(np.array([theta_b0]), np.array([theta[-1]]))[0]
        expected = np.sum(-h + np.log(h_prime)) - y.size * log_p_trunc
        np.testing.assert_allclose(-problem.nll(theta), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# The #163 evidence point
# ---------------------------------------------------------------------------


class TestEvidencePoint:
    """Seed 0, ``q = 0.8``, ``θ = [θ_b[0], ½, 1, 1.5, 2 | γ = 0.3]``."""

    @staticmethod
    def _per_row(problem: _Problem, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        basis, cd, x_s = problem.basis, problem.cd, problem.scaling
        assert x_s is not None
        f = np.exp(0.5 * x_s[:, 0] * theta[_P])
        both, h_lo, h_hi = _interval_h(basis, cd, theta[:_P], f)
        # Open rows (t_4, ∞): log S = −h.
        return np.where(both, _pexp_log_p(h_lo, h_hi), -h_lo)

    @pytest.mark.parametrize("theta_b0", [-0.5, -1e-6, 0.0])
    def test_value_is_pexp(self, grid_problem: _Problem, theta_b0: float) -> None:
        theta = np.array([theta_b0, 0.5, 1.0, 1.5, 2.0, 0.3])
        rows = self._per_row(grid_problem, theta)
        assert np.all(rows <= 0.0)  # no row is rewarded outside the support
        np.testing.assert_allclose(-grid_problem.nll(theta), rows.sum(), rtol=1e-12)

    def test_no_reward_below_support(self, grid_problem: _Problem) -> None:
        """#163: ℓ was −194.0 at ``θ_b[0] = −0.5`` against −243.7 at 0."""
        lls = [
            -grid_problem.nll(np.array([t, 0.5, 1.0, 1.5, 2.0, 0.3]))
            for t in (-0.5, -1e-6, 0.0)
        ]
        assert lls[0] < lls[2]
        np.testing.assert_allclose(lls[1], lls[2], atol=1e-4)
