"""Exponential link + scaling term with a shift block (issue #115).

ADR 0005 Decision 1 fixes the lower-support-end rule ``h(a|x_i) ≥ 0`` for every
training row under ``base_distribution="exponential"``.  With a scaling term
the rule is one smooth nonlinear inequality per row (ADR 0005 Decision 2)::

    g_i(θ) = θ_b[0] · f_i + x_d,i · β + offset_i ≥ 0,
    f_i = exp(0.5 · x_s,i · γ),

with Jacobian, in the ``[θ_b | β | γ]`` layout,
``∂g_i/∂θ_b[0] = f_i``, ``∂g_i/∂β = x_d,i``,
``∂g_i/∂γ = 0.5 · θ_b[0] · f_i · x_s,i`` and zero on ``θ_b[1:]``.

auglag (the default solver) carries the block as a nonlinear inequality next to
the linear monotonicity rows (ADR 0005 Decision 3, issue #115).  SLSQP takes the
same ``(g, J)`` as a ``type='ineq'`` dict and trust-constr as a
:class:`scipy.optimize.NonlinearConstraint` (issue #116); all three solvers fit
the same fixture.

R reference (``reference/mlt_exponential_scaling_shift_*.txt``, generated in
``reference/generate_reference.R``): R ``mlt`` has no support rule (ADR 0005
Decision 1), so — as ADR 0005 Decision 4 prescribes for shift + scale — the
reference is a multi-start explicit-constraint ``alabama::auglag`` fit on mlt's
likelihood (checked against ``mlt``'s ``loglik`` / ``scorei``) with the ``hin``
rows ``g_i(θ)`` and ``D θ_b ≥ 0`` written out, polished by Newton on the KKT
system of its (non-degenerate) active set.  It validates mltpy's optimiser on
mltpy's model, not R's modelling choice.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

from mltpy import MLT
from mltpy.basis import BernsteinBasis
from mltpy.constraints import (
    MonotonicityConstraint,
    ScaledSupportConstraint,
    build_constraint_matrices,
    build_constraints,
    build_scaled_support_constraint,
)
from mltpy.likelihood import negative_log_likelihood
from mltpy.optimizer import OptimizerConfig, optimize
from mltpy.variables import CensoredData, CensoringType

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"

_ORDER = 4
_P = _ORDER + 1
# Layout of the fixture's θ: [θ_b (5) | β (1) | γ (1)].
_BETA, _GAMMA = _P, _P + 1
# Feasibility tolerance of the support rows (AugLagOptions.feas_tol and the
# predict-side model._EXPON_SUPPORT_TOL).
_FEAS_TOL = 1e-7
_SUPPORT_MSG = "h(a|x) < 0"


def _central_jacobian(
    g: ScaledSupportConstraint, theta: NDArray[np.float64], eps: float = 1e-6
) -> NDArray[np.float64]:
    """Central finite-difference Jacobian of ``g`` (test reference only)."""
    cols = []
    for j in range(theta.size):
        e = np.zeros_like(theta)
        e[j] = eps
        cols.append((g(theta + e)[0] - g(theta - e)[0]) / (2.0 * eps))
    return np.column_stack(cols)


def _design(
    n: int, q_d: int, q_s: int, seed: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n, q_d)),
        rng.normal(size=(n, q_s)),
        rng.normal(scale=0.3, size=n),
    )


# ---------------------------------------------------------------------------
# Constraint builder
# ---------------------------------------------------------------------------


class TestScaledSupportConstraint:
    """``build_scaled_support_constraint`` returns ``θ ↦ (g(θ), J(θ))``."""

    def test_known_values(self) -> None:
        """Hand-computed rows: p = 3, one shift and one scaling column."""
        X = np.array([[1.0], [-2.0]])
        X_s = np.array([[0.0], [2.0]])
        offset = np.array([0.5, -0.25])
        g = build_scaled_support_constraint(3, X, X_s, offset)
        theta = np.array([0.4, 1.0, 2.0, 0.3, np.log(3.0)])  # [θ_b | β | γ]
        values, jac = g(theta)
        # f = (1, exp(log 3)) = (1, 3).
        np.testing.assert_allclose(
            values, [0.4 * 1 + 0.3 * 1 + 0.5, 0.4 * 3 - 0.6 - 0.25], rtol=1e-14
        )
        np.testing.assert_allclose(
            jac,
            [
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [3.0, 0.0, 0.0, -2.0, 0.5 * 0.4 * 3.0 * 2.0],
            ],
            rtol=1e-14,
        )
        assert g.n_rows == 2
        assert g.n_total == 5

    @pytest.mark.parametrize("q_d", [1, 2])
    @pytest.mark.parametrize("q_s", [1, 3])
    @pytest.mark.parametrize("with_offset", [False, True])
    def test_gamma_zero_equals_linear_rows(
        self, q_d: int, q_s: int, with_offset: bool
    ) -> None:
        """At ``γ = 0`` the block is the unscaled linear support rows (#160)."""
        p, n = 4, 7
        X, X_s, off = _design(n, q_d, q_s)
        offset = off if with_offset else None
        g = build_scaled_support_constraint(p, X, X_s, offset)
        lin = build_constraint_matrices(
            p, total_params=p + q_d, nonneg_lower=True, X=X, offset=offset
        )
        rows, rhs = lin.A_ineq[p - 1 :], lin.b_ineq[p - 1 :]
        rng = np.random.default_rng(1)
        theta_shift = rng.normal(size=p + q_d)
        theta = np.concatenate([theta_shift, np.zeros(q_s)])
        values, jac = g(theta)
        np.testing.assert_allclose(values, rows @ theta_shift - rhs, atol=1e-14)
        np.testing.assert_allclose(jac[:, : p + q_d], rows, atol=1e-14)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize(("q_d", "q_s"), [(1, 1), (2, 1), (1, 2), (3, 2)])
    def test_jacobian_matches_finite_differences(
        self, seed: int, q_d: int, q_s: int
    ) -> None:
        p, n = 5, 9
        X, X_s, off = _design(n, q_d, q_s, seed)
        g = build_scaled_support_constraint(p, X, X_s, off)
        theta = np.random.default_rng(seed + 10).normal(size=p + q_d + q_s)
        _, jac = g(theta)
        np.testing.assert_allclose(
            jac, _central_jacobian(g, theta), rtol=1e-7, atol=1e-8
        )
        # θ_b[1:] never enters the support row.
        np.testing.assert_array_equal(jac[:, 1:p], 0.0)

    def test_single_observation(self) -> None:
        g = build_scaled_support_constraint(
            2, np.array([[2.0]]), np.array([[1.0]]), None
        )
        values, jac = g(np.array([1.0, 2.0, -0.5, 0.0]))
        np.testing.assert_allclose(values, [0.0])
        assert jac.shape == (1, 4)

    def test_offset_none_equals_zero(self) -> None:
        X, X_s, _ = _design(5, 1, 1)
        theta = np.array([0.2, 1.0, 0.3, 0.7])
        a = build_scaled_support_constraint(2, X, X_s, None)(theta)
        b = build_scaled_support_constraint(2, X, X_s, np.zeros(5))(theta)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_scale_only_with_offset(self) -> None:
        """``q_d = 0``: ``g_i = θ_b[0] f_i + offset_i`` (non-zero offset)."""
        X_s = np.array([[0.0], [2.0]])
        g = build_scaled_support_constraint(2, None, X_s, np.array([0.1, -0.2]))
        values, jac = g(np.array([0.5, 1.0, np.log(2.0)]))
        np.testing.assert_allclose(values, [0.6, 0.5 * 2.0 - 0.2])
        np.testing.assert_allclose(jac[:, 0], [1.0, 2.0])

    def test_inputs_are_copied(self) -> None:
        """Mutating the design after building does not change the rows."""
        X, X_s, off = _design(4, 1, 1)
        g = build_scaled_support_constraint(2, X, X_s, off)
        theta = np.array([0.2, 1.0, 0.3, 0.7])
        before = g(theta)[0].copy()
        X[:] = 99.0
        X_s[:] = 99.0
        off[:] = 99.0
        np.testing.assert_array_equal(g(theta)[0], before)

    def test_large_gamma_stays_finite_in_float64(self) -> None:
        """Moderate ``|x_s γ|`` (up to ~700) keeps ``f_i`` finite."""
        g = build_scaled_support_constraint(
            2, np.ones((2, 1)), np.array([[1.0], [-1.0]]), None
        )
        values, jac = g(np.array([1.0, 2.0, 0.0, 1000.0]))
        assert np.all(np.isfinite(values)) and np.all(np.isfinite(jac))
        assert values[1] == pytest.approx(np.exp(-500.0), rel=1e-12)

    @pytest.mark.parametrize(
        ("kwargs", "exc", "match"),
        [
            ({"X_scale": np.ones((3, 0))}, ValueError, "X_scale"),
            ({"X_scale": np.ones(3)}, ValueError, "X_scale"),
            ({"X": np.ones((2, 1))}, ValueError, "rows"),
            ({"X": np.ones(3)}, ValueError, "X"),
            ({"offset": np.ones(2)}, ValueError, "offset"),
            ({"offset": np.array([0.0, np.nan, 0.0])}, ValueError, "finite"),
            ({"X_scale": np.array([[0.0], [np.inf], [1.0]])}, ValueError, "finite"),
            ({"n_params": 0}, ValueError, "n_params"),
            ({"n_params": 2.0}, TypeError, "n_params"),
        ],
    )
    def test_invalid_inputs(
        self, kwargs: dict[str, object], exc: type[Exception], match: str
    ) -> None:
        args: dict[str, object] = {
            "n_params": 3,
            "X": np.ones((3, 1)),
            "X_scale": np.ones((3, 1)),
            "offset": None,
        }
        args.update(kwargs)
        with pytest.raises(exc, match=match):
            build_scaled_support_constraint(**args)  # type: ignore[arg-type]

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize(("q_d", "q_s"), [(1, 1), (0, 2), (2, 3)])
    def test_hessian_matches_finite_differences(
        self, seed: int, q_d: int, q_s: int
    ) -> None:
        """``Σ v_i ∇²g_i`` vs central differences of ``vᵀ J(θ)`` (#116)."""
        p, n = 4, 7
        X, X_s, off = _design(n, max(q_d, 1), q_s, seed=seed)
        g = build_scaled_support_constraint(p, X[:, :q_d] if q_d else None, X_s, off)
        rng = np.random.default_rng(seed + 10)
        theta = rng.normal(size=g.n_total)
        v = rng.uniform(0.0, 2.0, size=n)
        eps = 1e-6
        fd = np.empty((g.n_total, g.n_total))
        for j in range(g.n_total):
            e = np.zeros(g.n_total)
            e[j] = eps
            fd[:, j] = (v @ g(theta + e)[1] - v @ g(theta - e)[1]) / (2.0 * eps)
        hess = g.hessian(theta, v)
        np.testing.assert_allclose(hess, fd, rtol=1e-6, atol=1e-8)
        np.testing.assert_array_equal(hess, hess.T)

    def test_hessian_known_values(self) -> None:
        """One row, q_s = 1: cross term 0.5·f·x_s, γγ term 0.25·θ_b[0]·f·x_s²."""
        g = build_scaled_support_constraint(
            2, np.array([[1.0]]), np.array([[2.0]]), np.array([0.5])
        )
        # θ = [θ_b (2) | β | γ], γ = 0 → f = 1.
        hess = g.hessian(np.array([3.0, 4.0, -1.0, 0.0]), np.array([2.0]))
        expected = np.zeros((4, 4))
        expected[0, 3] = expected[3, 0] = 2.0 * 0.5 * 2.0
        expected[3, 3] = 2.0 * 0.25 * 3.0 * 4.0
        np.testing.assert_array_equal(hess, expected)

    def test_hessian_zero_weights(self) -> None:
        X, X_s, off = _design(4, 1, 2)
        g = build_scaled_support_constraint(3, X, X_s, off)
        hess = g.hessian(np.ones(g.n_total), np.zeros(4))
        np.testing.assert_array_equal(hess, 0.0)

    def test_hessian_bad_shapes(self) -> None:
        X, X_s, off = _design(4, 1, 1)
        g = build_scaled_support_constraint(3, X, X_s, off)
        with pytest.raises(ValueError, match="theta must have shape"):
            g.hessian(np.ones(g.n_total + 1), np.ones(4))
        with pytest.raises(ValueError, match="one weight per support row"):
            g.hessian(np.ones(g.n_total), np.ones(3))

    def test_wrong_theta_length(self) -> None:
        g = build_scaled_support_constraint(3, np.ones((2, 1)), np.ones((2, 1)), None)
        with pytest.raises(ValueError, match="theta"):
            g(np.zeros(4))

    def test_empty_design(self) -> None:
        """Zero training rows: an empty block (shape (0,) and (0, n))."""
        g = build_scaled_support_constraint(2, np.ones((0, 1)), np.ones((0, 1)), None)
        values, jac = g(np.zeros(4))
        assert values.shape == (0,) and jac.shape == (0, 4)


class TestConstraintMatricesSurface:
    """``build_constraint_matrices(X_scale=…)`` surfaces the nonlinear block."""

    def test_shift_block_gives_nonlinear_block(self) -> None:
        p, q_d, q_s, n = 4, 1, 1, 6
        X, X_s, off = _design(n, q_d, q_s)
        cm = build_constraint_matrices(
            p,
            total_params=p + q_d + q_s,
            nonneg_lower=True,
            X=X,
            offset=off,
            q_s=q_s,
            X_scale=X_s,
        )
        # Linear block: monotonicity only, γ and β columns zero.
        assert cm.A_ineq.shape == (p - 1, p + q_d + q_s)
        np.testing.assert_array_equal(cm.A_ineq[:, p:], 0.0)
        np.testing.assert_array_equal(cm.b_ineq, 0.0)
        assert isinstance(cm.nonlinear_ineq, ScaledSupportConstraint)
        theta = np.random.default_rng(3).normal(size=p + q_d + q_s)
        ref = build_scaled_support_constraint(p, X, X_s, off)(theta)
        got = cm.nonlinear_ineq(theta)
        np.testing.assert_array_equal(got[0], ref[0])
        np.testing.assert_array_equal(got[1], ref[1])

    def test_scale_only_nonzero_offset_gives_nonlinear_block(self) -> None:
        p, q_s, n = 4, 1, 5
        _, X_s, off = _design(n, 1, q_s)
        cm = build_constraint_matrices(
            p,
            total_params=p + q_s,
            nonneg_lower=True,
            offset=off,
            q_s=q_s,
            X_scale=X_s,
        )
        assert cm.A_ineq.shape == (p - 1, p + q_s)
        assert cm.nonlinear_ineq is not None
        assert cm.nonlinear_ineq.n_rows == n

    @pytest.mark.parametrize("with_xs", [False, True])
    def test_scale_only_zero_offset_stays_linear(self, with_xs: bool) -> None:
        """The #114 collapse ``θ_b[0] ≥ 0`` is unchanged, with or without X_scale."""
        p, q_s = 4, 1
        kw: dict[str, object] = {"X_scale": np.ones((3, q_s))} if with_xs else {}
        cm = build_constraint_matrices(
            p,
            total_params=p + q_s,
            nonneg_lower=True,
            q_s=q_s,
            **kw,  # type: ignore[arg-type]
        )
        assert cm.nonlinear_ineq is None
        assert cm.A_ineq.shape == (p, p + q_s)

    def test_shift_block_without_x_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="X_scale"):
            build_constraint_matrices(
                4, total_params=6, nonneg_lower=True, X=np.ones((3, 1)), q_s=1
            )

    def test_x_scale_column_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="q_s"):
            build_constraint_matrices(
                4,
                total_params=6,
                nonneg_lower=True,
                X=np.ones((3, 1)),
                q_s=1,
                X_scale=np.ones((3, 2)),
            )

    def test_without_nonneg_lower_no_block(self) -> None:
        cm = build_constraint_matrices(
            4, total_params=6, q_s=1, X_scale=np.ones((3, 1))
        )
        assert cm.nonlinear_ineq is None

    def test_default_field_is_none(self) -> None:
        cm = build_constraint_matrices(4)
        assert cm.nonlinear_ineq is None


class TestScipyConstraintForms:
    """``build_constraints(X_scale=…)`` gives the block in scipy's forms (#116)."""

    @staticmethod
    def _build(
        solver: str, *, q_d: int = 1, with_offset: bool = True
    ) -> tuple[list[object], ScaledSupportConstraint, int]:
        p, q_s, n = 4, 2, 6
        X, X_s, off = _design(n, max(q_d, 1), q_s, seed=5)
        x_d = X[:, :q_d] if q_d else None
        offset = off if with_offset else None
        total = p + q_d + q_s
        cons = build_constraints(
            p,
            solver=solver,  # type: ignore[call-overload]
            total_params=total,
            nonneg_lower=True,
            X=x_d,
            offset=offset,
            q_s=q_s,
            X_scale=X_s,
        )
        return cons, build_scaled_support_constraint(p, x_d, X_s, offset), total

    @pytest.mark.parametrize(("q_d", "with_offset"), [(1, True), (2, False), (0, True)])
    def test_slsqp_dict(self, q_d: int, with_offset: bool) -> None:
        cons, g, total = self._build("slsqp", q_d=q_d, with_offset=with_offset)
        # [monotonicity, support]; the support dict is g itself.
        assert len(cons) == 2
        d = cons[1]
        assert isinstance(d, dict) and d["type"] == "ineq"
        theta = np.random.default_rng(1).normal(size=total)
        np.testing.assert_array_equal(d["fun"](theta), g(theta)[0])
        np.testing.assert_array_equal(d["jac"](theta), g(theta)[1])
        mono = cons[0]
        assert isinstance(mono, dict)
        assert np.asarray(mono["jac"](theta)).shape == (3, total)

    @pytest.mark.parametrize(("q_d", "with_offset"), [(1, True), (2, False), (0, True)])
    def test_trust_constr_nonlinear_constraint(
        self, q_d: int, with_offset: bool
    ) -> None:
        cons, g, total = self._build("trust-constr", q_d=q_d, with_offset=with_offset)
        assert len(cons) == 2
        assert isinstance(cons[0], LinearConstraint)
        nlc = cons[1]
        assert isinstance(nlc, NonlinearConstraint)
        assert np.all(np.asarray(nlc.lb) == 0.0)
        assert np.all(np.isposinf(np.asarray(nlc.ub)))
        theta = np.random.default_rng(2).normal(size=total)
        np.testing.assert_array_equal(nlc.fun(theta), g(theta)[0])
        np.testing.assert_array_equal(nlc.jac(theta), g(theta)[1])
        v = np.linspace(0.0, 1.0, g.n_rows)
        np.testing.assert_array_equal(nlc.hess(theta, v), g.hessian(theta, v))

    def test_gamma_zero_matches_linear_rows(self) -> None:
        """At γ = 0 the SLSQP block equals the unscaled linear support rows."""
        p, q_s, n = 4, 1, 5
        X, X_s, off = _design(n, 1, q_s, seed=9)
        scaled = build_constraints(
            p,
            solver="slsqp",
            total_params=p + 1 + q_s,
            nonneg_lower=True,
            X=X,
            offset=off,
            q_s=q_s,
            X_scale=X_s,
        )
        linear = build_constraints(
            p, solver="slsqp", total_params=p + 1, nonneg_lower=True, X=X, offset=off
        )
        theta = np.array([0.3, 0.5, 0.9, 1.2, -0.4])
        np.testing.assert_allclose(
            scaled[1]["fun"](np.append(theta, 0.0)),
            linear[1]["fun"](theta),
            rtol=0,
            atol=1e-15,
        )

    def test_scale_only_zero_offset_stays_linear(self) -> None:
        """The #114 collapse is unchanged when X_scale is passed."""
        cons = build_constraints(
            4,
            solver="trust-constr",
            total_params=5,
            nonneg_lower=True,
            q_s=1,
            X_scale=np.ones((3, 1)),
        )
        assert all(isinstance(c, LinearConstraint) for c in cons)
        assert len(cons) == 2

    @pytest.mark.parametrize("solver", ["slsqp", "trust-constr"])
    def test_missing_x_scale_raises(self, solver: str) -> None:
        with pytest.raises(ValueError, match="X_scale"):
            build_constraints(
                4,
                solver=solver,  # type: ignore[call-overload]
                total_params=6,
                nonneg_lower=True,
                X=np.ones((3, 1)),
                q_s=1,
            )

    def test_x_scale_row_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same number of rows"):
            build_constraints(
                4,
                solver="slsqp",
                total_params=6,
                nonneg_lower=True,
                X=np.ones((3, 1)),
                q_s=1,
                X_scale=np.ones((4, 1)),
            )

    def test_boundary_rows_follow_support(self) -> None:
        """``lower`` / ``upper`` pins come after the nonlinear block."""
        cons = build_constraints(
            4,
            1.0,
            2.0,
            solver="trust-constr",
            total_params=6,
            nonneg_lower=True,
            X=np.ones((3, 1)),
            q_s=1,
            X_scale=np.ones((3, 1)),
        )
        assert isinstance(cons[1], NonlinearConstraint)
        assert isinstance(cons[2], LinearConstraint)


# ---------------------------------------------------------------------------
# R fixture and fitted models
# ---------------------------------------------------------------------------


def _load(name: str) -> NDArray[np.float64]:
    return np.loadtxt(REF_DIR / f"mlt_exponential_scaling_shift_{name}.txt")


@pytest.fixture(scope="module")
def ref() -> dict[str, NDArray[np.float64]]:
    """The R fixture ``mlt_exponential_scaling_shift_*``."""
    return {
        "y": _load("y"),
        "x_d": _load("xd").reshape(-1, 1),
        "x_s": _load("xs").reshape(-1, 1),
        "offset": _load("offset"),
        "support": _load("support"),
        "theta": _load("theta"),
        "ll": np.atleast_1d(_load("ll")),
        "mu": _load("mu"),
    }


def _model(
    ref: dict[str, NDArray[np.float64]],
    *,
    x_s: NDArray[np.float64] | None = None,
    censoring: CensoringType = CensoringType.NONE,
    **config: object,
) -> MLT:
    sup = ref["support"]
    config.setdefault("random_state", 0)
    return MLT(
        order=_ORDER,
        support=(float(sup[0]), float(sup[1])),
        censoring=censoring,
        base_distribution="exponential",
        scaling=ref["x_s"] if x_s is None else x_s,
        optimizer_config=OptimizerConfig(**config),  # type: ignore[arg-type]
    )


def _trust_constr_reference(
    ref: dict[str, NDArray[np.float64]],
    y: NDArray[np.float64] | CensoredData,
    x_d: NDArray[np.float64] | None,
    censoring: CensoringType,
) -> float:
    """Independent log-likelihood optimum: scipy trust-constr, rows inline.

    The support rows ``θ_b[0]·exp(0.5·x_s·γ) + x_d·β + offset ≥ 0`` are
    written out here (not via mltpy's builder) as a ``NonlinearConstraint``,
    monotonicity as a ``LinearConstraint``; the objective is mltpy's
    likelihood.  Started from the feasible default start.
    """
    sup = ref["support"]
    basis = BernsteinBasis(order=_ORDER, support=(float(sup[0]), float(sup[1])))
    x_s, off = ref["x_s"][:, 0], ref["offset"]
    q_d = 0 if x_d is None else x_d.shape[1]

    def nll(t: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        v, gr = negative_log_likelihood(
            t,
            basis,
            y,
            x_d,
            censoring=censoring,
            gradient=True,
            base_distribution="exponential",
            offset=off,
            scaling=ref["x_s"],
        )
        return float(v), np.asarray(gr)

    def support(t: NDArray[np.float64]) -> NDArray[np.float64]:
        shift = 0.0 if x_d is None else x_d @ t[_P : _P + q_d]
        return np.asarray(t[0] * np.exp(0.5 * x_s * t[-1]) + shift + off)

    start = np.concatenate(
        [
            np.linspace(0.0, 1.0, _P) + max(0.0, float(np.max(-off))),
            np.zeros(q_d + 1),
        ]
    )
    mono = np.hstack(
        [MonotonicityConstraint(_P).as_matrix(), np.zeros((_P - 1, q_d + 1))]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # quasi-Newton update notices
        res = minimize(
            nll,
            start,
            jac=True,
            method="trust-constr",
            constraints=[
                LinearConstraint(mono, lb=0.0),
                NonlinearConstraint(support, 0.0, np.inf),
            ],
            options={"gtol": 1e-10, "xtol": 1e-12, "maxiter": 5000},
        )
    assert support(res.x).min() >= -1e-6
    assert np.all(mono @ res.x >= -1e-6)
    return float(-res.fun)


def _censored(
    ref: dict[str, NDArray[np.float64]], censoring: CensoringType
) -> CensoredData:
    """Censor the fixture's ``y``: top 20 % right, bottom 20 % left, or ±20 %."""
    y = ref["y"]
    if censoring is CensoringType.RIGHT:
        c = float(np.quantile(y, 0.8))
        return CensoredData.right_censored(np.minimum(y, c), y <= c)
    if censoring is CensoringType.LEFT:
        c = float(np.quantile(y, 0.2))
        return CensoredData.left_censored(np.maximum(y, c), y < c)
    upper = np.minimum(1.25 * y, float(ref["support"][1]))
    return CensoredData.interval_censored(0.8 * y, upper)


@pytest.fixture(scope="module")
def fitted(ref: dict[str, NDArray[np.float64]]) -> MLT:
    model = _model(ref)
    model.fit(ref["y"], X=ref["x_d"], offset=ref["offset"])
    return model


def _support_values(
    model: MLT,
    x_d: NDArray[np.float64] | None,
    x_s: NDArray[np.float64],
    offset: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """``h(a|x_i)`` from the fitted θ, offset included."""
    g = build_scaled_support_constraint(_P, x_d, x_s, offset)
    return g(model.theta_)[0]


def _support_violations(rec: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in rec if _SUPPORT_MSG in str(w.message)]


class TestRParity:
    """auglag matches the explicit-constraint ``alabama::auglag`` reference.

    Tolerances are CLAUDE.md's ``rtol=1e-6, atol=1e-10`` on logLik and θ.
    The active set is non-degenerate (two support rows, multipliers ≈ 70), so
    no active-boundary relaxation is needed.
    """

    def test_converged(self, fitted: MLT) -> None:
        assert fitted.result_.converged
        assert fitted.result_.kkt_residual is not None
        assert fitted.result_.kkt_residual < 1e-4

    def test_log_likelihood(
        self, fitted: MLT, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        np.testing.assert_allclose(
            fitted.result_.log_likelihood, ref["ll"][0], rtol=1e-6, atol=1e-10
        )

    def test_theta(self, fitted: MLT, ref: dict[str, NDArray[np.float64]]) -> None:
        np.testing.assert_allclose(fitted.theta_, ref["theta"], rtol=1e-6, atol=1e-10)

    def test_gamma_and_beta_accessors(
        self, fitted: MLT, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        """Raw ``ctm`` order and signs: no flip on β or γ (ADR 0005 Context)."""
        np.testing.assert_allclose(
            fitted.gamma_, ref["theta"][_GAMMA:], rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            fitted.theta_[_BETA], ref["theta"][_BETA], rtol=1e-6, atol=1e-10
        )

    def test_multipliers(
        self, fitted: MLT, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        """``mu_ineq`` = [monotonicity (4) | support (n)], as R's KKT fit."""
        mu = fitted.result_.mu_ineq
        assert mu is not None
        assert mu.shape == ref["mu"].shape
        np.testing.assert_allclose(mu, ref["mu"], rtol=1e-6, atol=1e-10)
        np.testing.assert_array_equal(
            np.flatnonzero(mu > 1e-3), np.flatnonzero(ref["mu"])
        )

    def test_reference_gradient_matches_finite_differences(
        self, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        """Full ``[θ_b | β | γ]`` analytic NLL gradient vs central differences."""
        sup = ref["support"]
        basis = BernsteinBasis(order=_ORDER, support=(float(sup[0]), float(sup[1])))
        kw: dict[str, object] = {
            "base_distribution": "exponential",
            "offset": ref["offset"],
            "scaling": ref["x_s"],
        }
        theta = ref["theta"]
        _, grad = negative_log_likelihood(
            theta,
            basis,
            ref["y"],
            ref["x_d"],
            gradient=True,
            **kw,  # type: ignore[arg-type]
        )
        eps = 1e-6
        fd = np.empty_like(theta)
        for j in range(theta.size):
            e = np.zeros_like(theta)
            e[j] = eps
            hi = negative_log_likelihood(theta + e, basis, ref["y"], ref["x_d"], **kw)  # type: ignore[arg-type]
            lo = negative_log_likelihood(theta - e, basis, ref["y"], ref["x_d"], **kw)  # type: ignore[arg-type]
            fd[j] = (float(hi) - float(lo)) / (2.0 * eps)  # type: ignore[arg-type]
        np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-5)

    def test_kkt_stationarity(
        self, fitted: MLT, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        """∇NLL(θ̂) = A(θ̂)ᵀμ with the reported rows ``[D ; J(θ̂)]``."""
        sup = ref["support"]
        basis = BernsteinBasis(order=_ORDER, support=(float(sup[0]), float(sup[1])))
        _, grad = negative_log_likelihood(
            fitted.theta_,
            basis,
            ref["y"],
            ref["x_d"],
            gradient=True,
            base_distribution="exponential",
            offset=ref["offset"],
            scaling=ref["x_s"],
        )
        A = fitted.result_.constraint_A_ineq
        mu = fitted.result_.mu_ineq
        assert A is not None and mu is not None
        resid = np.asarray(grad) - A.T @ mu
        # |∇f| ≈ 1.6e2; the residual is ≈ 2e-6 (≈ 1e-8 relative).
        assert np.max(np.abs(resid)) < 1e-5


class TestFeasibility:
    """``min_y h(y|x_i) = h(a|x_i) ≥ 0`` on every training row."""

    def test_support_rows_nonnegative(
        self, fitted: MLT, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        g = _support_values(fitted, ref["x_d"], ref["x_s"], ref["offset"])
        assert g.min() >= -_FEAS_TOL
        # The two binding rows sit on the boundary.
        assert np.sum(g < 1e-6) == 2

    def test_h_nonnegative_on_support_grid(
        self, fitted: MLT, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        sup = ref["support"]
        grid = np.linspace(sup[0], sup[1], 201)
        h0 = fitted.basis.evaluate(grid) @ fitted.theta_[:_P]
        f = np.exp(0.5 * ref["x_s"][:, 0] * fitted.theta_[_GAMMA])
        shift = ref["x_d"][:, 0] * fitted.theta_[_BETA] + ref["offset"]
        h = f[:, None] * h0[None, :] + shift[:, None]
        assert h.min() >= -_FEAS_TOL
        # The minimum over y is attained at the lower end a.
        np.testing.assert_allclose(h.min(axis=1), h[:, 0], atol=1e-12)

    @pytest.mark.parametrize("what", ["distribution", "trafo", "quantile"])
    def test_in_sample_predict_silent(
        self, fitted: MLT, ref: dict[str, NDArray[np.float64]], what: str
    ) -> None:
        """The #157 warning stays silent at the training rows."""
        n = ref["y"].size
        y_new = np.full(n, 0.5) if what == "quantile" else ref["y"]
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = fitted.predict(
                y_new,
                X_new=ref["x_d"],
                what=what,  # type: ignore[arg-type]
                offset_new=ref["offset"],
                X_scale_new=ref["x_s"],
            )
        assert _support_violations(rec) == []
        assert np.all(np.isfinite(out))

    def test_cdf_at_lower_end_is_atom(
        self, fitted: MLT, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        """``F(a|x_i) = 1 − exp(−h(a|x_i)) ∈ [0, 1)``: zero on the binding rows."""
        a = float(ref["support"][0])
        n = ref["y"].size
        cdf = fitted.predict(
            np.full(n, a),
            X_new=ref["x_d"],
            what="distribution",
            offset_new=ref["offset"],
            X_scale_new=ref["x_s"],
        )
        g = _support_values(fitted, ref["x_d"], ref["x_s"], ref["offset"])
        np.testing.assert_allclose(cdf, -np.expm1(-g), rtol=1e-10, atol=1e-12)
        assert cdf.min() >= -_FEAS_TOL


_SOLVERS = ("auglag", "slsqp", "trust-constr")
_SCIPY_SOLVERS = ("slsqp", "trust-constr")
# θ tolerances against the reference.  logLik is held to CLAUDE.md's
# rtol=1e-6 by every solver, and auglag and trust-constr (with the analytic
# constraint Hessian) meet rtol=1e-6, atol=1e-10 on θ too.  SLSQP stops on
# its line search ~3.5e-6 relative short along the flat y-shape direction
# θ_b[1:] (logLik within 5e-8).  Before #165 it started from a flattened θ_b,
# failed its first attempt and ended ~3e-5 short from a restart, which needed
# 1e-4 here (as tests/test_exponential_scaling_no_shift.py still uses).
_THETA_TOL: dict[str, dict[str, float]] = {
    "auglag": {"rtol": 1e-6, "atol": 1e-10},
    "slsqp": {"rtol": 1e-5, "atol": 1e-5},
    "trust-constr": {"rtol": 1e-6, "atol": 1e-10},
}


def _solver(model: MLT) -> str:
    return str(model.optimizer_config.solver)


@pytest.fixture(scope="module", params=_SOLVERS)
def solver_fitted(
    request: pytest.FixtureRequest, ref: dict[str, NDArray[np.float64]]
) -> MLT:
    """The R fixture fitted on each of the three solvers (issue #116)."""
    model = _model(ref, solver=request.param)
    model.fit(ref["y"], X=ref["x_d"], offset=ref["offset"])
    return model


class TestSolverParity:
    """auglag, SLSQP and trust-constr fit the shift block identically (#116).

    The scipy solvers get the support rows natively (SLSQP ``type='ineq'``
    dict, trust-constr ``NonlinearConstraint``), built from the same
    :class:`ScaledSupportConstraint` auglag uses.  Tolerances are CLAUDE.md's
    ``rtol=1e-6, atol=1e-10`` against the ``alabama::auglag`` reference.
    """

    def test_converged(self, solver_fitted: MLT) -> None:
        assert solver_fitted.result_.converged

    def test_log_likelihood(
        self, solver_fitted: MLT, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        np.testing.assert_allclose(
            solver_fitted.result_.log_likelihood, ref["ll"][0], rtol=1e-6, atol=1e-10
        )

    def test_theta(
        self, solver_fitted: MLT, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        np.testing.assert_allclose(
            solver_fitted.theta_, ref["theta"], **_THETA_TOL[_solver(solver_fitted)]
        )

    def test_agrees_with_auglag(self, solver_fitted: MLT, fitted: MLT) -> None:
        np.testing.assert_allclose(
            solver_fitted.theta_, fitted.theta_, **_THETA_TOL[_solver(solver_fitted)]
        )
        np.testing.assert_allclose(
            solver_fitted.result_.log_likelihood,
            fitted.result_.log_likelihood,
            rtol=1e-6,
            atol=1e-10,
        )

    def test_support_rows_nonnegative(
        self, solver_fitted: MLT, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        g = _support_values(solver_fitted, ref["x_d"], ref["x_s"], ref["offset"])
        assert g.min() >= -_FEAS_TOL
        # The reference's two binding rows sit on the boundary.
        assert np.sum(g < 1e-6) == 2

    def test_h_nonnegative_on_support_grid(
        self, solver_fitted: MLT, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        """``min_y h(y|x_i) ≥ 0`` for every training row, on a dense grid."""
        sup = ref["support"]
        grid = np.linspace(sup[0], sup[1], 201)
        theta = solver_fitted.theta_
        h0 = solver_fitted.basis.evaluate(grid) @ theta[:_P]
        f = np.exp(0.5 * ref["x_s"][:, 0] * theta[_GAMMA])
        shift = ref["x_d"][:, 0] * theta[_BETA] + ref["offset"]
        h = f[:, None] * h0[None, :] + shift[:, None]
        assert h.min() >= -_FEAS_TOL
        assert np.all(np.diff(theta[:_P]) >= -_FEAS_TOL)


class TestSolverDispatch:
    """``optimize()`` routes the nonlinear block to every solver."""

    @pytest.mark.parametrize("solver", _SCIPY_SOLVERS)
    def test_optimize_direct_scipy(
        self, ref: dict[str, NDArray[np.float64]], solver: str
    ) -> None:
        """``optimize()`` fits the shift block on SLSQP / trust-constr (#116)."""
        sup = ref["support"]
        res = optimize(
            BernsteinBasis(order=_ORDER, support=(float(sup[0]), float(sup[1]))),
            ref["y"],
            ref["x_d"],
            base_distribution="exponential",
            offset=ref["offset"],
            scaling=ref["x_s"],
            config=OptimizerConfig(random_state=0, solver=solver),  # type: ignore[arg-type]
        )
        assert res.converged
        np.testing.assert_allclose(
            res.log_likelihood, ref["ll"][0], rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(res.theta, ref["theta"], **_THETA_TOL[solver])

    @pytest.mark.parametrize("solver", _SCIPY_SOLVERS)
    def test_scale_only_nonzero_offset_scipy(
        self, ref: dict[str, NDArray[np.float64]], solver: str
    ) -> None:
        """``q_d = 0`` with an offset is nonlinear too; scipy solvers fit it."""
        model = _model(ref, solver=solver)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="delta_grad == 0.0")
            model.fit(ref["y"], offset=ref["offset"])
        g = _support_values(model, None, ref["x_s"], ref["offset"])
        assert g.min() >= -_FEAS_TOL
        assert model.result_.converged
        ref_ll = _trust_constr_reference(ref, ref["y"], None, CensoringType.NONE)
        assert model.result_.log_likelihood >= ref_ll - 1e-6
        np.testing.assert_allclose(model.result_.log_likelihood, ref_ll, rtol=1e-6)

    @pytest.mark.parametrize("solver", _SCIPY_SOLVERS)
    def test_fixed_beta_reduction(
        self, ref: dict[str, NDArray[np.float64]], solver: str
    ) -> None:
        """``fixed_params`` reduction slices the nonlinear block to the free θ."""
        beta = float(ref["theta"][_BETA])
        model = _model(ref, solver=solver, fixed_params={_BETA: beta})
        model.fit(ref["y"], X=ref["x_d"], offset=ref["offset"])
        assert model.theta_[_BETA] == beta
        np.testing.assert_allclose(model.theta_, ref["theta"], **_THETA_TOL[solver])

    @pytest.mark.parametrize("solver", _SCIPY_SOLVERS)
    def test_fixed_gamma_zero_equals_unscaled_fit(
        self, ref: dict[str, NDArray[np.float64]], solver: str
    ) -> None:
        """γ pinned at 0 turns the block into the unscaled linear rows (#160)."""
        model = _model(ref, solver=solver, fixed_params={_GAMMA: 0.0})
        model.fit(ref["y"], X=ref["x_d"], offset=ref["offset"])
        sup = ref["support"]
        unscaled = MLT(
            order=_ORDER,
            support=(float(sup[0]), float(sup[1])),
            base_distribution="exponential",
            optimizer_config=OptimizerConfig(random_state=0, solver=solver),  # type: ignore[arg-type]
        )
        unscaled.fit(ref["y"], X=ref["x_d"], offset=ref["offset"])
        assert model.theta_[_GAMMA] == 0.0
        np.testing.assert_allclose(
            model.result_.log_likelihood, unscaled.result_.log_likelihood, rtol=1e-6
        )

    @pytest.mark.parametrize("solver", _SCIPY_SOLVERS)
    @pytest.mark.parametrize(
        "censoring", [CensoringType.RIGHT, CensoringType.LEFT, CensoringType.INTERVAL]
    )
    def test_censored_fit_scipy(
        self,
        ref: dict[str, NDArray[np.float64]],
        censoring: CensoringType,
        solver: str,
    ) -> None:
        """Censored fits on the scipy solvers match the trust-constr reference."""
        data = _censored(ref, censoring)
        model = _model(ref, censoring=censoring, solver=solver)
        model.fit(data, X=ref["x_d"], offset=ref["offset"])
        g = _support_values(model, ref["x_d"], ref["x_s"], ref["offset"])
        assert g.min() >= -_FEAS_TOL
        assert np.all(np.diff(model.theta_[:_P]) >= -_FEAS_TOL)
        # Right-censoring leaves θ_b[-1] (above the censoring point) nearly
        # unidentified; trust-constr then ends ~2e-8 relative short of the
        # reference in logLik, inside CLAUDE.md's rtol but not within 1e-6
        # absolute, so only the relative comparison is made here.
        ref_ll = _trust_constr_reference(ref, data, ref["x_d"], censoring)
        np.testing.assert_allclose(model.result_.log_likelihood, ref_ll, rtol=1e-6)

    def test_trust_constr_first_attempt_warning_free(
        self, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        """With the analytic constraint Hessian, trust-constr needs no restart.

        A quasi-Newton constraint Hessian sees ``J`` unchanged along every
        step in ``θ_b[1:]`` / ``β`` and warns ``delta_grad == 0.0``; it also
        ran out of iterations from the default start.
        """
        model = _model(ref, solver="trust-constr", max_restarts=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.fit(ref["y"], X=ref["x_d"], offset=ref["offset"])
        assert model.result_.converged
        assert model.result_.n_restarts == 0
        np.testing.assert_allclose(model.theta_, ref["theta"], rtol=1e-6, atol=1e-10)

    @pytest.mark.parametrize("scaled", [True, False])
    @pytest.mark.parametrize("solver", _SCIPY_SOLVERS)
    def test_first_attempt_converges_warning_free(
        self, ref: dict[str, NDArray[np.float64]], solver: str, scaled: bool
    ) -> None:
        """Both scipy solvers converge from the default start (issue #165).

        The support floor ``c = max(−offset) ≈ 0.74`` used to flatten the
        start to ``θ_b = [c, c, c, 0.75, 1]``: h' vanished to second order at
        the lower end, ‖∇ℓ‖ was ~8e7, and SLSQP's first step overflowed
        ``exp(0.5·x_s·γ)``.  Unscaled, both solvers ran out of iterations.
        The parallel lift keeps ‖∇ℓ‖ ~5e2, as on the other links.
        """
        sup = ref["support"]
        basis = BernsteinBasis(order=_ORDER, support=(float(sup[0]), float(sup[1])))
        x_s = ref["x_s"] if scaled else None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = optimize(
                basis,
                ref["y"],
                ref["x_d"],
                base_distribution="exponential",
                offset=ref["offset"],
                scaling=x_s,
                config=OptimizerConfig(random_state=0, solver=solver, max_restarts=0),  # type: ignore[arg-type]
            )
        assert res.converged
        assert res.n_restarts == 0
        if scaled:
            expected = float(ref["ll"][0])
        else:
            # No R fixture for the unscaled fit; auglag is the reference
            # solver (test_fixed_gamma_zero_equals_unscaled_fit uses the same).
            expected = optimize(
                basis,
                ref["y"],
                ref["x_d"],
                base_distribution="exponential",
                offset=ref["offset"],
                config=OptimizerConfig(random_state=0),
            ).log_likelihood
        np.testing.assert_allclose(res.log_likelihood, expected, rtol=1e-6, atol=1e-10)

    @pytest.mark.parametrize("solver", _SCIPY_SOLVERS)
    def test_restarts_stay_feasible(
        self, ref: dict[str, NDArray[np.float64]], solver: str
    ) -> None:
        """Restarts perturb the γ = 0 start, where the rows are linear (#160)."""
        model = _model(ref, solver=solver, random_state=7, max_restarts=3)
        model.fit(ref["y"], X=ref["x_d"], offset=ref["offset"])
        np.testing.assert_allclose(model.theta_, ref["theta"], **_THETA_TOL[solver])

    def test_optimize_direct(self, ref: dict[str, NDArray[np.float64]]) -> None:
        """``optimize()`` itself fits the shift block on auglag."""
        sup = ref["support"]
        res = optimize(
            BernsteinBasis(order=_ORDER, support=(float(sup[0]), float(sup[1]))),
            ref["y"],
            ref["x_d"],
            base_distribution="exponential",
            offset=ref["offset"],
            scaling=ref["x_s"],
            config=OptimizerConfig(random_state=0),
        )
        np.testing.assert_allclose(res.log_likelihood, ref["ll"][0], rtol=1e-6)
        assert res.constraint_A_ineq is not None and res.mu_ineq is not None
        n = ref["y"].size
        assert res.constraint_A_ineq.shape == (_P - 1 + n, _P + 2)
        assert res.mu_ineq.shape == (_P - 1 + n,)
        # Reported support rows are the Jacobian J(θ̂) at the returned θ.
        g = build_scaled_support_constraint(_P, ref["x_d"], ref["x_s"], ref["offset"])
        np.testing.assert_array_equal(res.constraint_A_ineq[_P - 1 :], g(res.theta)[1])
        np.testing.assert_array_equal(
            res.constraint_A_ineq[: _P - 1],
            np.hstack([MonotonicityConstraint(_P).as_matrix(), np.zeros((_P - 1, 2))]),
        )


class TestComposition:
    """``fixed_params``, offsets, censoring and inference on the nonlinear path."""

    def test_fixed_beta_at_optimum_reproduces_fit(
        self, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        """Pinning β at R's value is an equality row next to the block.

        The fit stops at auglag's KKT floor (residual ≈ 1.5e-5, ADR 0004) on
        the flat ``θ_b[1:]`` direction, and with two active support rows the
        Newton-CG polish does not run.  logLik is held to CLAUDE.md's
        tolerance (measured 1.4e-7 short); θ lands ≈ 9e-6 relative off R's.
        Stopping closer depended on the start: the flattened pre-#165 start
        happened to stop 2e-7 away.
        """
        beta = float(ref["theta"][_BETA])
        model = _model(ref, fixed_params={_BETA: beta})
        model.fit(ref["y"], X=ref["x_d"], offset=ref["offset"])
        assert model.theta_[_BETA] == pytest.approx(beta, abs=1e-8)
        np.testing.assert_allclose(
            model.result_.log_likelihood, ref["ll"][0], rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(model.theta_, ref["theta"], rtol=5e-5, atol=1e-10)
        assert model.result_.constraint_C_eq is not None

    def test_fixed_gamma_zero_equals_unscaled_fit(
        self, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        """γ pinned at 0 turns the block into the unscaled linear rows (#160)."""
        model = _model(ref, fixed_params={_GAMMA: 0.0})
        model.fit(ref["y"], X=ref["x_d"], offset=ref["offset"])
        sup = ref["support"]
        unscaled = MLT(
            order=_ORDER,
            support=(float(sup[0]), float(sup[1])),
            base_distribution="exponential",
            optimizer_config=OptimizerConfig(random_state=0),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unscaled.fit(ref["y"], X=ref["x_d"], offset=ref["offset"])
        assert model.theta_[_GAMMA] == pytest.approx(0.0, abs=1e-8)
        np.testing.assert_allclose(
            model.result_.log_likelihood, unscaled.result_.log_likelihood, rtol=1e-6
        )
        np.testing.assert_allclose(model.theta_[:_GAMMA], unscaled.theta_, atol=1e-4)

    def test_scale_only_nonzero_offset_matches_scipy_reference(
        self, ref: dict[str, NDArray[np.float64]]
    ) -> None:
        """``q_d = 0``, non-zero offset: ``θ_b[0]·f_i ≥ −offset_i`` (nonlinear)."""
        model = _model(ref)
        model.fit(ref["y"], offset=ref["offset"])
        g = _support_values(model, None, ref["x_s"], ref["offset"])
        assert g.min() >= -_FEAS_TOL
        assert model.result_.converged
        ref_ll = _trust_constr_reference(ref, ref["y"], None, CensoringType.NONE)
        # auglag is at least as good as the reference, and they agree.
        assert model.result_.log_likelihood >= ref_ll - 1e-6
        np.testing.assert_allclose(model.result_.log_likelihood, ref_ll, rtol=1e-6)

    @pytest.mark.parametrize(
        "censoring", [CensoringType.RIGHT, CensoringType.LEFT, CensoringType.INTERVAL]
    )
    def test_censored_fit_matches_scipy_reference(
        self, ref: dict[str, NDArray[np.float64]], censoring: CensoringType
    ) -> None:
        """Censoring changes nothing about the constraint (ADR 0005 Decision 3).

        Right-censored rows contribute ``−log S = h(c_i|x_i)`` with no
        ``log h'`` term; this is the case that drove auglag to a non-monotone
        ``θ_b`` before monotonicity was kept as hard bounds in the inner
        solve (``optimizer._increment_auglag``).
        """
        data = _censored(ref, censoring)
        model = _model(ref, censoring=censoring)
        model.fit(data, X=ref["x_d"], offset=ref["offset"])
        g = _support_values(model, ref["x_d"], ref["x_s"], ref["offset"])
        assert g.min() >= -_FEAS_TOL
        assert np.all(np.diff(model.theta_[:_P]) >= -_FEAS_TOL)
        assert model.result_.converged
        ref_ll = _trust_constr_reference(ref, data, ref["x_d"], censoring)
        assert model.result_.log_likelihood >= ref_ll - 1e-6
        np.testing.assert_allclose(model.result_.log_likelihood, ref_ll, rtol=1e-6)

    def test_inference_finite(self, fitted: MLT) -> None:
        """Wald covariance on the active (linearised) rows is finite, symmetric."""
        V = fitted.vcov()
        assert V.shape == (_P + 2, _P + 2)
        assert np.all(np.isfinite(V))
        np.testing.assert_allclose(V, V.T, atol=1e-10)
        assert np.all(np.diag(V) >= -1e-10)

    def test_restarts_reach_same_optimum(
        self, ref: dict[str, NDArray[np.float64]], fitted: MLT
    ) -> None:
        """Different seeds (restart perturbations) land on the same optimum."""
        model = _model(ref, random_state=7, max_restarts=3)
        model.fit(ref["y"], X=ref["x_d"], offset=ref["offset"])
        np.testing.assert_allclose(model.theta_, fitted.theta_, rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("scale", [10.0, 0.01])
    def test_extreme_scaling_covariate(
        self, ref: dict[str, NDArray[np.float64]], scale: float
    ) -> None:
        """Rescaled ``x_s`` (γ absorbs the scale) keeps the fit feasible/finite."""
        x_s = ref["x_s"] * scale
        model = _model(ref, x_s=x_s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(ref["y"], X=ref["x_d"], offset=ref["offset"])
        g = _support_values(model, ref["x_d"], x_s, ref["offset"])
        assert g.min() >= -_FEAS_TOL
        # γ absorbs the rescaling, so the optimum's logLik is R's.
        np.testing.assert_allclose(
            model.result_.log_likelihood, ref["ll"][0], rtol=1e-6, atol=1e-10
        )
