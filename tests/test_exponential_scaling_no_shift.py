"""Exponential link + scaling term with no shift block (issue #114).

ADR 0005 Decision 1 fixes the lower-support-end rule ``h(a|x_i) ≥ 0`` for
every training row under ``base_distribution="exponential"``.  With a scaling
term and no shift block (``q_d = 0``)::

    h(a|x_i) = θ_b[0] · f_i,    f_i = exp(0.5 · x_s,i · γ) > 0,

so the rule collapses to the single linear row ``θ_b[0] ≥ 0`` (ADR 0005
Decision 2, simplification 1).  SLSQP and trust-constr fit it as a linear
row; auglag solves in increment coordinates where ``θ_b[0] ≥ 0`` and
``D θ_b ≥ 0`` are hard bounds of every inner solve (issue #161; ADR 0005
#114 / #161 amendments).  A shift block (or a non-zero offset) makes the row
non-linear in ``γ``: auglag fits it through its nonlinear inequality block
(issue #115, ``tests/test_exponential_scaling_shift.py``), while SLSQP and
trust-constr still raise ``NotImplementedError`` (issue #116).

R reference (``reference/mlt_exponential_scaling_*.txt``, generated in
``reference/generate_reference.R``): ``mlt(ctm(Bernstein, scaling = ~x_s,
todistr = "Exponential"), fixed = c("Bs1(y)" = 0))``, polished by
equality-constrained Newton on its active set with mlt's own score and Hessian
(stationarity ~1e-14) and cross-checked against ``tram::Aareg(y ~ 0 | x_s)``.
The support row is active at the MLE, so pinning ``Bs1`` poses exactly
mltpy's problem.  Raw ``ctm`` coefficient order is ``[θ_b | γ]`` and ``γ``
needs no sign flip; with ``q_d = 0`` there is no β to negate.
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint

from mltpy import MLT
from mltpy.basis import BernsteinBasis, InteractionBasis
from mltpy.constraints import build_constraint_matrices, build_constraints
from mltpy.likelihood import log_likelihood, negative_log_likelihood
from mltpy.model import ConditionalTransformationModel
from mltpy.optimizer import OptimizerConfig, optimize
from mltpy.variables import CensoredData, CensoringType

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"

_ORDER = 4
_P = _ORDER + 1
# Feasibility tolerance of the support rows (AugLagOptions.feas_tol; the
# predict-side check uses the same value, model._EXPON_SUPPORT_TOL).
_FEAS_TOL = 1e-7
_SUPPORT_MSG = "h(a|x) < 0"
_SOLVERS = ("auglag", "slsqp", "trust-constr")

# Tolerances against the R fixture.  logLik is held to CLAUDE.md's rtol=1e-6
# by every solver.  On θ, the support row θ_b[0] ≥ 0 and the monotonicity row
# θ_b[2] = θ_b[3] are both active at the optimum (ADR 0005 Decision 4: "θ is
# compared at the active-boundary tolerance").  The R reference itself is
# polished to stationarity ~1e-14, and auglag meets CLAUDE.md's rtol=1e-6
# with a 1e-6 absolute floor for the coefficients that sit at 0 (θ_b[0]) or
# on the flat Bs4/Bs5 direction.  SLSQP and trust-constr stop earlier along
# that flat direction (SEs ≈ 0.2-1.7, research/exp-scaling-r-route §10), so
# they are compared at 1e-4.
_LL_RTOL = 1e-6
_THETA_TOL = {
    "auglag": {"rtol": 1e-6, "atol": 1e-6},
    "slsqp": {"rtol": 1e-4, "atol": 1e-4},
    "trust-constr": {"rtol": 1e-4, "atol": 1e-4},
}


def _load(name: str) -> NDArray[np.float64]:
    return np.loadtxt(REF_DIR / name, dtype=np.float64)


@pytest.fixture(scope="module")
def ref() -> dict[str, object]:
    """The R fixture ``mlt_exponential_scaling_*``."""
    theta = _load("mlt_exponential_scaling_theta.txt")
    support = _load("mlt_exponential_scaling_support.txt")
    return {
        "y": _load("mlt_exponential_scaling_y.txt"),
        "x_s": _load("mlt_exponential_scaling_xs.txt").reshape(-1, 1),
        "support": (float(support[0]), float(support[1])),
        "theta_b": theta[:_P],
        "gamma": theta[_P:],
        "ll": float(_load("mlt_exponential_scaling_ll.txt")),
    }


def _fit(
    ref: dict[str, object],
    *,
    solver: str = "auglag",
    X: NDArray[np.float64] | None = None,
    offset: NDArray[np.float64] | None = None,
) -> MLT:
    model = MLT(
        order=_ORDER,
        support=ref["support"],  # type: ignore[arg-type]
        base_distribution="exponential",
        scaling=ref["x_s"],  # type: ignore[arg-type]
        optimizer_config=OptimizerConfig(solver=solver, random_state=0),
    )
    with warnings.catch_warnings():
        # ConvergenceWarning on the degenerate active set is informational
        # (CLAUDE.md, Solver dispatch); feasibility and parity are asserted.
        warnings.simplefilter("ignore")
        model.fit(ref["y"], X=X, offset=offset)  # type: ignore[arg-type]
    return model


@pytest.fixture(scope="module", params=_SOLVERS)
def fitted(request: pytest.FixtureRequest, ref: dict[str, object]) -> MLT:
    model = _fit(ref, solver=request.param)
    model._test_solver = request.param  # type: ignore[attr-defined]
    return model


@pytest.fixture(scope="module")
def fitted_auglag(ref: dict[str, object]) -> MLT:
    return _fit(ref)


def _h_grid(
    model: MLT, ref: dict[str, object], n_grid: int = 201
) -> NDArray[np.float64]:
    """``h(y|x_i)`` on a support grid for every training row, shape (n, n_grid)."""
    a, b = ref["support"]  # type: ignore[misc]
    grid = np.linspace(a, b, n_grid)
    theta_b = model.theta_[:_P]
    h0 = model.basis.evaluate(grid) @ theta_b  # (n_grid,)
    f = np.exp(0.5 * (ref["x_s"] @ model.gamma_))  # type: ignore[operator]
    return np.asarray(f[:, None] * h0[None, :], dtype=np.float64)


# ---------------------------------------------------------------------------
# Guards: the combination is accepted without a shift block only
# ---------------------------------------------------------------------------


class TestGuards:
    """``MLT.__init__`` and ``optimize()`` reject only the non-linear cases.

    Since issue #115 the default auglag solver fits the non-linear cases (a
    shift block or a non-zero offset); the rejection tests below therefore
    run on SLSQP / trust-constr, which still raise (issue #116).
    """

    def test_init_accepts_scale_only(self, ref: dict[str, object]) -> None:
        model = MLT(
            order=_ORDER,
            support=ref["support"],  # type: ignore[arg-type]
            base_distribution="exponential",
            scaling=ref["x_s"],  # type: ignore[arg-type]
        )
        assert model.scaling is not None

    @pytest.mark.parametrize("solver", ["slsqp", "trust-constr"])
    @pytest.mark.parametrize("q_d", [1, 2])
    def test_fit_with_shift_block_raises(
        self, ref: dict[str, object], q_d: int, solver: str
    ) -> None:
        y = ref["y"]
        X = np.random.default_rng(q_d).normal(size=(len(y), q_d))  # type: ignore[arg-type]
        model = MLT(
            order=_ORDER,
            support=ref["support"],  # type: ignore[arg-type]
            base_distribution="exponential",
            scaling=ref["x_s"],  # type: ignore[arg-type]
            optimizer_config=OptimizerConfig(solver=solver),  # type: ignore[arg-type]
        )
        with pytest.raises(NotImplementedError, match="shift") as exc:
            model.fit(y, X=X)  # type: ignore[arg-type]
        assert "0005" in str(exc.value)
        assert "#116" in str(exc.value)
        assert not model.is_fitted_

    @pytest.mark.parametrize("solver", ["slsqp", "trust-constr"])
    def test_optimize_with_shift_block_raises(
        self, ref: dict[str, object], solver: str
    ) -> None:
        y = ref["y"]
        X = np.ones((len(y), 1))  # type: ignore[arg-type]
        with pytest.raises(NotImplementedError, match="shift"):
            optimize(
                BernsteinBasis(order=_ORDER, support=ref["support"]),  # type: ignore[arg-type]
                y,  # type: ignore[arg-type]
                X,
                base_distribution="exponential",
                scaling=ref["x_s"],  # type: ignore[arg-type]
                config=OptimizerConfig(solver=solver),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("solver", ["slsqp", "trust-constr"])
    @pytest.mark.parametrize("value", [0.3, -0.3])
    def test_fit_with_nonzero_offset_raises(
        self, ref: dict[str, object], value: float, solver: str
    ) -> None:
        """``θ_b[0]·f_i ≥ −offset_i`` is non-linear in γ for a non-zero offset."""
        y = ref["y"]
        offset = np.full(len(y), value)  # type: ignore[arg-type]
        model = MLT(
            order=_ORDER,
            support=ref["support"],  # type: ignore[arg-type]
            base_distribution="exponential",
            scaling=ref["x_s"],  # type: ignore[arg-type]
            optimizer_config=OptimizerConfig(solver=solver),  # type: ignore[arg-type]
        )
        with pytest.raises(NotImplementedError, match="offset"):
            model.fit(y, offset=offset)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [0.3, -0.3])
    def test_auglag_fits_nonzero_offset(
        self, ref: dict[str, object], value: float
    ) -> None:
        """auglag fits the non-linear rows (issue #115): h(a|x_i) ≥ 0."""
        y = ref["y"]
        offset = np.full(len(y), value)  # type: ignore[arg-type]
        model = _fit(ref, offset=offset)
        f = np.exp(0.5 * (ref["x_s"] @ model.gamma_))  # type: ignore[operator]
        assert np.min(model.theta_[0] * f + value) >= -_FEAS_TOL

    def test_zero_offset_matches_no_offset(
        self, ref: dict[str, object], fitted_auglag: MLT
    ) -> None:
        y = ref["y"]
        model = _fit(ref, offset=np.zeros(len(y)))  # type: ignore[arg-type]
        np.testing.assert_allclose(model.theta_, fitted_auglag.theta_, atol=1e-10)

    def test_interaction_basis_still_rejected(self, ref: dict[str, object]) -> None:
        basis = InteractionBasis(
            BernsteinBasis(order=2, support=ref["support"]),  # type: ignore[arg-type]
            BernsteinBasis(order=1, support=(-4.0, 4.0)),
        )
        with pytest.raises(ValueError, match="0003-scaling-with-interaction"):
            ConditionalTransformationModel(
                basis,
                base_distribution="exponential",
                scaling=ref["x_s"],  # type: ignore[arg-type]
            )

    def test_other_links_unaffected(self, ref: dict[str, object]) -> None:
        """A shift block with scaling still fits on a non-exponential link."""
        y = ref["y"]
        X = np.random.default_rng(1).normal(size=(len(y), 1))  # type: ignore[arg-type]
        model = MLT(
            order=_ORDER,
            support=ref["support"],  # type: ignore[arg-type]
            base_distribution="normal",
            scaling=ref["x_s"],  # type: ignore[arg-type]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y, X=X)  # type: ignore[arg-type]
        assert model.is_fitted_


# ---------------------------------------------------------------------------
# Constraint builders: the collapsed linear row with q_s γ columns
# ---------------------------------------------------------------------------


class TestConstraintRows:
    """``q_s`` pads the support row ``θ_b[0] ≥ 0`` with zero γ columns."""

    @pytest.mark.parametrize("q_s", [1, 3])
    def test_matrices_single_support_row(self, q_s: int) -> None:
        cm = build_constraint_matrices(
            _P, total_params=_P + q_s, nonneg_lower=True, q_s=q_s
        )
        assert cm.A_ineq.shape == (_P - 1 + 1, _P + q_s)
        expected = np.zeros(_P + q_s)
        expected[0] = 1.0
        np.testing.assert_array_equal(cm.A_ineq[-1], expected)
        np.testing.assert_array_equal(cm.b_ineq, np.zeros(_P))
        # Monotonicity rows leave γ free.
        np.testing.assert_array_equal(cm.A_ineq[:, _P:], 0.0)

    @pytest.mark.parametrize("solver", ["slsqp", "trust-constr"])
    def test_scipy_forms_single_support_row(self, solver: str) -> None:
        q_s = 2
        cons = build_constraints(
            _P,
            solver=solver,  # type: ignore[call-overload]
            total_params=_P + q_s,
            nonneg_lower=True,
            q_s=q_s,
        )
        theta = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 9.0, -9.0])
        if solver == "slsqp":
            assert cons[1]["fun"](theta).tolist() == [-0.5]
        else:
            lc = cons[1]
            assert isinstance(lc, LinearConstraint)
            assert (np.asarray(lc.A) @ theta).tolist() == [-0.5]
            assert np.asarray(lc.lb).tolist() == [0.0]

    def test_zero_offset_accepted(self) -> None:
        cm = build_constraint_matrices(
            _P, total_params=_P + 1, nonneg_lower=True, q_s=1, offset=np.zeros(4)
        )
        assert cm.b_ineq[-1] == 0.0

    def test_shift_block_with_scaling_needs_x_scale(self) -> None:
        """Issue #115: the auglag builder carries the shift-block rows as a
        nonlinear block, which needs the scaling design (formerly a
        ``NotImplementedError``; covered in ``test_exponential_scaling_shift``).
        """
        with pytest.raises(ValueError, match="X_scale"):
            build_constraint_matrices(
                _P,
                total_params=_P + 2,
                nonneg_lower=True,
                X=np.ones((3, 1)),
                q_s=1,
            )

    def test_nonzero_offset_with_scaling_rejected(self) -> None:
        with pytest.raises(NotImplementedError, match="offset"):
            build_constraints(
                _P,
                total_params=_P + 1,
                nonneg_lower=True,
                q_s=1,
                offset=np.array([0.0, 0.1]),
            )

    @pytest.mark.parametrize("q_s", [-1, 2.0])
    def test_invalid_q_s(self, q_s: object) -> None:
        with pytest.raises((ValueError, TypeError), match="q_s"):
            build_constraint_matrices(
                _P,
                total_params=_P + 1,
                nonneg_lower=True,
                q_s=q_s,  # type: ignore[arg-type]
            )

    def test_q_s_exceeding_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="q_s"):
            build_constraint_matrices(_P, total_params=_P, nonneg_lower=True, q_s=1)

    def test_q_s_ignored_without_nonneg_lower(self) -> None:
        """``q_s`` only shapes the support row; monotonicity is unchanged."""
        a = build_constraint_matrices(_P, total_params=_P + 1)
        b = build_constraint_matrices(_P, total_params=_P + 1, q_s=1)
        np.testing.assert_array_equal(a.A_ineq, b.A_ineq)


# ---------------------------------------------------------------------------
# R parity and feasibility
# ---------------------------------------------------------------------------


class TestRParity:
    """Fit matches ``mlt(fixed = c("Bs1(y)" = 0))`` (raw γ, no sign flip)."""

    def test_log_likelihood(self, fitted: MLT, ref: dict[str, object]) -> None:
        np.testing.assert_allclose(
            fitted.result_.log_likelihood, ref["ll"], rtol=_LL_RTOL, atol=1e-10
        )
        # mltpy's own likelihood at R's θ reproduces R's logLik.
        ll_at_r = log_likelihood(
            np.concatenate([ref["theta_b"], ref["gamma"]]),  # type: ignore[list-item]
            fitted.basis,
            ref["y"],  # type: ignore[arg-type]
            base_distribution="exponential",
            scaling=ref["x_s"],  # type: ignore[arg-type]
        )
        np.testing.assert_allclose(ll_at_r, ref["ll"], rtol=1e-12, atol=1e-10)

    def test_theta_b_and_gamma(self, fitted: MLT, ref: dict[str, object]) -> None:
        tol = _THETA_TOL[fitted._test_solver]  # type: ignore[attr-defined]
        np.testing.assert_allclose(fitted.theta_[:_P], ref["theta_b"], **tol)
        np.testing.assert_allclose(fitted.gamma_, ref["gamma"], **tol)

    def test_support_row_feasible(self, fitted: MLT, ref: dict[str, object]) -> None:
        """``min_y h(y|x_i) ≥ 0`` for every training row (ADR 0005 Decision 1)."""
        h = _h_grid(fitted, ref)
        assert h.min(axis=1).min() >= -_FEAS_TOL
        # The minimum is attained at the lower support end a.
        np.testing.assert_allclose(h.min(axis=1), h[:, 0], atol=0.0)
        # ... and h(y_i|x_i) > 0 at every observation.
        h_obs = fitted.predict(
            ref["y"],  # type: ignore[arg-type]
            what="trafo",
            X_scale_new=ref["x_s"],  # type: ignore[arg-type]
        )
        assert np.all(h_obs > 0.0)

    def test_monotone(self, fitted: MLT) -> None:
        assert np.all(np.diff(fitted.theta_[:_P]) >= -_FEAS_TOL)


# ---------------------------------------------------------------------------
# Fitted surface: finite accessors and predictions
# ---------------------------------------------------------------------------


_WHATS = (
    "trafo",
    "distribution",
    "survivor",
    "density",
    "logdensity",
    "hazard",
    "cumhazard",
)


class TestFittedSurface:
    def test_finite_accessors(self, fitted_auglag: MLT) -> None:
        assert fitted_auglag.theta_.shape == (_P + 1,)
        assert np.all(np.isfinite(fitted_auglag.theta_))
        assert fitted_auglag.gamma_.shape == (1,)
        assert np.all(np.isfinite(fitted_auglag.gamma_))
        assert np.isfinite(fitted_auglag.result_.log_likelihood)

    @pytest.mark.parametrize("what", _WHATS)
    def test_predictions_finite_and_silent(
        self, fitted_auglag: MLT, ref: dict[str, object], what: str
    ) -> None:
        """In-sample predictions are finite and the #157 warning stays silent."""
        y = ref["y"]
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=f".*{_SUPPORT_MSG}.*")
            out = fitted_auglag.predict(
                y,  # type: ignore[arg-type]
                what=what,  # type: ignore[arg-type]
                X_scale_new=ref["x_s"],  # type: ignore[arg-type]
            )
        assert out.shape == (len(y),)  # type: ignore[arg-type]
        assert np.all(np.isfinite(out))

    def test_cdf_matches_closed_form(
        self, fitted_auglag: MLT, ref: dict[str, object]
    ) -> None:
        """``F(y|x) = 1 − exp(−h_0(y)·f)`` and ``F(a|x) = 0`` (θ_b[0] = 0)."""
        y = ref["y"]
        x_s = ref["x_s"]
        h = fitted_auglag.predict(y, what="trafo", X_scale_new=x_s)  # type: ignore[arg-type]
        cdf = fitted_auglag.predict(y, what="distribution", X_scale_new=x_s)  # type: ignore[arg-type]
        np.testing.assert_allclose(cdf, -np.expm1(-h), rtol=1e-12)
        a = ref["support"][0]  # type: ignore[index]
        cdf_a = fitted_auglag.predict(
            np.full(len(y), a),  # type: ignore[arg-type]
            what="distribution",
            X_scale_new=x_s,  # type: ignore[arg-type]
        )
        np.testing.assert_allclose(cdf_a, 0.0, atol=1e-6)

    @pytest.mark.parametrize("p", [0.05, 0.5, 0.9])
    def test_quantile_round_trip(
        self, fitted_auglag: MLT, ref: dict[str, object], p: float
    ) -> None:
        x_s = np.array([[-1.0], [0.0], [1.0]])
        probs = np.full(3, p)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            q = fitted_auglag.predict(probs, what="quantile", X_scale_new=x_s)
        assert np.all(np.isfinite(q))
        cdf = fitted_auglag.predict(q, what="distribution", X_scale_new=x_s)
        np.testing.assert_allclose(cdf, probs, atol=1e-6)

    def test_scaling_changes_distribution(self, fitted_auglag: MLT) -> None:
        """γ > 0: larger x_s ⇒ larger h ⇒ stochastically smaller Y."""
        y = np.full(2, 0.5)
        x_s = np.array([[-1.0], [1.0]])
        cdf = fitted_auglag.predict(y, what="distribution", X_scale_new=x_s)
        assert fitted_auglag.gamma_[0] > 0.0
        assert cdf[1] > cdf[0]


# ---------------------------------------------------------------------------
# Censoring, edge cases, numerical stability
# ---------------------------------------------------------------------------


def _censored(ref: dict[str, object], censoring: CensoringType) -> CensoredData:
    """Censor ~30 % of the fixture rows (all rows for INTERVAL)."""
    y = np.asarray(ref["y"])
    cens = np.random.default_rng(114).uniform(size=len(y)) < 0.3
    if censoring is CensoringType.RIGHT:
        return CensoredData.right_censored(y, cens)
    if censoring is CensoringType.LEFT:
        return CensoredData.left_censored(y, cens)
    upper = ref["support"][1]  # type: ignore[index]
    return CensoredData.interval_censored(0.9 * y, np.minimum(1.1 * y, upper))


class TestCensoredAndEdgeCases:
    @pytest.mark.parametrize(
        "censoring",
        [CensoringType.RIGHT, CensoringType.LEFT, CensoringType.INTERVAL],
    )
    def test_censored_fit_agrees_across_solvers(
        self, ref: dict[str, object], censoring: CensoringType
    ) -> None:
        """The support row does not depend on y, so censoring changes nothing.

        No R reference here (the R fixture is exact data); auglag must agree
        with trust-constr, which keeps the linear rows satisfied throughout.
        """
        cd = _censored(ref, censoring)
        fits = {}
        for solver in ("auglag", "trust-constr"):
            model = MLT(
                order=_ORDER,
                support=ref["support"],  # type: ignore[arg-type]
                censoring=censoring,
                base_distribution="exponential",
                scaling=ref["x_s"],  # type: ignore[arg-type]
                optimizer_config=OptimizerConfig(solver=solver, random_state=0),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(cd)
            fits[solver] = model
        aug, tc = fits["auglag"], fits["trust-constr"]
        assert aug.result_.converged
        assert aug.theta_[0] >= -_FEAS_TOL
        assert np.all(np.diff(aug.theta_[:_P]) >= -_FEAS_TOL)
        np.testing.assert_allclose(
            aug.result_.log_likelihood, tc.result_.log_likelihood, rtol=1e-7
        )
        np.testing.assert_allclose(aug.theta_, tc.theta_, atol=1e-5)

    def test_extrapolated_x_s_stays_in_support(self, fitted_auglag: MLT) -> None:
        """Scale-only: ``h(a|x) = θ_b[0]·f(x) ≥ 0`` for every x, not just the hull.

        The #157 warning cannot fire on this path (it can once a shift block
        lands, #115).
        """
        # Training x_s spans about ±2.5; ±10 is well outside the hull.
        x_s = np.array([[-10.0], [0.0], [10.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fitted_auglag.predict(np.full(3, 0.5), what="distribution", X_scale_new=x_s)

    def test_support_multiplier_reported(self, fitted_auglag: MLT) -> None:
        """The boxed support row carries the bound multiplier in ``mu_ineq``.

        R's polished reference stores the KKT multipliers of its rows
        ``c(θ_b[0] ≥ 0, D θ_b ≥ 0)`` in ``mlt_exponential_scaling_mu.txt``
        (154.8 on the support row, 0.81 on θ_b[2] ≤ θ_b[3]).  mltpy orders
        the rows the other way round: monotonicity first, then support.
        Multipliers are estimated from the gradient at the solver's θ (Δθ
        ~3e-7 on a flat surface), hence the looser tolerance.
        """
        mu_r = _load("mlt_exponential_scaling_mu.txt")
        mu = fitted_auglag.result_.mu_ineq
        assert mu is not None
        np.testing.assert_allclose(mu, np.roll(mu_r, -1), rtol=1e-3, atol=1e-6)

    def test_multi_column_scaling(self, ref: dict[str, object]) -> None:
        y = ref["y"]
        x_s = np.column_stack(
            [ref["x_s"], np.random.default_rng(3).normal(size=len(y))]  # type: ignore[arg-type,list-item]
        )
        model = MLT(
            order=_ORDER,
            support=ref["support"],  # type: ignore[arg-type]
            base_distribution="exponential",
            scaling=x_s,
            optimizer_config=OptimizerConfig(random_state=0),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y)  # type: ignore[arg-type]
        assert model.gamma_.shape == (2,)
        assert np.all(np.isfinite(model.gamma_))
        assert model.theta_[0] >= -_FEAS_TOL
        # Adding a free parameter cannot lower the maximised log-likelihood.
        assert model.result_.log_likelihood >= ref["ll"] - 1e-6  # type: ignore[operator]

    def test_large_scaling_covariate(self, ref: dict[str, object]) -> None:
        """A wide x_s range (f_i spans many orders of magnitude) stays finite."""
        y = ref["y"]
        x_s = 10.0 * ref["x_s"]  # type: ignore[operator]
        model = MLT(
            order=_ORDER,
            support=ref["support"],  # type: ignore[arg-type]
            base_distribution="exponential",
            scaling=x_s,
            optimizer_config=OptimizerConfig(random_state=0),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y)  # type: ignore[arg-type]
        assert np.all(np.isfinite(model.theta_))
        assert np.isfinite(model.result_.log_likelihood)
        assert model.theta_[0] >= -_FEAS_TOL
        # Same model, reparametrised: γ' = γ / 10 at the same optimum.
        np.testing.assert_allclose(model.result_.log_likelihood, ref["ll"], rtol=1e-6)


# ---------------------------------------------------------------------------
# Censored rows with no exact row to hold h_0 monotone (issue #161)
# ---------------------------------------------------------------------------

_CENSORED_SOLVERS = ("auglag", "trust-constr", "slsqp")


def _fit_censored(
    cd: CensoredData,
    x_s: NDArray[np.float64],
    support: tuple[float, float],
    censoring: CensoringType,
) -> dict[str, MLT]:
    """Fit the scale-only exponential model with every solver."""
    fits = {}
    for solver in _CENSORED_SOLVERS:
        model = MLT(
            order=_ORDER,
            support=support,
            censoring=censoring,
            base_distribution="exponential",
            scaling=x_s,
            optimizer_config=OptimizerConfig(solver=solver, random_state=0),
        )
        with warnings.catch_warnings():
            # ConvergenceWarning is informational (CLAUDE.md, Solver dispatch).
            warnings.simplefilter("ignore")
            model.fit(cd)
        fits[solver] = model
    return fits


def _assert_auglag_matches_reference_solvers(fits: dict[str, MLT]) -> None:
    """auglag converges, stays feasible and reaches the best scipy optimum.

    trust-constr and SLSQP keep the linear rows satisfied at every iterate,
    so they are the reference here (no R fixture exists for censored data).
    The two stop at slightly different points along flat directions
    (logLik differences up to ~7e-7 relative), so auglag is compared with
    the better converged one at CLAUDE.md's rtol=1e-6.
    """
    aug = fits["auglag"]
    assert aug.result_.converged
    theta_b = aug.theta_[:_P]
    assert theta_b[0] >= -_FEAS_TOL
    assert np.all(np.diff(theta_b) >= -_FEAS_TOL)
    refs = [
        fits[s].result_.log_likelihood
        for s in ("trust-constr", "slsqp")
        if fits[s].result_.converged
    ]
    assert refs, "no converged reference solver"
    np.testing.assert_allclose(aug.result_.log_likelihood, max(refs), rtol=1e-6)


def _simulated(seed: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The issue #161 design: Exp(rate 2·exp(0.3·x_s)) responses, n = 150."""
    rng = np.random.default_rng(seed)
    x_s = rng.normal(size=(150, 1))
    y = rng.exponential(size=150) / (2.0 * np.exp(0.5 * 0.6 * x_s[:, 0]))
    return y, x_s


class TestCensoredOnlyRegion:
    """auglag on the #114 path with censored rows exact rows do not cover.

    A right-censored row contributes ``−log S = h(y_i|x_i)`` and no
    ``−log h'`` term.  If no exact observation sits near ``y_i``, nothing in
    the likelihood holds ``h_0`` monotone there: with ``D θ_b ≥ 0`` only a
    PHR row, ``h_0(y_i)`` drops below ``θ_b[0]`` and the NLL falls
    exponentially in ``γ`` through ``f_i = exp(0.5·x_s,i·γ)``.  Before the
    fix, auglag reported ``converged=False`` at logLik ≈ +2000 with
    ``min D θ_b ≈ −10`` on all nine designs below.  The scale-only path now
    solves in increment coordinates, where ``D θ_b ≥ 0`` is a hard bound
    (``optimizer._increment_auglag``, ADR 0005 #161 amendment).

    Design (the reproducer of #161): every row below the common time ``c``
    is censored at its own time, and the rows above ``c`` are exact at
    ``c``, so ``[0, c)`` holds censored rows only.
    """

    @pytest.mark.parametrize("q", [0.5, 0.8, 0.95])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_right_censored_below_exact(self, seed: int, q: float) -> None:
        y, x_s = _simulated(seed)
        c = float(np.quantile(y, q))
        cd = CensoredData.right_censored(np.minimum(y, c), y <= c)
        fits = _fit_censored(cd, x_s, (0.0, 1.01 * c), CensoringType.RIGHT)
        _assert_auglag_matches_reference_solvers(fits)

    def test_right_censored_below_exact_fixture(self, ref: dict[str, object]) -> None:
        """Same design on the R fixture's data (diverged to logLik +1868)."""
        y = np.asarray(ref["y"])
        c = float(np.quantile(y, 0.8))
        cd = CensoredData.right_censored(np.minimum(y, c), y <= c)
        fits = _fit_censored(
            cd,
            ref["x_s"],  # type: ignore[arg-type]
            (0.0, 1.01 * c),
            CensoringType.RIGHT,
        )
        _assert_auglag_matches_reference_solvers(fits)

    def test_monotonicity_multipliers_reported(self, ref: dict[str, object]) -> None:
        """Active monotonicity rows carry their bound multipliers in ``mu_ineq``.

        At this optimum ``θ_b[0] = … = θ_b[3] = 0 < θ_b[4]``: the support row
        and the first three monotonicity rows are active.  In increment coordinates
        those are bounds, so their force must come back through ``ν_lo``;
        stationarity in θ then reads ``∇nll = Aᵀ μ`` on the reported rows.
        """
        y = np.asarray(ref["y"])
        c = float(np.quantile(y, 0.8))
        cd = CensoredData.right_censored(np.minimum(y, c), y <= c)
        aug = _fit_censored(
            cd,
            ref["x_s"],  # type: ignore[arg-type]
            (0.0, 1.01 * c),
            CensoringType.RIGHT,
        )["auglag"]
        res = aug.result_
        assert res.mu_ineq is not None and res.constraint_A_ineq is not None
        assert np.all(res.mu_ineq >= 0.0)
        # Rows: D θ_b ≥ 0 (k = 0…3), then the support row θ_b[0] ≥ 0.  The
        # first three monotonicity rows and the support row are active; the
        # last monotonicity row (θ_b[4] > θ_b[3]) is not.
        assert np.all(res.mu_ineq[[0, 1, 2, 4]] > 1e-3)
        assert res.mu_ineq[3] == 0.0
        out = negative_log_likelihood(
            aug.theta_,
            aug.basis,
            cd,
            None,
            CensoringType.RIGHT,
            gradient=True,
            base_distribution="exponential",
            scaling=ref["x_s"],  # type: ignore[arg-type]
        )
        assert isinstance(out, tuple)
        grad = out[1]
        scale = max(1.0, float(np.max(np.abs(grad))))
        np.testing.assert_allclose(
            grad, res.constraint_A_ineq.T @ res.mu_ineq, atol=1e-4 * scale
        )


class TestTypeICensoring:
    """Administrative (Type-I) censoring at a common time, issue #161 item 4.

    Every row beyond the censoring time is censored *at* that time, and
    exact rows cover the rest of the support, so these designs never
    triggered the #161 divergence (auglag agreed before the fix as well).
    They pin the behaviour of the usual survival design on this path.
    """

    @pytest.mark.parametrize("q", [0.5, 0.8, 0.95])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_right(self, seed: int, q: float) -> None:
        y, x_s = _simulated(seed)
        c = float(np.quantile(y, q))
        cd = CensoredData.right_censored(np.minimum(y, c), y > c)
        fits = _fit_censored(cd, x_s, (0.0, 1.01 * c), CensoringType.RIGHT)
        _assert_auglag_matches_reference_solvers(fits)

    @pytest.mark.parametrize("q", [0.5, 0.8, 0.95])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_left(self, seed: int, q: float) -> None:
        y, x_s = _simulated(seed)
        c = float(np.quantile(y, 1.0 - q))
        cd = CensoredData.left_censored(np.maximum(y, c), y < c)
        fits = _fit_censored(cd, x_s, (0.0, float(y.max())), CensoringType.LEFT)
        _assert_auglag_matches_reference_solvers(fits)

    @pytest.mark.parametrize("q", [0.5, 0.8, 0.95])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_interval_inspection_grid(self, seed: int, q: float) -> None:
        """Four common inspection times; the last interval is open, (t_4, ∞)."""
        y, x_s = _simulated(seed)
        grid = np.quantile(y, np.linspace(0.0, q, 5)[1:])
        k = np.searchsorted(grid, y)
        lower = np.concatenate([[0.0], grid])[k]
        upper = np.concatenate([grid, [np.inf]])[k]
        cd = CensoredData.interval_censored(lower, upper)
        fits = _fit_censored(
            cd, x_s, (0.0, 1.01 * float(grid[-1])), CensoringType.INTERVAL
        )
        _assert_auglag_matches_reference_solvers(fits)

    def test_right_fixture(self, ref: dict[str, object]) -> None:
        y = np.asarray(ref["y"])
        c = float(np.quantile(y, 0.8))
        cd = CensoredData.right_censored(np.minimum(y, c), y > c)
        fits = _fit_censored(
            cd,
            ref["x_s"],  # type: ignore[arg-type]
            (0.0, 1.01 * c),
            CensoringType.RIGHT,
        )
        _assert_auglag_matches_reference_solvers(fits)
