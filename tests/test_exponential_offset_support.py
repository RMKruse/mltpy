"""Exponential-link support rule with an ``offset`` on the shift path (issue #160).

ADR 0005 Decision 1 fixes the lower-support-end rule ``h(a|x_i) ≥ 0`` for every
training row under ``base_distribution="exponential"``, where ``h`` is the
transformation as the likelihood evaluates it, offset included::

    h(a|x_i) = θ_b[0] + x_i·β + offset_i.

The shift-path support rows therefore read ``[1, 0, …, 0 | x_i]·θ ≥ −offset_i``
(no covariates: the single row ``θ_b[0] ≥ max_i(−offset_i)``), matching the
``InteractionBasis`` rows of #158.  Before the fix the right-hand side was
zero: a negative offset left the support on every training row (inflated
logLik, negative CDF near ``a``) and a positive one pinned the fit at a
spurious atom of mass ``1 − e^{−offset}``.

R reference: R ``mlt`` has no support rule on the exponential link, so the
fixture ``reference/mlt_exponential_offset_*.txt`` is an explicit-constraint
``alabama::auglag`` fit on mlt's likelihood with ``hin`` rows
``θ_b[0] + x_i·β + offset_i ≥ 0``, polished on its active set and
cross-checked against an exact LP + ``mlt(fixed=)`` decomposition (see
``reference/generate_reference.R``).
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint

from mltpy.basis import BernsteinBasis
from mltpy.constraints import build_constraint_matrices, build_constraints
from mltpy.likelihood import log_likelihood
from mltpy.model import ConditionalTransformationModel
from mltpy.optimizer import OptimizerConfig, _perturb_and_project

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"

_SUPPORT = (0.0, 1.0)
_ORDER = 3
_P = _ORDER + 1
# Feasibility tolerance of the support rows (AugLagOptions.feas_tol; the
# predict-side check uses the same value, model._EXPON_SUPPORT_TOL).
_FEAS_TOL = 1e-7
_SUPPORT_MSG = "h(a|x) < 0"
_SOLVERS = ("auglag", "slsqp", "trust-constr")

# A constant offset puts the shift fit on a maximally degenerate active set:
# at the optimum β = 0 and *every* support row θ_b[0] + x_i·β ≥ −c is active
# (LICQ fails; the same geometry as offset=None).  auglag resolves it to ~1e-9.
# trust-constr is an interior-point method and stops with ~1e-6 slack on the
# rows, which also costs it ~1e-5 relative in logLik.  SLSQP can end on
# "Positive directional derivative for linesearch" (reported converged=False)
# a few 1e-8 off the boundary.
_ACTIVE_TOL = {"auglag": 1e-6, "slsqp": 1e-6, "trust-constr": 1e-5}
_LL_RTOL = {"auglag": 1e-6, "slsqp": 1e-5, "trust-constr": 1e-4}
# θ against the R fixture (non-degenerate: two support rows + one
# monotonicity row active).  auglag meets CLAUDE.md's atol=1e-10 (measured
# ~9e-11); SLSQP stops ~3e-6 short along the flat y-shape direction and
# trust-constr keeps ~1e-7 interior slack, both at logLik rtol=1e-6.
_THETA_ATOL = {"auglag": 1e-10, "slsqp": 1e-5, "trust-constr": 1e-6}


def _repro_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The reproduction from issue #160."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, 200)
    y = rng.uniform(0.05, 0.95, 200)
    return x, y


def _fit(
    y: NDArray[np.float64],
    X: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
    *,
    solver: str = "auglag",
    fixed_params: dict[int, float] | None = None,
) -> ConditionalTransformationModel:
    model = ConditionalTransformationModel(
        BernsteinBasis(order=_ORDER, support=_SUPPORT),
        base_distribution="exponential",
        optimizer_config=OptimizerConfig(
            solver=solver, random_state=0, fixed_params=fixed_params
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(y, X=X, offset=offset)
    return model


def _h_at_lower_end(
    model: ConditionalTransformationModel,
    X: NDArray[np.float64] | None,
    offset: NDArray[np.float64] | None,
    n: int,
) -> NDArray[np.float64]:
    """``h(a|x_i) = θ_b[0] + x_i·β + offset_i`` per training row."""
    assert model.theta_ is not None
    h_a = np.full(n, model.theta_[0])
    if X is not None:
        h_a = h_a + X @ model.theta_[_P:]
    return h_a if offset is None else h_a + offset


# ---------------------------------------------------------------------------
# Constraint builders
# ---------------------------------------------------------------------------


_X3 = np.array([[0.1], [0.5], [0.9]])
_OFF3 = np.array([0.5, -1.0, 0.0])


class TestSupportRowsRhs:
    """The support rows carry ``b = −offset`` on all three constraint forms."""

    @pytest.mark.parametrize(
        "offset", [_OFF3, np.zeros(3), np.array([1e6, -1e6, 1e-12])]
    )
    def test_auglag_rhs_is_minus_offset(self, offset: NDArray[np.float64]) -> None:
        cm = build_constraint_matrices(
            _P, total_params=_P + 1, nonneg_lower=True, X=_X3, offset=offset
        )
        np.testing.assert_array_equal(cm.b_ineq[: _P - 1], 0.0)
        np.testing.assert_array_equal(cm.b_ineq[_P - 1 :], -offset)
        # Rows themselves are unchanged by the offset.
        np.testing.assert_array_equal(cm.A_ineq[_P - 1 :, 0], 1.0)
        np.testing.assert_array_equal(cm.A_ineq[_P - 1 :, _P:], _X3)

    @pytest.mark.parametrize(
        ("offset", "rhs"),
        [
            (np.array([0.5, -1.0, 0.0]), 1.0),
            (np.array([0.5, 0.25]), -0.25),
            (np.array([-2.0]), 2.0),
        ],
    )
    @pytest.mark.parametrize("X", [None, np.zeros((3, 0))], ids=["none", "zero-col"])
    def test_no_covariates_single_row_max(
        self,
        offset: NDArray[np.float64],
        rhs: float,
        X: NDArray[np.float64] | None,
    ) -> None:
        if X is not None:
            X = np.zeros((offset.shape[0], 0))
        cm = build_constraint_matrices(
            _P, total_params=_P, nonneg_lower=True, X=X, offset=offset
        )
        assert cm.A_ineq.shape == (_P, _P)
        np.testing.assert_array_equal(cm.A_ineq[-1], np.eye(_P)[0])
        assert cm.b_ineq[-1] == rhs

    @pytest.mark.parametrize("X", [None, _X3], ids=["no-cov", "cov"])
    def test_offset_none_and_zero_are_zero_rhs(
        self, X: NDArray[np.float64] | None
    ) -> None:
        total = _P + (0 if X is None else 1)
        base = build_constraint_matrices(_P, total_params=total, nonneg_lower=True, X=X)
        zero = build_constraint_matrices(
            _P, total_params=total, nonneg_lower=True, X=X, offset=np.zeros(3)
        )
        np.testing.assert_array_equal(base.b_ineq, 0.0)
        np.testing.assert_array_equal(zero.b_ineq, base.b_ineq)
        np.testing.assert_array_equal(zero.A_ineq, base.A_ineq)

    def test_offset_ignored_without_nonneg_lower(self) -> None:
        cm = build_constraint_matrices(_P, offset=_OFF3)
        assert cm.A_ineq.shape == (_P - 1, _P)
        np.testing.assert_array_equal(cm.b_ineq, 0.0)

    def test_slsqp_support_fun_subtracts_rhs(self) -> None:
        cons = build_constraints(
            _P,
            solver="slsqp",
            total_params=_P + 1,
            nonneg_lower=True,
            X=_X3,
            offset=_OFF3,
        )
        support = cons[1]
        theta = np.array([0.2, 0.4, 0.6, 0.8, 0.5])
        h_a = theta[0] + _X3[:, 0] * theta[_P] + _OFF3
        np.testing.assert_allclose(support["fun"](theta), h_a, rtol=0, atol=1e-15)
        jac = support["jac"](theta)
        np.testing.assert_array_equal(jac[:, 0], 1.0)
        np.testing.assert_array_equal(jac[:, _P:], _X3)

    def test_trust_constr_lb_is_minus_offset(self) -> None:
        cons = build_constraints(
            _P,
            solver="trust-constr",
            total_params=_P + 1,
            nonneg_lower=True,
            X=_X3,
            offset=_OFF3,
        )
        support = cons[1]
        assert isinstance(support, LinearConstraint)
        np.testing.assert_array_equal(support.lb, -_OFF3)
        assert np.all(np.isinf(support.ub))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"X": _X3, "offset": np.zeros(2)}, "offset"),
            ({"X": None, "offset": np.zeros(0)}, "offset"),
            ({"X": _X3, "offset": np.array([0.0, np.nan, 0.0])}, "offset"),
            ({"X": _X3, "offset": np.array([0.0, np.inf, 0.0])}, "offset"),
            ({"X": None, "offset": np.array([-np.inf])}, "offset"),
        ],
    )
    def test_invalid_offset_raises(self, kwargs: dict[str, object], match: str) -> None:
        total = _P + (0 if kwargs["X"] is None else 1)
        with pytest.raises(ValueError, match=match):
            build_constraint_matrices(
                _P,
                total_params=total,
                nonneg_lower=True,
                **kwargs,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match=match):
            build_constraints(
                _P,
                total_params=total,
                nonneg_lower=True,
                **kwargs,  # type: ignore[arg-type]
            )


class TestPerturbFloor:
    """Restart points are lifted onto ``θ_b[0] ≥ support_floor``."""

    @pytest.mark.parametrize("floor", [0.0, 0.7, 25.0])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_lifted_to_floor(self, floor: float, seed: int) -> None:
        theta = np.array([-3.0, -1.0, 0.0, 1.0, 0.4])
        out = _perturb_and_project(
            theta,
            _P,
            np.random.default_rng(seed),
            nonneg_lower=True,
            support_floor=floor,
        )
        assert out[0] >= floor
        assert np.all(np.diff(out[:_P]) >= 0.0)
        assert out[_P] == theta[_P]  # β untouched

    def test_floor_ignored_without_nonneg_lower(self) -> None:
        theta = np.array([-3.0, -2.0, -1.0, 0.0])
        out = _perturb_and_project(
            theta, _P, np.random.default_rng(0), support_floor=5.0
        )
        assert out[0] < 0.0


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def repro_reference() -> ConditionalTransformationModel:
    """The reproduction fit without an offset, against which shifts compare."""
    x, y = _repro_data()
    return _fit(y, x[:, None], None)


class TestReproduction:
    """Issue #160 reproduction, both signs, every solver.

    A constant offset ``c`` is absorbed exactly by shifting every Bernstein
    coefficient by ``−c`` (``h'`` and the monotonicity rows only see
    differences), so the offset fit is the offset-free fit with
    ``θ_b ↦ θ_b − c`` at the same log-likelihood.
    """

    @pytest.mark.parametrize("solver", _SOLVERS)
    @pytest.mark.parametrize("off_val", [-0.5, 0.5])
    def test_constant_offset_shifts_theta_b(
        self,
        solver: str,
        off_val: float,
        repro_reference: ConditionalTransformationModel,
    ) -> None:
        x, y = _repro_data()
        X = x[:, None]
        off = np.full(y.shape[0], off_val)
        model = _fit(y, X, off, solver=solver)
        assert model.theta_ is not None and repro_reference.theta_ is not None

        h_a = _h_at_lower_end(model, X, off, y.shape[0])
        assert h_a.min() >= -_FEAS_TOL
        # Active, not pinned at the old zero-RHS rule (h(a|x_i) = offset_i).
        assert h_a.min() <= _ACTIVE_TOL[solver]

        np.testing.assert_allclose(
            model.result_.log_likelihood,
            repro_reference.result_.log_likelihood,
            rtol=_LL_RTOL[solver],
        )
        np.testing.assert_allclose(
            model.theta_[:_P] + off_val,
            repro_reference.theta_[:_P],
            rtol=0,
            atol=1e-4,
        )
        np.testing.assert_allclose(model.theta_[_P:], 0.0, atol=1e-6)

    @pytest.mark.parametrize("solver", _SOLVERS)
    @pytest.mark.parametrize("off_val", [-0.5, 0.5])
    def test_in_sample_predict_is_silent(self, solver: str, off_val: float) -> None:
        x, y = _repro_data()
        X = x[:, None]
        off = np.full(y.shape[0], off_val)
        model = _fit(y, X, off, solver=solver)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            cdf = model.predict(
                np.zeros(y.shape[0]),
                X_new=X,
                offset_new=off,
                what="distribution",
            )
        assert not [w for w in record if _SUPPORT_MSG in str(w.message)]
        assert np.all(np.diag(np.atleast_2d(cdf)) >= -_FEAS_TOL)


class TestNoCovariates:
    """Without covariates the rule is ``θ_b[0] ≥ max_i(−offset_i)``."""

    @pytest.mark.parametrize("solver", _SOLVERS)
    @pytest.mark.parametrize(
        "offset_kind", ["varying", "constant-negative", "constant-positive"]
    )
    def test_theta0_on_floor(self, solver: str, offset_kind: str) -> None:
        _, y = _repro_data()
        n = y.shape[0]
        offset = {
            "varying": np.random.default_rng(1).normal(0.0, 0.4, n),
            "constant-negative": np.full(n, -0.3),
            "constant-positive": np.full(n, 0.3),
        }[offset_kind]
        model = _fit(y, None, offset, solver=solver)
        free = _fit(y, None, None, solver=solver)
        assert model.theta_ is not None and free.theta_ is not None
        floor = float(np.max(-offset))
        # The likelihood pushes θ_b[0] down, so the single support row is
        # active at max_i(−offset_i); the y-shape (differences) is unchanged.
        assert model.theta_[0] >= floor - _FEAS_TOL
        assert model.theta_[0] <= floor + 1e-6
        np.testing.assert_allclose(
            np.diff(model.theta_), np.diff(free.theta_), rtol=0, atol=1e-4
        )


class TestPerRowOffsetRReference:
    """Per-row offset + covariate against the explicit-constraint R fixture."""

    @staticmethod
    def _load() -> dict[str, NDArray[np.float64]]:
        kinds = ("y", "x", "offset", "theta", "ll")
        paths = {k: REF_DIR / f"mlt_exponential_offset_{k}.txt" for k in kinds}
        if not all(p.exists() for p in paths.values()):
            pytest.skip(
                "mlt_exponential_offset_* reference files not yet generated — "
                "run Rscript reference/generate_reference.R"
            )
        return {k: np.atleast_1d(np.loadtxt(p)) for k, p in paths.items()}

    def test_loglik_formula_matches_r(self) -> None:
        ref = self._load()
        ll = log_likelihood(
            ref["theta"],
            BernsteinBasis(order=_ORDER, support=_SUPPORT),
            ref["y"],
            X=ref["x"][:, None],
            base_distribution="exponential",
            offset=ref["offset"],
        )
        np.testing.assert_allclose(ll, ref["ll"][0], rtol=1e-6, atol=1e-10)

    @pytest.mark.parametrize("solver", _SOLVERS)
    def test_fit_reaches_r_optimum(self, solver: str) -> None:
        ref = self._load()
        X = ref["x"][:, None]
        model = _fit(ref["y"], X, ref["offset"], solver=solver)
        assert model.theta_ is not None

        h_a = _h_at_lower_end(model, X, ref["offset"], ref["y"].shape[0])
        assert h_a.min() >= -_FEAS_TOL
        np.testing.assert_allclose(
            model.result_.log_likelihood, ref["ll"][0], rtol=1e-6, atol=1e-10
        )
        np.testing.assert_allclose(
            model.theta_, ref["theta"], rtol=1e-6, atol=_THETA_ATOL[solver]
        )

    def test_auglag_converges(self) -> None:
        ref = self._load()
        model = _fit(ref["y"], ref["x"][:, None], ref["offset"])
        assert model.result_.converged


class TestOffsetNoneUnchanged:
    """``offset=None`` and ``offset=0`` give the same fit as before."""

    @pytest.mark.parametrize("solver", _SOLVERS)
    def test_zero_offset_equals_none(self, solver: str) -> None:
        x, y = _repro_data()
        X = x[:, None]
        none = _fit(y, X, None, solver=solver)
        zero = _fit(y, X, np.zeros(y.shape[0]), solver=solver)
        assert none.theta_ is not None and zero.theta_ is not None
        np.testing.assert_allclose(zero.theta_, none.theta_, rtol=0, atol=1e-12)
        assert zero.result_.log_likelihood == pytest.approx(
            none.result_.log_likelihood, rel=1e-12
        )
        assert _h_at_lower_end(none, X, None, y.shape[0]).min() >= -_FEAS_TOL


class TestEdgeCases:
    @pytest.mark.parametrize("solver", _SOLVERS)
    def test_fixed_beta_with_offset(self, solver: str) -> None:
        """``fixed_params`` reductions keep the shifted right-hand side."""
        x, y = _repro_data()
        X = x[:, None]
        off = np.random.default_rng(2).normal(0.0, 0.3, y.shape[0])
        model = _fit(y, X, off, solver=solver, fixed_params={_P: 0.4})
        assert model.theta_ is not None
        assert model.theta_[_P] == pytest.approx(0.4, abs=1e-8)
        h_a = _h_at_lower_end(model, X, off, y.shape[0])
        assert h_a.min() >= -_FEAS_TOL
        assert h_a.min() <= 1e-6

    @pytest.mark.parametrize("solver", _SOLVERS)
    def test_single_observation(self, solver: str) -> None:
        model = _fit(np.array([0.4]), None, np.array([-0.8]), solver=solver)
        assert model.theta_ is not None
        assert model.theta_[0] >= 0.8 - _FEAS_TOL

    @pytest.mark.parametrize("solver", _SOLVERS)
    @pytest.mark.parametrize("off_val", [-25.0, 25.0])
    def test_large_offset(self, solver: str, off_val: float) -> None:
        """|offset| near the h clip still lands on the support boundary.

        The start sits 25 away from the boundary on the degenerate active set
        (see ``_ACTIVE_TOL``).  SLSQP then ends on its line-search failure and
        flags ``converged=False``; its best restart is within 1e-5 of the
        boundary rather than 1e-7.
        """
        x, y = _repro_data()
        X = x[:, None]
        off = np.full(y.shape[0], off_val)
        model = _fit(y, X, off, solver=solver)
        h_a = _h_at_lower_end(model, X, off, y.shape[0])
        feas_tol = _FEAS_TOL if model.result_.converged else 1e-5
        assert h_a.min() >= -feas_tol
        assert h_a.min() <= _ACTIVE_TOL[solver]
