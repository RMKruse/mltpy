"""Exponential-link support rule on the ``InteractionBasis`` path (issue #158).

ADR 0005 Decision 1 fixes the lower-support-end rule ``h(a|x_i) ≥ 0`` for every
training row under ``base_distribution="exponential"``.  On the interaction path
``h(a|x) = (a(a) ⊗ b(x))ᵀ vec(Θ) + offset``; for a Bernstein y-basis
``a(a) = e_0`` so ``h(a|x) = Θ[0, :] · b(x)``.  Without the support rows the
exact-data log-likelihood ``Σ −h(y_i|x_i) + log h'(y_i|x_i)`` is unbounded and
the fit runs off to ``h → −∞`` (the issue's reproduction).

Because ``b(x)`` is non-linear in ``x``, the training rows do **not** guarantee
``h(a|x) ≥ 0`` between training points, so ``predict`` runs the #157 support
check on this path too, and quantiles below the support atom use the
support-atom wording of ``_warn_quantile_saturation``.

R parity: R ``mlt`` enforces no support rule on the exponential link (ADR 0005,
Decision 1 — a recorded divergence), so there is no R fixture for these fits;
the tests check the constraint itself, its KKT-level consequences and the
predict-side contract.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray

from mltpy.basis import (
    BernsteinBasis,
    InteractionBasis,
    InterceptBasis,
    OneHotBasis,
)
from mltpy.constraints import build_constraint_matrices_interaction
from mltpy.model import ConditionalTransformationModel
from mltpy.optimizer import OptimizerConfig

_SUPPORT = (0.0, 1.0)
# Feasibility tolerance of the support rows (AugLagOptions.feas_tol; the
# predict-side check uses the same value, model._EXPON_SUPPORT_TOL).
_FEAS_TOL = 1e-7
_SUPPORT_MSG = "h(a|x) < 0"
_ATOM_MSG = "support atom"
_SOLVERS = ("auglag", "slsqp", "trust-constr")


def _basis(y_order: int = 3, x_order: int = 2) -> InteractionBasis:
    return InteractionBasis(
        BernsteinBasis(order=y_order, support=_SUPPORT),
        BernsteinBasis(order=x_order, support=_SUPPORT),
    )


def _repro_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The reproduction from issue #158."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, 200)
    y = rng.uniform(0.05, 0.95, 200)
    return x, y


def _h_at_lower_end(
    model: ConditionalTransformationModel,
    x: NDArray[np.float64],
    offset: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """``h(a|x) = (a(a) ⊗ b(x))ᵀ vec(Θ) + offset`` evaluated from the design."""
    assert isinstance(model.basis, InteractionBasis)
    assert model.theta_ is not None
    a = model.basis.support[0]
    design = model.basis.evaluate(np.full(x.shape[0], a), x)
    h_a = design @ model.theta_[: model.basis.n_params]
    return h_a if offset is None else h_a + offset


def _fit(
    y: NDArray[np.float64],
    x: NDArray[np.float64],
    *,
    solver: str = "auglag",
    offset: NDArray[np.float64] | None = None,
    basis: InteractionBasis | None = None,
) -> ConditionalTransformationModel:
    model = ConditionalTransformationModel(
        basis if basis is not None else _basis(),
        base_distribution="exponential",
        optimizer_config=OptimizerConfig(solver=solver, random_state=0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(y, X=x[:, None], offset=offset)
    return model


@pytest.fixture(scope="module", params=_SOLVERS)
def repro_fit(
    request: pytest.FixtureRequest,
) -> tuple[ConditionalTransformationModel, NDArray[np.float64], NDArray[np.float64]]:
    x, y = _repro_data()
    return _fit(y, x, solver=request.param), x, y


@pytest.fixture(scope="module")
def auglag_fit() -> ConditionalTransformationModel:
    x, y = _repro_data()
    return _fit(y, x)


def _with_theta(
    model: ConditionalTransformationModel, Theta: NDArray[np.float64]
) -> ConditionalTransformationModel:
    """Copy of a fitted model with a hand-set ``Θ`` (row-major ``vec_C``)."""
    out = copy.deepcopy(model)
    out.theta_ = np.asarray(Theta, dtype=np.float64).ravel()
    return out


def _messages(record: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in record if issubclass(w.category, UserWarning)]


# ---------------------------------------------------------------------------
# Constraint builder
# ---------------------------------------------------------------------------


class TestSupportRows:
    """``build_constraint_matrices_interaction(..., nonneg_lower=True)``."""

    @pytest.mark.parametrize(
        ("x_basis", "x"),
        [
            (BernsteinBasis(order=1, support=_SUPPORT), np.array([0.0, 0.3, 1.0])),
            (BernsteinBasis(order=3, support=_SUPPORT), np.array([0.2, 0.5, 0.9])),
            (OneHotBasis(K=3), np.array([0.0, 2.0, 1.0, 2.0])),
            (InterceptBasis(support=_SUPPORT), np.array([0.0, 1.0])),
        ],
    )
    def test_rows_are_e0_kron_bx(self, x_basis: object, x: NDArray[np.float64]) -> None:
        basis = InteractionBasis(BernsteinBasis(order=3, support=_SUPPORT), x_basis)
        p, q = basis.n_y_params, basis.n_x_params
        cm = build_constraint_matrices_interaction(basis, nonneg_lower=True, X=x)
        n_mono = (p - 1) * q
        assert cm.A_ineq.shape == (n_mono + x.shape[0], p * q)
        support = cm.A_ineq[n_mono:]
        # Θ[0, j] sits at vec_C index j, so row i is [b(x_i) | 0 … 0].
        expected = np.zeros((x.shape[0], p * q))
        expected[:, :q] = basis.x_basis.evaluate(x)
        np.testing.assert_allclose(support, expected, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(cm.b_ineq, np.zeros(n_mono + x.shape[0]))

    @pytest.mark.parametrize("offset", [np.array([0.5, -1.0, 0.0]), np.zeros(3)])
    def test_rhs_is_minus_offset(self, offset: NDArray[np.float64]) -> None:
        basis = _basis()
        x = np.array([0.1, 0.5, 0.9])
        cm = build_constraint_matrices_interaction(
            basis, nonneg_lower=True, X=x, offset=offset
        )
        n_mono = (basis.n_y_params - 1) * basis.n_x_params
        np.testing.assert_array_equal(cm.b_ineq[:n_mono], 0.0)
        np.testing.assert_array_equal(cm.b_ineq[n_mono:], -offset)

    def test_default_has_monotonicity_rows_only(self) -> None:
        basis = _basis()
        cm = build_constraint_matrices_interaction(basis)
        assert cm.A_ineq.shape == (
            (basis.n_y_params - 1) * basis.n_x_params,
            basis.n_params,
        )

    def test_single_observation(self) -> None:
        basis = _basis()
        cm = build_constraint_matrices_interaction(
            basis, nonneg_lower=True, X=np.array([[0.5]])
        )
        n_mono = (basis.n_y_params - 1) * basis.n_x_params
        assert cm.A_ineq.shape[0] == n_mono + 1

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"nonneg_lower": True}, "X must be provided"),
            (
                {"nonneg_lower": True, "X": np.array([0.5]), "q_s": 1},
                "scaling",
            ),
            (
                {
                    "nonneg_lower": True,
                    "X": np.array([0.1, 0.5]),
                    "offset": np.zeros(3),
                },
                "offset",
            ),
        ],
    )
    def test_invalid_inputs_raise(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            build_constraint_matrices_interaction(_basis(), **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


class TestFitRespectsSupport:
    """The issue reproduction no longer diverges, for every solver."""

    def test_training_rows_on_support(
        self,
        repro_fit: tuple[
            ConditionalTransformationModel, NDArray[np.float64], NDArray[np.float64]
        ],
    ) -> None:
        model, x, _ = repro_fit
        assert np.all(_h_at_lower_end(model, x) >= -_FEAS_TOL)

    def test_columns_monotone_and_bounded(
        self,
        repro_fit: tuple[
            ConditionalTransformationModel, NDArray[np.float64], NDArray[np.float64]
        ],
    ) -> None:
        model, _, _ = repro_fit
        Theta = model.Theta_
        assert Theta is not None
        assert np.all(np.diff(Theta, axis=0) >= -_FEAS_TOL)
        # The divergent fit reached |Θ| ~ 3e4 and logLik ~ +8e3.
        assert np.all(np.abs(Theta) < 50.0)
        assert np.isfinite(model.result_.log_likelihood)  # type: ignore[union-attr]
        assert model.result_.log_likelihood < 500.0  # type: ignore[union-attr]

    def test_loglik_matches_score(
        self,
        repro_fit: tuple[
            ConditionalTransformationModel, NDArray[np.float64], NDArray[np.float64]
        ],
    ) -> None:
        model, x, y = repro_fit
        np.testing.assert_allclose(
            model.result_.log_likelihood,  # type: ignore[union-attr]
            model.score(y, x[:, None]),
            rtol=1e-10,
        )

    def test_auglag_converges(self, auglag_fit: ConditionalTransformationModel) -> None:
        assert auglag_fit.result_ is not None
        assert auglag_fit.result_.converged

    def test_solvers_agree(self) -> None:
        x, y = _repro_data()
        lls = [_fit(y, x, solver=s).result_.log_likelihood for s in _SOLVERS]  # type: ignore[union-attr]
        np.testing.assert_allclose(lls, lls[0], rtol=0, atol=1e-3)

    @pytest.mark.parametrize("solver", _SOLVERS)
    def test_offset_enters_support_rule(self, solver: str) -> None:
        x, y = _repro_data()
        offset = np.where(x < 0.5, 0.3, -0.2)
        model = _fit(y, x, solver=solver, offset=offset)
        assert np.all(_h_at_lower_end(model, x, offset) >= -_FEAS_TOL)

    @pytest.mark.parametrize("n", [1, 2])
    def test_tiny_samples(self, n: int) -> None:
        x = np.linspace(0.2, 0.8, n)
        y = np.linspace(0.3, 0.7, n)
        model = _fit(y, x, basis=_basis(y_order=2, x_order=1))
        assert np.all(_h_at_lower_end(model, x) >= -_FEAS_TOL)

    @pytest.mark.parametrize("scale", [1e-3, 1e3])
    def test_extreme_response_scale(self, scale: float) -> None:
        """Support rows hold when the response lives on a tiny or huge scale."""
        x, y = _repro_data()
        basis = InteractionBasis(
            BernsteinBasis(order=3, support=(0.0, scale)),
            BernsteinBasis(order=2, support=_SUPPORT),
        )
        model = _fit(y * scale, x, basis=basis)
        assert np.all(_h_at_lower_end(model, x) >= -_FEAS_TOL)
        assert np.isfinite(model.result_.log_likelihood)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Predict-side support warning (#157 contract on the interaction path)
# ---------------------------------------------------------------------------

# Columns non-decreasing; Θ[0, :] = [0, -1, 0] gives
# h(a|x) = -2·x·(1-x) for an order-2 Bernstein x-basis: 0 at x ∈ {0, 1}, −0.5
# at x = 0.5 — below the support *inside* the hull of {0, 1}.
_THETA_DIP = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 1.0], [2.0, 1.0, 2.0], [3.0, 2.0, 3.0]]
)


class TestPredictSupportWarning:
    @pytest.mark.parametrize("what", ["distribution", "density", "quantile"])
    def test_in_sample_is_silent(
        self, auglag_fit: ConditionalTransformationModel, what: str
    ) -> None:
        x, _ = _repro_data()
        y_new = np.full(x.shape[0], 0.5)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            auglag_fit.predict(y_new, X_new=x[:, None], what=what)
        assert not any(_SUPPORT_MSG in m for m in _messages(record))

    @pytest.mark.parametrize("what", ["distribution", "density", "quantile"])
    def test_in_hull_dip_warns(
        self, auglag_fit: ConditionalTransformationModel, what: str
    ) -> None:
        model = _with_theta(auglag_fit, _THETA_DIP)
        x_new = np.array([[0.0], [0.5], [1.0]])
        y_new = np.full(3, 0.5)
        with pytest.warns(UserWarning, match=r"h\(a\|x\) < 0 at 1 row\(s\) \[1\]"):
            model.predict(y_new, X_new=x_new, what=what)

    def test_values_unclamped(self, auglag_fit: ConditionalTransformationModel) -> None:
        model = _with_theta(auglag_fit, _THETA_DIP)
        x_new = np.array([[0.0], [0.5], [1.0]])
        with pytest.warns(UserWarning, match="unclamped"):
            cdf = model.predict(np.zeros(3), X_new=x_new, what="distribution")
        # F(a|0.5) = 1 − exp(0.5) < 0: returned as is.
        np.testing.assert_allclose(cdf[1], 1.0 - np.exp(0.5), rtol=1e-12)
        np.testing.assert_allclose(cdf[[0, 2]], 0.0, atol=1e-12)

    def test_offset_enters_check(
        self, auglag_fit: ConditionalTransformationModel
    ) -> None:
        model = _with_theta(auglag_fit, _THETA_DIP)
        x_new = np.array([[0.5]])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            model.predict(
                np.array([0.5]), X_new=x_new, what="distribution", offset_new=np.ones(1)
            )
        assert not any(_SUPPORT_MSG in m for m in _messages(record))

    @pytest.mark.parametrize("solver", _SOLVERS)
    def test_extrapolated_x_warns(self, solver: str) -> None:
        """Fit on x ∈ [0, 0.4]; x ∈ {0.7, 1} lies outside the training range.

        Exact data with no offset presses every ``h(a|x_i)`` to 0, which pins
        ``Θ[0, :] = 0`` once the ``b(x_i)`` span ``ℝ^q``, so ``h(a|x) ≡ 0``.
        The known offset ``−x³`` is not representable by the order-2
        x-basis: the fitted quadratic ``Θ[0, :]·b(x)`` sits just above ``x³``
        on the training range and falls below it beyond, so
        ``h(a|x) = Θ[0, :]·b(x) − x³ ≈ −0.093`` at 0.7 and ``≈ −0.44`` at 1.
        """
        rng = np.random.default_rng(3)
        x = rng.uniform(0.0, 0.4, 200)
        y = rng.uniform(0.05, 0.95, 200)
        model = _fit(y, x, solver=solver, offset=-(x**3))
        assert np.all(_h_at_lower_end(model, x, -(x**3)) >= -_FEAS_TOL)
        x_far = np.array([0.1, 0.7, 1.0])
        h_far = _h_at_lower_end(model, x_far, -(x_far**3))
        assert h_far[0] >= -_FEAS_TOL
        assert np.all(h_far[1:] < -0.05)
        with pytest.warns(UserWarning, match=r"h\(a\|x\) < 0 at 2 row\(s\) \[1, 2\]"):
            model.predict(np.full(3, 0.5), X_new=x_far[:, None], offset_new=-(x_far**3))

    def test_warning_points_at_caller(
        self, auglag_fit: ConditionalTransformationModel
    ) -> None:
        model = _with_theta(auglag_fit, _THETA_DIP)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            model.predict(np.array([0.5]), X_new=np.array([[0.5]]))
        hits = [w for w in record if _SUPPORT_MSG in str(w.message)]
        assert hits and hits[0].filename == __file__

    def test_empty_prediction(self, auglag_fit: ConditionalTransformationModel) -> None:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out = auglag_fit.predict(
                np.empty(0), X_new=np.empty((0, 1)), what="distribution"
            )
        assert out.shape == (0,)
        assert not any(_SUPPORT_MSG in m for m in _messages(record))


# ---------------------------------------------------------------------------
# Quantile wording on the interaction path
# ---------------------------------------------------------------------------

# Θ[0, :] = 0.5 ⇒ h(a|x) = 0.5 for every x, so F(a|x) = 1 − e^{−0.5} ≈ 0.393.
_THETA_ATOM = np.array(
    [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
)
_ATOM_MASS = 1.0 - np.exp(-0.5)


class TestQuantileAtomWording:
    @pytest.mark.parametrize("p", [1e-6, 0.1, _ATOM_MASS - 1e-3])
    def test_below_atom_returns_a_with_atom_wording(
        self, auglag_fit: ConditionalTransformationModel, p: float
    ) -> None:
        model = _with_theta(auglag_fit, _THETA_ATOM)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            q = model.predict(np.array([p]), X_new=np.array([[0.3]]), what="quantile")
        msgs = _messages(record)
        assert any(_ATOM_MSG in m for m in msgs)
        assert not any("asymptotic limit" in m for m in msgs)
        hits = [w for w in record if _ATOM_MSG in str(w.message)]
        assert hits[0].filename == __file__
        np.testing.assert_allclose(q, 0.0, atol=1e-8)

    @pytest.mark.parametrize("p", [0.5, 0.9])
    def test_above_atom_is_silent(
        self, auglag_fit: ConditionalTransformationModel, p: float
    ) -> None:
        model = _with_theta(auglag_fit, _THETA_ATOM)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            q = model.predict(np.array([p]), X_new=np.array([[0.3]]), what="quantile")
        assert _messages(record) == []
        assert 0.0 < q[0] < 1.0

    @pytest.mark.parametrize("n", [0, 1, 400])
    def test_simulate_drops_atom_warning(
        self, auglag_fit: ConditionalTransformationModel, n: int
    ) -> None:
        """#159 on the interaction path: draws hit the atom ``a`` by design."""
        model = _with_theta(auglag_fit, _THETA_ATOM)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            draws = model.simulate(n, X=np.full((n, 1), 0.3), random_state=0)
        assert not any(_ATOM_MSG in m for m in _messages(record))
        assert draws.shape == (n,)
        if n == 400:
            # P(Y = a | x) = F(a|x) ≈ 0.393; 400 draws give SE ≈ 0.024.
            frac = float(np.mean(np.abs(draws) < 1e-8))
            assert abs(frac - _ATOM_MASS) < 0.1

    def test_upper_clip_keeps_endpoint_wording(
        self, auglag_fit: ConditionalTransformationModel
    ) -> None:
        # h(b|x) = 3 ⇒ F(b|x) = 1 − e^{−3} ≈ 0.95; p = 0.999 clips at b.
        model = _with_theta(auglag_fit, _THETA_ATOM)
        with pytest.warns(UserWarning, match="asymptotic limit"):
            model.predict(np.array([0.999]), X_new=np.array([[0.3]]), what="quantile")

    def test_normal_link_keeps_endpoint_wording(self) -> None:
        x, y = _repro_data()
        model = ConditionalTransformationModel(_basis(), base_distribution="normal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y, X=x[:, None])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            model.predict(np.array([1e-12]), X_new=np.array([[0.3]]), what="quantile")
        msgs = _messages(record)
        assert any("asymptotic limit" in m for m in msgs)
        assert not any(_ATOM_MSG in m for m in msgs)
