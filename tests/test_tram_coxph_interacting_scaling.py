"""``Coxph(interacting=..., scaling=...)`` — issue #106.

The non-proportional (interacting) Cox model with a scaled baseline,

    log[-log S(t | x, x_s)] = ([a(t) ⊗ b(x)]ᵀ vec(Θ)) · exp(0.5 · x_s · γ),

reached through the ``Coxph`` convenience surface rather than
``ConditionalTransformationModel(basis=InteractionBasis(...), scaling=...)``.
Parameter layout is ``theta_ = [vec_C(Θ) | γ]`` (ADR 0003, no β shift
block).  The interaction likelihood is exact-data only, so censored rows are
rejected with a clear error.

R reference data (``reference/stram_bs_bs_mev_*``) is produced by
``reference/generate_reference.R`` via ``mlt::ctm(response = Bernstein-y,
interacting = Bernstein-x, scaling = ~x_s, todistr = "MinExtrVal")`` — the
``stram`` route confirmed in ADR 0003.  γ is sign- and magnitude-aligned
with mltpy (no flip).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mltpy import (
    ConditionalTransformationModel,
    InteractionBasis,
    OptimizerConfig,
    log_likelihood,
)
from mltpy.basis import BernsteinBasis
from mltpy.tram import Coxph
from mltpy.variables import CensoredData, CensoringType

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"
_PREFIX = "stram_bs_bs_mev_"
_REF_FILES = [
    "y_train",
    "x_train",
    "x_s_train",
    "y_support",
    "theta",
    "gamma",
    "loglik",
    "grid",
    "survivor",
    "hazard",
]

# Bases mirroring the R block: Bernstein-y order 2 (p = 3) ⊗ Bernstein-x
# order 1 (q = 2) on x ∈ [0, 1].
_ORDER_Y = 2
_ORDER_X = 1

_EXPERIMENTAL = pytest.mark.filterwarnings(
    "ignore:scaling= with InteractionBasis is an experimental path"
)


def _x_basis() -> BernsteinBasis:
    return BernsteinBasis(order=_ORDER_X, support=(0.0, 1.0))


# ---------------------------------------------------------------------------
# R parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mev_ref() -> dict[str, np.ndarray]:
    paths = {k: REF_DIR / f"{_PREFIX}{k}.txt" for k in _REF_FILES}
    if not all(p.exists() for p in paths.values()):
        pytest.skip(
            f"{_PREFIX}* fixtures not generated — run "
            "Rscript reference/generate_reference.R"
        )
    return {k: np.loadtxt(p) for k, p in paths.items()}


def _theta_r_mltpy_layout(mev_ref: dict[str, np.ndarray]) -> np.ndarray:
    """R's coef(fit) reordered to mltpy's ``[vec_C(Θ) | γ]``.

    R orders vec(Θ) y-fastest (``Bs_i(y):Bs_j(x)``), i.e. column-major;
    mltpy's vec_C is row-major.
    """
    p, q = _ORDER_Y + 1, _ORDER_X + 1
    theta_r = mev_ref["theta"]
    return np.concatenate([theta_r[: p * q].reshape(q, p).T.ravel(), theta_r[p * q :]])


# Solvers fitted against the R reference, with the relative tolerance each
# reaches on the weakly-determined Θ block (and the S / hazard predictions
# built from it).  γ and ℓ meet rtol=1e-6 under the default auglag solver;
# Θ under auglag sits ~1e-5 off the optimum because its inner L-BFGS-B stalls
# on an infeasible trial step and the result is finished by the Newton-CG
# polish.  trust-constr converges tightly, showing the residual is solver
# precision rather than a model mismatch (see also
# test_loglik_at_r_theta_matches_r).
_SOLVER_THETA_RTOL = {"auglag": 1e-4, "trust-constr": 1e-6}


@pytest.fixture(scope="module")
def mev_fits(mev_ref: dict[str, np.ndarray]) -> dict[str, Coxph]:
    a, b = mev_ref["y_support"]
    fits: dict[str, Coxph] = {}
    for solver in _SOLVER_THETA_RTOL:
        model = Coxph(
            support=(float(a), float(b)),
            order=_ORDER_Y,
            interacting=_x_basis(),
            scaling=mev_ref["x_s_train"].reshape(-1, 1),
            optimizer_config=OptimizerConfig(solver=solver, random_state=0),
        )
        with pytest.warns(UserWarning, match="experimental"):
            model.fit(mev_ref["y_train"], X=mev_ref["x_train"])
        fits[solver] = model
    return fits


class TestRParity:
    def test_gamma_matches_r(
        self, mev_fits: dict[str, Coxph], mev_ref: dict[str, np.ndarray]
    ) -> None:
        gamma_r = np.atleast_1d(mev_ref["gamma"])
        assert_allclose(mev_fits["auglag"].gamma_, gamma_r, rtol=1e-6, atol=1e-10)

    def test_loglik_matches_r(
        self, mev_fits: dict[str, Coxph], mev_ref: dict[str, np.ndarray]
    ) -> None:
        result = mev_fits["auglag"].result_
        assert result is not None
        assert_allclose(
            result.log_likelihood, float(mev_ref["loglik"]), rtol=1e-6, atol=1e-10
        )

    def test_loglik_at_r_theta_matches_r(
        self, mev_fits: dict[str, Coxph], mev_ref: dict[str, np.ndarray]
    ) -> None:
        """mltpy's likelihood evaluated at R's θ reproduces R's logLik."""
        model = mev_fits["auglag"]
        ll = log_likelihood(
            _theta_r_mltpy_layout(mev_ref),
            model.basis,
            mev_ref["y_train"],
            mev_ref["x_train"].reshape(-1, 1),
            CensoringType.NONE,
            base_distribution="min_extreme_value",
            scaling=mev_ref["x_s_train"].reshape(-1, 1),
        )
        assert_allclose(ll, float(mev_ref["loglik"]), rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("solver", list(_SOLVER_THETA_RTOL))
    def test_theta_matches_r(
        self,
        mev_fits: dict[str, Coxph],
        mev_ref: dict[str, np.ndarray],
        solver: str,
    ) -> None:
        assert_allclose(
            mev_fits[solver].theta_,
            _theta_r_mltpy_layout(mev_ref),
            rtol=_SOLVER_THETA_RTOL[solver],
            atol=1e-10,
        )

    @pytest.mark.parametrize("solver", list(_SOLVER_THETA_RTOL))
    @pytest.mark.parametrize("method", ["survival", "hazard"])
    def test_domain_methods_match_r(
        self,
        mev_fits: dict[str, Coxph],
        mev_ref: dict[str, np.ndarray],
        method: str,
        solver: str,
    ) -> None:
        """survival / hazard with the parallel X_scale= match R row-wise."""
        grid = mev_ref["grid"]
        t, x, x_s = grid[:, 0], grid[:, 1], grid[:, 2:3]
        got = getattr(mev_fits[solver], method)(t, X=x, X_scale=x_s)
        expected = mev_ref["survivor" if method == "survival" else "hazard"]
        assert_allclose(got, expected, rtol=_SOLVER_THETA_RTOL[solver], atol=1e-10)


# ---------------------------------------------------------------------------
# Convenience surface (synthetic data)
# ---------------------------------------------------------------------------


def _synthetic(
    n: int = 200, seed: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Exact event times from log[-log S] = exp(0.2·x_s)·(2(t − 4.5) + 0.8x)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    x_s = rng.normal(size=(n, 1))
    k = np.exp(0.2 * x_s[:, 0])
    t = 4.5 - 0.4 * (x - 0.5) + np.log(-np.log(rng.uniform(size=n))) / (2.0 * k)
    return t, x, x_s, (float(t.min() - 0.1), float(t.max() + 0.1))


def _coxph(support: tuple[float, float], x_s: np.ndarray | None = None) -> Coxph:
    return Coxph(
        support=support,
        order=_ORDER_Y,
        interacting=_x_basis(),
        scaling=x_s,
        optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
    )


@pytest.fixture(scope="module")
def synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    return _synthetic()


@pytest.fixture(scope="module")
def fitted_synthetic(
    synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
) -> Coxph:
    t, x, x_s, support = synthetic
    model = _coxph(support, x_s)
    with pytest.warns(UserWarning, match="experimental"):
        model.fit(t, X=x)
    return model


def _analytic_h(
    model: Coxph, t: np.ndarray, x: np.ndarray, x_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """h and ∂h/∂t from the tensor basis, Θ and γ: (Z·vec Θ)·exp(0.5·x_s·γ)."""
    assert isinstance(model.basis, InteractionBasis)
    assert model.Theta_ is not None
    vec_theta = model.Theta_.ravel()
    f = np.exp(0.5 * (x_s @ model.gamma_))
    h = (model.basis.evaluate(t, x) @ vec_theta) * f
    hp = (model.basis.derivative(t, x) @ vec_theta) * f
    return h, hp


_T_GRID = np.array([3.6, 4.0, 4.4, 4.8, 5.0, 3.8])
_X_GRID = np.array([0.1, 0.4, 0.9, 0.5, 0.2, 0.7])


class TestConvenienceSurface:
    def test_builds_scaled_interaction_model(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
    ) -> None:
        _, _, x_s, support = synthetic
        model = _coxph(support, x_s)
        assert isinstance(model.basis, InteractionBasis)
        assert model.base_distribution == "min_extreme_value"
        assert model.scaling is not None
        assert_allclose(model.scaling, x_s)

    def test_matches_base_class_path(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        fitted_synthetic: Coxph,
    ) -> None:
        """Coxph is a thin wrapper over the scaled InteractionBasis CTM."""
        t, x, x_s, support = synthetic
        ref = ConditionalTransformationModel(
            basis=InteractionBasis(
                y_basis=BernsteinBasis(order=_ORDER_Y, support=support),
                x_basis=_x_basis(),
            ),
            censoring=CensoringType.RIGHT,
            scaling=x_s,
            base_distribution="min_extreme_value",
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        with pytest.warns(UserWarning, match="experimental"):
            ref.fit(t, X=x)
        assert_allclose(fitted_synthetic.theta_, ref.theta_, rtol=1e-12, atol=0)

    def test_exposes_theta_and_gamma(self, fitted_synthetic: Coxph) -> None:
        p, q = _ORDER_Y + 1, _ORDER_X + 1
        assert fitted_synthetic.Theta_ is not None
        assert fitted_synthetic.Theta_.shape == (p, q)
        assert fitted_synthetic.gamma_.shape == (1,)
        assert fitted_synthetic.theta_ is not None
        assert fitted_synthetic.theta_.shape == (p * q + 1,)
        # Θ columns are non-decreasing in t (column-wise monotonicity).
        assert np.all(np.diff(fitted_synthetic.Theta_, axis=0) >= -1e-8)

    @pytest.mark.parametrize("x_s_value", [-1.5, 0.0, 2.0])
    def test_survival_matches_analytic(
        self, fitted_synthetic: Coxph, x_s_value: float
    ) -> None:
        """Minimum extreme value link: S(t | x, x_s) = exp(−exp(h))."""
        x_s = np.full((_T_GRID.size, 1), x_s_value)
        h, _ = _analytic_h(fitted_synthetic, _T_GRID, _X_GRID, x_s)
        got = fitted_synthetic.survival(_T_GRID, X=_X_GRID, X_scale=x_s)
        assert_allclose(got, np.exp(-np.exp(h)), rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("x_s_value", [-1.5, 0.0, 2.0])
    def test_hazard_matches_analytic(
        self, fitted_synthetic: Coxph, x_s_value: float
    ) -> None:
        """Minimum extreme value link: λ(t | x, x_s) = f(h)·h'/S(h) = exp(h)·h'."""
        x_s = np.full((_T_GRID.size, 1), x_s_value)
        h, hp = _analytic_h(fitted_synthetic, _T_GRID, _X_GRID, x_s)
        got = fitted_synthetic.hazard(_T_GRID, X=_X_GRID, X_scale=x_s)
        assert_allclose(got, np.exp(h) * hp, rtol=1e-10, atol=1e-14)

    @pytest.mark.filterwarnings("ignore:predict.*exceeds:UserWarning")
    @pytest.mark.parametrize("x_s_value", [-60.0, 60.0])
    def test_extreme_scaling_stays_finite(
        self, fitted_synthetic: Coxph, x_s_value: float
    ) -> None:
        """|x_s| large enough to push h past the ±30 clip stays well-defined."""
        x_s = np.full((_T_GRID.size, 1), x_s_value)
        s = fitted_synthetic.survival(_T_GRID, X=_X_GRID, X_scale=x_s)
        lam = fitted_synthetic.hazard(_T_GRID, X=_X_GRID, X_scale=x_s)
        assert np.all(np.isfinite(s)) and np.all((s >= 0.0) & (s <= 1.0))
        assert np.all(np.isfinite(lam)) and np.all(lam >= 0.0)

    def test_quantile_and_simulate_not_yet_supported(
        self, fitted_synthetic: Coxph
    ) -> None:
        """Scaled interaction quantile inversion is deferred to issue #104."""
        x_s = np.zeros((_X_GRID.size, 1))
        with pytest.raises(NotImplementedError, match="#104"):
            fitted_synthetic.predict(
                np.full(_X_GRID.size, 0.5),
                X_new=_X_GRID,
                X_scale_new=x_s,
                what="quantile",
            )
        with pytest.raises(NotImplementedError, match="#104"):
            fitted_synthetic.simulate(_X_GRID.size, X=_X_GRID, X_scale=x_s)

    def test_scaling_changes_survival(self, fitted_synthetic: Coxph) -> None:
        """Different x_s values give different survival curves at the same x."""
        lo = np.full((_T_GRID.size, 1), -1.5)
        hi = np.full((_T_GRID.size, 1), 2.0)
        s_lo = fitted_synthetic.survival(_T_GRID, X=_X_GRID, X_scale=lo)
        s_hi = fitted_synthetic.survival(_T_GRID, X=_X_GRID, X_scale=hi)
        assert np.max(np.abs(s_lo - s_hi)) > 1e-3

    @pytest.mark.parametrize("method", ["survival", "hazard"])
    def test_domain_methods_require_x_scale(
        self, fitted_synthetic: Coxph, method: str
    ) -> None:
        with pytest.raises(ValueError, match="X_scale_new must be"):
            getattr(fitted_synthetic, method)(_T_GRID, X=_X_GRID)

    def test_all_exact_censored_data_accepted(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        fitted_synthetic: Coxph,
    ) -> None:
        """A CensoredData with no censored rows is exact data."""
        t, x, x_s, support = synthetic
        cd = CensoredData.right_censored(t, censored=np.zeros(t.size, dtype=bool))
        model = _coxph(support, x_s)
        with pytest.warns(UserWarning, match="experimental"):
            model.fit(cd, X=x)
        assert_allclose(model.theta_, fitted_synthetic.theta_, rtol=1e-12, atol=0)

    @_EXPERIMENTAL
    @pytest.mark.parametrize("scaled", [False, True], ids=["unscaled", "scaled"])
    def test_censored_data_rejected(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        scaled: bool,
    ) -> None:
        """Right-censored rows raise a clear error, not a NaN log-likelihood."""
        t, x, x_s, support = synthetic
        censored = np.zeros(t.size, dtype=bool)
        censored[::5] = True
        cd = CensoredData.right_censored(t, censored=censored)
        model = _coxph(support, x_s if scaled else None)
        with pytest.raises(NotImplementedError, match="exact"):
            model.fit(cd, X=x)

    def test_interacting_only_exposes_theta(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
    ) -> None:
        """Without scaling=, Θ is exposed and γ is undefined."""
        t, x, _, support = synthetic
        model = _coxph(support).fit(t, X=x)
        assert model.Theta_ is not None
        assert model.Theta_.shape == (_ORDER_Y + 1, _ORDER_X + 1)
        assert model.theta_ is not None
        assert model.theta_.size == (_ORDER_Y + 1) * (_ORDER_X + 1)
        with pytest.raises(ValueError, match="gamma_ is undefined"):
            _ = model.gamma_
