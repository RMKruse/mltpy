"""``BoxCox(interacting=..., scaling=...)`` — issue #107.

The stratified (interacting) Box-Cox model with an optional scaled baseline,

    Φ⁻¹ F(y | x, x_s) = ([a(y) ⊗ b(x)]ᵀ vec(Θ)) · exp(0.5 · x_s · γ),

reached through the ``BoxCox`` convenience surface rather than
``ConditionalTransformationModel(basis=InteractionBasis(...), scaling=...)``.
Parameter layout is ``theta_ = [vec_C(Θ) | γ]`` (ADR 0003, no β shift
block), or ``vec_C(Θ)`` without ``scaling=``.  The interaction likelihood is
exact-data only, so censored rows are rejected with a clear error.

R reference data (``reference/stram_onehot_bs_boxcox_*``) is produced by
``reference/generate_reference.R`` from ``tram::BoxCox(y | s ~ 1 | z)`` — the
no-shift ``stram`` route of ADR 0003 Decision 6 — with a 3-level factor
stratum ``s`` (mltpy: ``OneHotBasis(K=3)``).  tram codes ``s`` with treatment
contrasts; the generator maps its Θ to the one-hot layout
(``*_theta_onehot.txt``).  γ is sign- and magnitude-aligned with mltpy (no
flip).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mltpy import (
    ConditionalTransformationModel,
    InteractionBasis,
    OneHotBasis,
    OptimizerConfig,
    log_likelihood,
)
from mltpy.basis import BernsteinBasis
from mltpy.tram import BoxCox
from mltpy.variables import CensoredData, CensoringType

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"
_PREFIX = "stram_onehot_bs_boxcox_"
_REF_FILES = [
    "y_train",
    "s_train",
    "x_s_train",
    "y_support",
    "theta_onehot",
    "gamma",
    "loglik",
    "grid",
    "distribution",
    "density",
    "qgrid",
    "quantile",
]

# Bases mirroring the R block: Bernstein-y order 2 (p = 3) ⊗ one-hot strata
# (K = 3).
_ORDER_Y = 2
_K = 3

_EXPERIMENTAL = pytest.mark.filterwarnings(
    "ignore:scaling= with InteractionBasis is an experimental path"
)


# ---------------------------------------------------------------------------
# R parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bx_ref() -> dict[str, np.ndarray]:
    paths = {k: REF_DIR / f"{_PREFIX}{k}.txt" for k in _REF_FILES}
    if not all(p.exists() for p in paths.values()):
        pytest.skip(
            f"{_PREFIX}* fixtures not generated — run "
            "Rscript reference/generate_reference.R"
        )
    return {k: np.loadtxt(p) for k, p in paths.items()}


def _theta_r_mltpy_layout(bx_ref: dict[str, np.ndarray]) -> np.ndarray:
    """R's fit as mltpy's ``[vec_C(Θ) | γ]`` with the one-hot Θ."""
    return np.concatenate([bx_ref["theta_onehot"], np.atleast_1d(bx_ref["gamma"])])


# Both the default auglag solver and trust-constr pin Θ (and so every
# prediction built from it) to R's polished MLE at the standard rtol=1e-6.
_SOLVERS = ("auglag", "trust-constr")
_RTOL = 1e-6


@pytest.fixture(scope="module")
def bx_fits(bx_ref: dict[str, np.ndarray]) -> dict[str, BoxCox]:
    a, b = bx_ref["y_support"]
    fits: dict[str, BoxCox] = {}
    for solver in _SOLVERS:
        model = BoxCox(
            support=(float(a), float(b)),
            order=_ORDER_Y,
            interacting=OneHotBasis(K=_K),
            scaling=bx_ref["x_s_train"].reshape(-1, 1),
            optimizer_config=OptimizerConfig(solver=solver, random_state=0),
        )
        with pytest.warns(UserWarning, match="experimental"):
            model.fit(bx_ref["y_train"], X=bx_ref["s_train"])
        fits[solver] = model
    return fits


class TestRParity:
    def test_gamma_matches_r(
        self, bx_fits: dict[str, BoxCox], bx_ref: dict[str, np.ndarray]
    ) -> None:
        gamma_r = np.atleast_1d(bx_ref["gamma"])
        assert_allclose(bx_fits["auglag"].gamma_, gamma_r, rtol=1e-6, atol=1e-10)

    def test_loglik_matches_r(
        self, bx_fits: dict[str, BoxCox], bx_ref: dict[str, np.ndarray]
    ) -> None:
        result = bx_fits["auglag"].result_
        assert result is not None
        assert_allclose(
            result.log_likelihood, float(bx_ref["loglik"]), rtol=1e-6, atol=1e-10
        )

    def test_loglik_at_r_theta_matches_r(
        self, bx_fits: dict[str, BoxCox], bx_ref: dict[str, np.ndarray]
    ) -> None:
        """mltpy's likelihood evaluated at R's θ reproduces R's logLik.

        Confirms the one-hot reparametrisation of tram's treatment-contrast Θ
        and the basis supports agree term-for-term.
        """
        model = bx_fits["auglag"]
        ll = log_likelihood(
            _theta_r_mltpy_layout(bx_ref),
            model.basis,
            bx_ref["y_train"],
            bx_ref["s_train"].reshape(-1, 1),
            CensoringType.NONE,
            base_distribution="normal",
            scaling=bx_ref["x_s_train"].reshape(-1, 1),
        )
        assert_allclose(ll, float(bx_ref["loglik"]), rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("solver", _SOLVERS)
    def test_theta_matches_r(
        self,
        bx_fits: dict[str, BoxCox],
        bx_ref: dict[str, np.ndarray],
        solver: str,
    ) -> None:
        assert_allclose(
            bx_fits[solver].theta_,
            _theta_r_mltpy_layout(bx_ref),
            rtol=_RTOL,
            atol=1e-10,
        )

    @pytest.mark.parametrize("solver", _SOLVERS)
    @pytest.mark.parametrize("what", ["distribution", "density"])
    def test_predict_matches_r(
        self,
        bx_fits: dict[str, BoxCox],
        bx_ref: dict[str, np.ndarray],
        what: str,
        solver: str,
    ) -> None:
        """predict() with the parallel X_scale_new= matches R row-wise."""
        grid = bx_ref["grid"]
        y, s, x_s = grid[:, 0], grid[:, 1:2], grid[:, 2:3]
        got = bx_fits[solver].predict(y, X_new=s, what=what, X_scale_new=x_s)  # type: ignore[arg-type]
        assert_allclose(got, bx_ref[what], rtol=_RTOL, atol=1e-10)

    @pytest.mark.parametrize("solver", _SOLVERS)
    def test_quantile_matches_r(
        self,
        bx_fits: dict[str, BoxCox],
        bx_ref: dict[str, np.ndarray],
        solver: str,
    ) -> None:
        """Quantiles are positions on the support, so their absolute error
        scales with its width: a θ within 1e-7 of R moves the median quantile
        (≈ 0.007, near zero) by ~7e-9, beyond a purely relative bound."""
        qgrid = bx_ref["qgrid"]
        p, s, x_s = qgrid[:, 0], qgrid[:, 1:2], qgrid[:, 2:3]
        got = bx_fits[solver].predict(p, X_new=s, what="quantile", X_scale_new=x_s)
        a, b = bx_ref["y_support"]
        assert_allclose(got, bx_ref["quantile"], rtol=_RTOL, atol=_RTOL * (b - a))


# ---------------------------------------------------------------------------
# Convenience surface (synthetic data)
# ---------------------------------------------------------------------------


def _synthetic(
    n: int = 300, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Exact responses from Φ⁻¹F = exp(0.2·x_s)·(1.5 + x)·(y − 0.8x)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    x_s = rng.normal(size=(n, 1))
    k = np.exp(0.2 * x_s[:, 0]) * (1.5 + x)
    y = 0.8 * x + rng.normal(size=n) / k
    return y, x, x_s, (float(y.min() - 0.2), float(y.max() + 0.2))


def _x_basis() -> BernsteinBasis:
    return BernsteinBasis(order=1, support=(0.0, 1.0))


def _boxcox(
    support: tuple[float, float],
    x_s: np.ndarray | None = None,
    censoring: CensoringType = CensoringType.NONE,
) -> BoxCox:
    return BoxCox(
        support=support,
        order=_ORDER_Y,
        interacting=_x_basis(),
        scaling=x_s,
        censoring=censoring,
        optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
    )


@pytest.fixture(scope="module")
def synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    return _synthetic()


@pytest.fixture(scope="module")
def fitted_synthetic(
    synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
) -> BoxCox:
    y, x, x_s, support = synthetic
    model = _boxcox(support, x_s)
    with pytest.warns(UserWarning, match="experimental"):
        model.fit(y, X=x)
    return model


_Y_GRID = np.array([-0.5, 0.0, 0.4, 0.8, 1.2, 0.2])
_X_GRID = np.array([0.1, 0.4, 0.9, 0.5, 0.2, 0.7])


def _analytic_h(
    model: BoxCox, y: np.ndarray, x: np.ndarray, x_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """h and ∂h/∂y from the tensor basis, Θ and γ: (Z·vec Θ)·exp(0.5·x_s·γ)."""
    assert isinstance(model.basis, InteractionBasis)
    assert model.Theta_ is not None
    vec_theta = model.Theta_.ravel()
    f = np.exp(0.5 * (x_s @ model.gamma_))
    h = (model.basis.evaluate(y, x) @ vec_theta) * f
    hp = (model.basis.derivative(y, x) @ vec_theta) * f
    return h, hp


class TestConvenienceSurface:
    @pytest.mark.parametrize(
        "x_basis",
        [BernsteinBasis(order=2, support=(0.0, 1.0)), OneHotBasis(K=3)],
        ids=["bernstein", "onehot"],
    )
    @pytest.mark.parametrize("scaled", [False, True], ids=["unscaled", "scaled"])
    def test_builds_interaction_model(
        self, x_basis: BernsteinBasis | OneHotBasis, scaled: bool
    ) -> None:
        x_s = np.zeros((4, 1)) if scaled else None
        model = BoxCox(support=(0.0, 1.0), order=3, interacting=x_basis, scaling=x_s)
        assert isinstance(model.basis, InteractionBasis)
        assert model.basis.x_basis is x_basis
        assert isinstance(model.basis.y_basis, BernsteinBasis)
        assert model.basis.y_basis.order == 3
        assert model.base_distribution == "normal"
        assert model.censoring == CensoringType.NONE
        assert (model.scaling is not None) == scaled
        assert "BoxCox(order=3" in repr(model)

    def test_matches_base_class_path(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        fitted_synthetic: BoxCox,
    ) -> None:
        """BoxCox is a thin wrapper over the scaled InteractionBasis CTM."""
        y, x, x_s, support = synthetic
        ref = ConditionalTransformationModel(
            basis=InteractionBasis(
                y_basis=BernsteinBasis(order=_ORDER_Y, support=support),
                x_basis=_x_basis(),
            ),
            scaling=x_s,
            base_distribution="normal",
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        with pytest.warns(UserWarning, match="experimental"):
            ref.fit(y, X=x)
        assert fitted_synthetic.theta_ is not None
        assert ref.theta_ is not None
        assert_allclose(fitted_synthetic.theta_, ref.theta_, rtol=1e-12, atol=0)

    def test_exposes_theta_and_gamma(self, fitted_synthetic: BoxCox) -> None:
        p, q = _ORDER_Y + 1, 2
        assert fitted_synthetic.Theta_ is not None
        assert fitted_synthetic.Theta_.shape == (p, q)
        assert fitted_synthetic.gamma_.shape == (1,)
        assert fitted_synthetic.theta_ is not None
        assert fitted_synthetic.theta_.shape == (p * q + 1,)
        # Θ columns are non-decreasing in y (column-wise monotonicity).
        assert np.all(np.diff(fitted_synthetic.Theta_, axis=0) >= -1e-8)

    @pytest.mark.parametrize("x_s_value", [-1.5, 0.0, 2.0])
    def test_distribution_and_density_match_analytic(
        self, fitted_synthetic: BoxCox, x_s_value: float
    ) -> None:
        """Normal link: F = Φ(h), f = φ(h)·h'."""
        from scipy.stats import norm

        x_s = np.full((_Y_GRID.size, 1), x_s_value)
        h, hp = _analytic_h(fitted_synthetic, _Y_GRID, _X_GRID, x_s)
        cdf = fitted_synthetic.predict(
            _Y_GRID, X_new=_X_GRID, what="distribution", X_scale_new=x_s
        )
        pdf = fitted_synthetic.predict(
            _Y_GRID, X_new=_X_GRID, what="density", X_scale_new=x_s
        )
        assert_allclose(cdf, norm.cdf(h), rtol=1e-10, atol=1e-14)
        assert_allclose(pdf, norm.pdf(h) * hp, rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("x_s_value", [-1.5, 0.0, 2.0])
    def test_quantile_inverts_distribution(
        self, fitted_synthetic: BoxCox, x_s_value: float
    ) -> None:
        probs = np.array([0.05, 0.25, 0.5, 0.75, 0.95, 0.5])
        x_s = np.full((probs.size, 1), x_s_value)
        q = fitted_synthetic.predict(
            probs, X_new=_X_GRID, what="quantile", X_scale_new=x_s
        )
        back = fitted_synthetic.predict(
            q, X_new=_X_GRID, what="distribution", X_scale_new=x_s
        )
        assert_allclose(back, probs, rtol=1e-8, atol=1e-10)

    def test_scaling_changes_distribution(self, fitted_synthetic: BoxCox) -> None:
        lo = np.full((_Y_GRID.size, 1), -1.5)
        hi = np.full((_Y_GRID.size, 1), 2.0)
        f_lo = fitted_synthetic.predict(_Y_GRID, X_new=_X_GRID, X_scale_new=lo)
        f_hi = fitted_synthetic.predict(_Y_GRID, X_new=_X_GRID, X_scale_new=hi)
        assert np.max(np.abs(f_lo - f_hi)) > 1e-3

    def test_predict_requires_x_scale_new(self, fitted_synthetic: BoxCox) -> None:
        with pytest.raises(ValueError, match="X_scale_new must be"):
            fitted_synthetic.predict(_Y_GRID, X_new=_X_GRID)

    def test_extreme_x_scale_stays_finite(self, fitted_synthetic: BoxCox) -> None:
        """Large |x_s| saturates F at 0/1 instead of producing NaN."""
        x_s = np.array([[-40.0], [-10.0], [10.0], [40.0], [0.0], [25.0]])
        cdf = fitted_synthetic.predict(
            _Y_GRID, X_new=_X_GRID, what="distribution", X_scale_new=x_s
        )
        assert np.all(np.isfinite(cdf))
        assert np.all((cdf >= 0.0) & (cdf <= 1.0))

    def test_summary_tabulates_gamma(self, fitted_synthetic: BoxCox) -> None:
        s = fitted_synthetic.summary()
        assert "Model:        BoxCox" in s
        assert "Scaling coefficients:" in s

    def test_fitted_transformation_not_available(
        self, fitted_synthetic: BoxCox
    ) -> None:
        """No contiguous baseline θ_b exists on the interaction path."""
        with pytest.raises(NotImplementedError, match="InteractionBasis"):
            fitted_synthetic.fitted_transformation(_Y_GRID)

    def test_interacting_only_exposes_theta(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
    ) -> None:
        """Without scaling=, Θ is exposed, γ is undefined and no warning fires."""
        y, x, _, support = synthetic
        model = _boxcox(support).fit(y, X=x)
        assert model.Theta_ is not None
        assert model.Theta_.shape == (_ORDER_Y + 1, 2)
        assert model.theta_ is not None
        assert model.theta_.size == (_ORDER_Y + 1) * 2
        with pytest.raises(ValueError, match="gamma_ is undefined"):
            _ = model.gamma_

    def test_all_exact_censored_data_accepted(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        fitted_synthetic: BoxCox,
    ) -> None:
        """A CensoredData with no censored rows is exact data."""
        y, x, x_s, support = synthetic
        cd = CensoredData.right_censored(y, censored=np.zeros(y.size, dtype=bool))
        model = _boxcox(support, x_s, censoring=CensoringType.RIGHT)
        with pytest.warns(UserWarning, match="experimental"):
            model.fit(cd, X=x)
        assert model.theta_ is not None
        assert fitted_synthetic.theta_ is not None
        assert_allclose(model.theta_, fitted_synthetic.theta_, rtol=1e-12, atol=0)

    @_EXPERIMENTAL
    @pytest.mark.parametrize("scaled", [False, True], ids=["unscaled", "scaled"])
    def test_censored_data_rejected(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        scaled: bool,
    ) -> None:
        """Censored rows raise a clear error, not a NaN log-likelihood."""
        y, x, x_s, support = synthetic
        censored = np.zeros(y.size, dtype=bool)
        censored[::5] = True
        cd = CensoredData.right_censored(y, censored=censored)
        model = _boxcox(support, x_s if scaled else None, CensoringType.RIGHT)
        with pytest.raises(NotImplementedError, match="exact"):
            model.fit(cd, X=x)

    def test_interacting_requires_x(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
    ) -> None:
        y, _, _, support = synthetic
        with pytest.raises(ValueError, match="X"):
            _boxcox(support).fit(y)
