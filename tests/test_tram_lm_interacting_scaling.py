"""``Lm(interacting=..., scaling=...)`` — issue #109.

The stratified (interacting) normal linear model with an optional scaled
baseline,

    Φ⁻¹ F(y | x, x_s) = ([a(y) ⊗ b(x)]ᵀ vec(Θ)) · exp(0.5 · x_s · γ),

with ``a(y)`` the order-1 Bernstein basis, so ``h`` is affine in ``y`` with
an ``x``-dependent intercept and slope: ``Y | x`` is normal with its own mean
and standard deviation per ``x``.  Reached through the ``Lm`` convenience
surface rather than
``ConditionalTransformationModel(basis=InteractionBasis(...), scaling=...)``.
Parameter layout is ``theta_ = [vec_C(Θ) | γ]`` (ADR 0003, no β shift block),
or ``vec_C(Θ)`` without ``scaling=``.  The interaction likelihood is
exact-data only, so censored rows are rejected with a clear error.

R reference data (``reference/stram_onehot_lm_*``) is produced by
``reference/generate_reference.R`` from ``tram::Lm(y | s ~ 1 | z)`` — the
no-shift ``stram`` route of ADR 0003 Decision 6 — with a 3-level factor
stratum ``s`` (mltpy: ``OneHotBasis(K=3)``).  tram's response basis is the
polynomial ``(1, y)`` and it codes ``s`` with treatment contrasts; the
generator maps its Θ to the Bernstein one-hot layout
(``*_theta_onehot.txt``).  γ is sign- and magnitude-aligned with mltpy (no
flip).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from mltpy import (
    ConditionalTransformationModel,
    InteractionBasis,
    OneHotBasis,
    OptimizerConfig,
    log_likelihood,
)
from mltpy.basis import BernsteinBasis, OrdinalBasis, PolynomialBasis
from mltpy.tram import Lm
from mltpy.variables import CensoredData, CensoringType

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"
_PREFIX = "stram_onehot_lm_"
_REF_FILES = [
    "y_train",
    "s_train",
    "x_s_train",
    "y_support",
    "theta",
    "theta_onehot",
    "gamma",
    "loglik",
    "grid",
    "distribution",
    "density",
    "qgrid",
    "quantile",
]

# Lm fixes the Bernstein-y order at 1 (p = 2); the R block uses K = 3 strata.
_P = 2
_K = 3

_IGNORE_EXPERIMENTAL_WARNING = pytest.mark.filterwarnings(
    "ignore:scaling= with InteractionBasis is an experimental path"
)


# ---------------------------------------------------------------------------
# R parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lm_ref() -> dict[str, np.ndarray]:
    paths = {k: REF_DIR / f"{_PREFIX}{k}.txt" for k in _REF_FILES}
    if not all(p.exists() for p in paths.values()):
        pytest.skip(
            f"{_PREFIX}* fixtures not generated — run "
            "Rscript reference/generate_reference.R"
        )
    return {k: np.loadtxt(p) for k, p in paths.items()}


def _theta_r_mltpy_layout(lm_ref: dict[str, np.ndarray]) -> np.ndarray:
    """R's fit as mltpy's ``[vec_C(Θ) | γ]`` with the Bernstein one-hot Θ."""
    return np.concatenate([lm_ref["theta_onehot"], np.atleast_1d(lm_ref["gamma"])])


# Both the default auglag solver and trust-constr pin Θ (and so every
# prediction built from it) to R's polished MLE at the standard rtol=1e-6.
_SOLVERS = ("auglag", "trust-constr")
_RTOL = 1e-6


@pytest.fixture(scope="module")
def lm_fits(lm_ref: dict[str, np.ndarray]) -> dict[str, Lm]:
    a, b = lm_ref["y_support"]
    fits: dict[str, Lm] = {}
    for solver in _SOLVERS:
        model = Lm(
            support=(float(a), float(b)),
            interacting=OneHotBasis(K=_K),
            scaling=lm_ref["x_s_train"].reshape(-1, 1),
            optimizer_config=OptimizerConfig(solver=solver, random_state=0),
        )
        with pytest.warns(UserWarning, match="experimental"):
            model.fit(lm_ref["y_train"], X=lm_ref["s_train"])
        fits[solver] = model
    return fits


class TestRParity:
    @pytest.mark.parametrize("solver", _SOLVERS)
    def test_gamma_matches_r(
        self,
        lm_fits: dict[str, Lm],
        lm_ref: dict[str, np.ndarray],
        solver: str,
    ) -> None:
        gamma_r = np.atleast_1d(lm_ref["gamma"])
        assert_allclose(lm_fits[solver].gamma_, gamma_r, rtol=1e-6, atol=1e-10)

    def test_onehot_theta_is_tram_polynomial_contrast_theta(
        self, lm_ref: dict[str, np.ndarray]
    ) -> None:
        """The fixture's Θ is tram's polynomial-y, treatment-contrast Θ remapped.

        tram orders its baseline y-fastest per x-column ``(Intercept)``,
        ``sb``, ``sc`` with rows ``(Intercept)``, ``y``; per-stratum
        ``(c0, c1)`` is the intercept column plus contrast column k.  The
        order-1 Bernstein coefficients are ``h`` at the support ends:
        ``θ0 = c0 + c1·a``, ``θ1 = c0 + c1·b``.
        """
        a, b = lm_ref["y_support"]
        C = lm_ref["theta"][: _P * _K].reshape(_K, _P).T  # (2, K) contrast
        C[:, 1:] += C[:, [0]]
        Theta = np.vstack([C[0] + C[1] * a, C[0] + C[1] * b])
        assert_allclose(Theta.ravel(), lm_ref["theta_onehot"], rtol=1e-13, atol=0)
        assert_allclose(lm_ref["theta"][_P * _K :], lm_ref["gamma"], rtol=0, atol=0)

    def test_loglik_matches_r(
        self, lm_fits: dict[str, Lm], lm_ref: dict[str, np.ndarray]
    ) -> None:
        result = lm_fits["auglag"].result_
        assert result is not None
        assert_allclose(
            result.log_likelihood, float(lm_ref["loglik"]), rtol=1e-6, atol=1e-10
        )

    def test_loglik_at_r_theta_matches_r(
        self, lm_fits: dict[str, Lm], lm_ref: dict[str, np.ndarray]
    ) -> None:
        """mltpy's likelihood evaluated at R's θ reproduces R's logLik.

        Confirms the polynomial → Bernstein and contrast → one-hot
        reparametrisations of tram's Θ agree term-for-term.
        """
        model = lm_fits["auglag"]
        ll = log_likelihood(
            _theta_r_mltpy_layout(lm_ref),
            model.basis,
            lm_ref["y_train"],
            lm_ref["s_train"].reshape(-1, 1),
            CensoringType.NONE,
            base_distribution="normal",
            scaling=lm_ref["x_s_train"].reshape(-1, 1),
        )
        assert_allclose(ll, float(lm_ref["loglik"]), rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("solver", _SOLVERS)
    def test_theta_matches_r(
        self,
        lm_fits: dict[str, Lm],
        lm_ref: dict[str, np.ndarray],
        solver: str,
    ) -> None:
        assert_allclose(
            lm_fits[solver].theta_,
            _theta_r_mltpy_layout(lm_ref),
            rtol=_RTOL,
            atol=1e-10,
        )

    @pytest.mark.parametrize("solver", _SOLVERS)
    @pytest.mark.parametrize("what", ["distribution", "density"])
    def test_predict_matches_r(
        self,
        lm_fits: dict[str, Lm],
        lm_ref: dict[str, np.ndarray],
        what: str,
        solver: str,
    ) -> None:
        """predict() with the parallel X_scale_new= matches R row-wise."""
        grid = lm_ref["grid"]
        y, s, x_s = grid[:, 0], grid[:, 1:2], grid[:, 2:3]
        got = lm_fits[solver].predict(y, X_new=s, what=what, X_scale_new=x_s)
        assert_allclose(got, lm_ref[what], rtol=_RTOL, atol=1e-10)

    @pytest.mark.parametrize("solver", _SOLVERS)
    def test_quantile_matches_r(
        self,
        lm_fits: dict[str, Lm],
        lm_ref: dict[str, np.ndarray],
        solver: str,
    ) -> None:
        """R's quantiles are closed-form: (Φ⁻¹(p)·exp(−0.5·z·γ) − c0_s) / c1_s."""
        qgrid = lm_ref["qgrid"]
        p, s, x_s = qgrid[:, 0], qgrid[:, 1:2], qgrid[:, 2:3]
        got = lm_fits[solver].predict(p, X_new=s, what="quantile", X_scale_new=x_s)
        assert_allclose(got, lm_ref["quantile"], rtol=_RTOL, atol=1e-10)


# ---------------------------------------------------------------------------
# Convenience surface (synthetic data)
# ---------------------------------------------------------------------------


def _synthetic(
    n: int = 300, seed: int = 11
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Exact responses from Φ⁻¹F = exp(0.2·x_s)·(1.5 + x)·(y − 0.8x).

    Affine in ``y`` with an ``x``-dependent slope, so it lies exactly in the
    order-1 Bernstein ⊗ order-1 Bernstein span.  The affine model does not
    depend on the support, so it is padded by 1 to keep tail quantiles at
    small x_s (wide spread) inside it.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    x_s = rng.normal(size=(n, 1))
    k = np.exp(0.2 * x_s[:, 0]) * (1.5 + x)
    y = 0.8 * x + rng.normal(size=n) / k
    return y, x, x_s, (float(y.min() - 1.0), float(y.max() + 1.0))


def _x_basis() -> BernsteinBasis:
    return BernsteinBasis(order=1, support=(0.0, 1.0))


def _lm(support: tuple[float, float], x_s: np.ndarray | None = None) -> Lm:
    return Lm(
        support=support,
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
) -> Lm:
    y, x, x_s, support = synthetic
    model = _lm(support, x_s)
    with pytest.warns(UserWarning, match="experimental"):
        model.fit(y, X=x)
    return model


@pytest.fixture(scope="module")
def fitted_unscaled(
    synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
) -> Lm:
    y, x, _, support = synthetic
    return _lm(support).fit(y, X=x)


_Y_GRID = np.array([-0.5, 0.0, 0.4, 0.8, 1.2, 0.2])
_X_GRID = np.array([0.1, 0.4, 0.9, 0.5, 0.2, 0.7])


def _analytic_h(
    model: Lm, y: np.ndarray, x: np.ndarray, x_s: np.ndarray
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
        [
            BernsteinBasis(order=2, support=(0.0, 1.0)),
            OrdinalBasis(K=3),
            OneHotBasis(K=3),
        ],
        ids=["bernstein", "ordinal", "onehot"],
    )
    @pytest.mark.parametrize("scaled", [False, True], ids=["unscaled", "scaled"])
    def test_builds_interaction_model(
        self, x_basis: BernsteinBasis | OrdinalBasis | OneHotBasis, scaled: bool
    ) -> None:
        x_s = np.zeros((4, 1)) if scaled else None
        model = Lm(support=(0.0, 1.0), interacting=x_basis, scaling=x_s)
        assert isinstance(model.basis, InteractionBasis)
        assert model.basis.x_basis is x_basis
        assert isinstance(model.basis.y_basis, BernsteinBasis)
        assert model.basis.y_basis.order == 1
        assert model.basis.y_basis.support == (0.0, 1.0)
        assert model.base_distribution == "normal"
        assert model.censoring == CensoringType.NONE
        assert (model.scaling is not None) == scaled
        assert "Lm(order=1" in repr(model)

    def test_order_still_rejected(self) -> None:
        """The order-1 Bernstein-y is fixed on the interacting path too."""
        with pytest.raises(TypeError):
            Lm(support=(0.0, 1.0), order=2, interacting=_x_basis())  # type: ignore[call-arg]

    def test_matches_base_class_path(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        fitted_synthetic: Lm,
    ) -> None:
        """Lm is a thin wrapper over the scaled InteractionBasis CTM."""
        y, x, x_s, support = synthetic
        ref = ConditionalTransformationModel(
            basis=InteractionBasis(
                y_basis=BernsteinBasis(order=1, support=support),
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

    def test_exposes_theta_and_gamma(self, fitted_synthetic: Lm) -> None:
        q = 2
        assert fitted_synthetic.Theta_ is not None
        assert fitted_synthetic.Theta_.shape == (_P, q)
        assert fitted_synthetic.gamma_.shape == (1,)
        assert fitted_synthetic.theta_ is not None
        assert fitted_synthetic.theta_.shape == (_P * q + 1,)
        # Θ columns are non-decreasing in y (column-wise monotonicity).
        assert np.all(np.diff(fitted_synthetic.Theta_, axis=0) >= -1e-8)

    def test_recovers_generating_model(self, fitted_unscaled: Lm) -> None:
        """Unscaled fit on scaled data still lands near the true per-x slope.

        With h(y | x) = (1.5 + x)·(y − 0.8x), the latent slope ∂h/∂y is 1.5
        at x = 0 and 2.5 at x = 1 (the omitted x_s term only inflates the
        marginal spread, so the slope is attenuated, not reversed).
        """
        assert fitted_unscaled.Theta_ is not None
        a, b = fitted_unscaled.basis.support
        slopes = np.diff(fitted_unscaled.Theta_, axis=0)[0] / (b - a)
        assert slopes[1] > slopes[0] > 0.0

    @pytest.mark.parametrize("x_s_value", [-1.5, 0.0, 2.0])
    def test_distribution_and_density_match_analytic(
        self, fitted_synthetic: Lm, x_s_value: float
    ) -> None:
        """Normal link: F = Φ(h), f = φ(h)·h'."""
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
        self, fitted_synthetic: Lm, x_s_value: float
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

    def test_scaling_changes_distribution(self, fitted_synthetic: Lm) -> None:
        lo = np.full((_Y_GRID.size, 1), -1.5)
        hi = np.full((_Y_GRID.size, 1), 2.0)
        f_lo = fitted_synthetic.predict(_Y_GRID, X_new=_X_GRID, X_scale_new=lo)
        f_hi = fitted_synthetic.predict(_Y_GRID, X_new=_X_GRID, X_scale_new=hi)
        assert np.max(np.abs(f_lo - f_hi)) > 1e-3

    def test_predict_requires_x_scale_new(self, fitted_synthetic: Lm) -> None:
        with pytest.raises(ValueError, match="X_scale_new must be"):
            fitted_synthetic.predict(_Y_GRID, X_new=_X_GRID)

    def test_extreme_x_scale_stays_finite(self, fitted_synthetic: Lm) -> None:
        """Large |x_s| saturates F at 0/1 instead of producing NaN."""
        x_s = np.array([[-40.0], [-10.0], [10.0], [40.0], [0.0], [25.0]])
        cdf = fitted_synthetic.predict(
            _Y_GRID, X_new=_X_GRID, what="distribution", X_scale_new=x_s
        )
        assert np.all(np.isfinite(cdf))
        assert np.all((cdf >= 0.0) & (cdf <= 1.0))

    def test_summary_tabulates_gamma(self, fitted_synthetic: Lm) -> None:
        s = fitted_synthetic.summary()
        assert "Model:        Lm" in s
        assert "Scaling coefficients:" in s

    @pytest.mark.parametrize("prop", ["sigma_", "intercept_", "coef_"])
    @pytest.mark.parametrize("scaled", [False, True], ids=["unscaled", "scaled"])
    def test_closed_form_accessors_raise(
        self,
        fitted_unscaled: Lm,
        fitted_synthetic: Lm,
        prop: str,
        scaled: bool,
    ) -> None:
        """No single (σ, μ, coef) mapping exists when Θ varies with x."""
        model = fitted_synthetic if scaled else fitted_unscaled
        with pytest.raises(NotImplementedError, match=rf"{prop}.*interacting=.*Theta_"):
            getattr(model, prop)

    def test_fitted_transformation_not_available(self, fitted_synthetic: Lm) -> None:
        """No contiguous baseline θ_b exists on the interaction path."""
        with pytest.raises(NotImplementedError, match="InteractionBasis"):
            fitted_synthetic.fitted_transformation(_Y_GRID)

    def test_interacting_only_exposes_theta(self, fitted_unscaled: Lm) -> None:
        """Without scaling=, Θ is exposed, γ is undefined and no warning fires."""
        assert fitted_unscaled.Theta_ is not None
        assert fitted_unscaled.Theta_.shape == (_P, 2)
        assert fitted_unscaled.theta_ is not None
        assert fitted_unscaled.theta_.size == _P * 2
        with pytest.raises(ValueError, match="gamma_ is undefined"):
            _ = fitted_unscaled.gamma_

    def test_all_exact_censored_data_accepted(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        fitted_synthetic: Lm,
    ) -> None:
        """A CensoredData with no censored rows is exact data."""
        y, x, x_s, support = synthetic
        cd = CensoredData.right_censored(y, censored=np.zeros(y.size, dtype=bool))
        model = _lm(support, x_s)
        with pytest.warns(UserWarning, match="experimental"):
            model.fit(cd, X=x)
        assert model.theta_ is not None
        assert fitted_synthetic.theta_ is not None
        assert_allclose(model.theta_, fitted_synthetic.theta_, rtol=1e-12, atol=0)

    @_IGNORE_EXPERIMENTAL_WARNING
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
        model = _lm(support, x_s if scaled else None)
        with pytest.raises(NotImplementedError, match="exact"):
            model.fit(cd, X=x)

    @_IGNORE_EXPERIMENTAL_WARNING
    @pytest.mark.parametrize(
        ("x_basis", "first_code"),
        [(OrdinalBasis(K=3), 1), (OneHotBasis(K=3), 0)],
        ids=["ordinal", "onehot"],
    )
    def test_categorical_x_basis_fits(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        x_basis: OrdinalBasis | OneHotBasis,
        first_code: int,
    ) -> None:
        """Categorical x-bases fit with one Θ column per basis function.

        OrdinalBasis(K) labels cut positions 1..K-1; OneHotBasis(K) labels
        categories 0..K-1.
        """
        y, _, x_s, support = synthetic
        n_codes = x_basis.order + 1
        rng = np.random.default_rng(0)
        codes = rng.integers(first_code, first_code + n_codes, y.size)
        model = Lm(support=support, interacting=x_basis, scaling=x_s).fit(y, X=codes)
        assert model.Theta_ is not None
        assert model.Theta_.shape == (_P, x_basis.order + 1)
        cdf = model.predict(y[:5], X_new=codes[:5], X_scale_new=x_s[:5])
        assert np.all((cdf > 0.0) & (cdf < 1.0))

    def test_simulate_with_x_scale(self, fitted_synthetic: Lm) -> None:
        """simulate() threads the parallel X_scale= to the quantile path."""
        a, b = fitted_synthetic.basis.support
        draws = fitted_synthetic.simulate(
            _X_GRID.size,
            X=_X_GRID,
            X_scale=np.full((_X_GRID.size, 1), 0.5),
            random_state=0,
        )
        assert draws.shape == (_X_GRID.size,)
        assert np.all((draws >= a) & (draws <= b))

    def test_unsupported_x_basis_rejected(self) -> None:
        """Θ column monotonicity needs a non-negative partition of unity."""
        with pytest.raises(ValueError, match="partition of unity"):
            Lm(
                support=(0.0, 1.0),
                interacting=PolynomialBasis(order=1, support=(0.0, 1.0)),  # type: ignore[arg-type]
            )

    def test_empty_y_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one observation"):
            _lm((0.0, 1.0)).fit(np.empty(0), X=np.empty(0))

    def test_scaling_rows_must_match_y(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
    ) -> None:
        y, x, x_s, support = synthetic
        with pytest.raises(ValueError, match="scaling has"):
            _lm(support, x_s[:-1]).fit(y, X=x)

    def test_x_scale_new_shape_checked(self, fitted_synthetic: Lm) -> None:
        with pytest.raises(ValueError, match="X_scale_new has"):
            fitted_synthetic.predict(
                _Y_GRID, X_new=_X_GRID, X_scale_new=np.zeros((2, 1))
            )

    def test_interacting_requires_x(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
    ) -> None:
        y, _, _, support = synthetic
        with pytest.raises(ValueError, match=r"requires X \(covariate matrix\)"):
            _lm(support).fit(y)
