"""``Colr(interacting=..., scaling=...)`` — issue #108.

The non-proportional-odds (interacting) continuous-outcome logistic model with
an optional scaled baseline,

    logit F(y | x, x_s) = ([a(y) ⊗ b(x)]ᵀ vec(Θ)) · exp(0.5 · x_s · γ),

reached through the ``Colr`` convenience surface rather than
``ConditionalTransformationModel(basis=InteractionBasis(...), scaling=...)``.
Parameter layout is ``theta_ = [vec_C(Θ) | γ]`` (ADR 0003, no β shift
block), or ``vec_C(Θ)`` without ``scaling=``.  The interaction likelihood is
exact-data only, so censored rows are rejected with a clear error.

R reference data (``reference/stram_bs_bs_logistic_*``) is produced by
``reference/generate_reference.R`` via ``mlt::ctm(response = Bernstein-y,
interacting = Bernstein-x, scaling = ~x_s, todistr = "Logistic")`` — the
``stram`` route confirmed in ADR 0003.  γ is sign- and magnitude-aligned
with mltpy (no flip).
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit

from mltpy import (
    ConditionalTransformationModel,
    InteractionBasis,
    OptimizerConfig,
    log_likelihood,
)
from mltpy.basis import BernsteinBasis, OrdinalBasis
from mltpy.tram import Colr
from mltpy.variables import CensoredData, CensoringType

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"
_PREFIX = "stram_bs_bs_logistic_"
_REF_FILES = [
    "y_train",
    "x_train",
    "x_s_train",
    "y_support",
    "theta",
    "gamma",
    "loglik",
    "grid",
    "distribution",
    "density",
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
def logistic_ref() -> dict[str, np.ndarray]:
    paths = {k: REF_DIR / f"{_PREFIX}{k}.txt" for k in _REF_FILES}
    if not all(p.exists() for p in paths.values()):
        pytest.skip(
            f"{_PREFIX}* fixtures not generated — run "
            "Rscript reference/generate_reference.R"
        )
    return {k: np.loadtxt(p) for k, p in paths.items()}


def _theta_r_mltpy_layout(logistic_ref: dict[str, np.ndarray]) -> np.ndarray:
    """R's coef(fit) reordered to mltpy's ``[vec_C(Θ) | γ]``.

    R orders vec(Θ) y-fastest (``Bs_i(y):Bs_j(x)``), i.e. column-major;
    mltpy's vec_C is row-major.
    """
    p, q = _ORDER_Y + 1, _ORDER_X + 1
    theta_r = logistic_ref["theta"]
    return np.concatenate([theta_r[: p * q].reshape(q, p).T.ravel(), theta_r[p * q :]])


@pytest.fixture(scope="module")
def logistic_fit(logistic_ref: dict[str, np.ndarray]) -> Colr:
    """Default (auglag) fit on the R data.

    The MLE is strictly interior, so every block — γ, Θ and the predictions
    built from them — meets the rtol=1e-6 R tolerance (unlike the
    ``stram_bs_bs_mev`` Coxph fixture, whose Θ needs a looser auglag bound).
    """
    a, b = logistic_ref["y_support"]
    model = Colr(
        support=(float(a), float(b)),
        order=_ORDER_Y,
        interacting=_x_basis(),
        scaling=logistic_ref["x_s_train"].reshape(-1, 1),
        optimizer_config=OptimizerConfig(random_state=0),
    )
    with pytest.warns(UserWarning, match="experimental"):
        model.fit(logistic_ref["y_train"], X=logistic_ref["x_train"])
    return model


class TestRParity:
    def test_converged(self, logistic_fit: Colr) -> None:
        assert logistic_fit.result_ is not None
        assert logistic_fit.result_.converged

    def test_gamma_matches_r(
        self, logistic_fit: Colr, logistic_ref: dict[str, np.ndarray]
    ) -> None:
        gamma_r = np.atleast_1d(logistic_ref["gamma"])
        assert_allclose(logistic_fit.gamma_, gamma_r, rtol=1e-6, atol=1e-10)

    def test_loglik_matches_r(
        self, logistic_fit: Colr, logistic_ref: dict[str, np.ndarray]
    ) -> None:
        result = logistic_fit.result_
        assert result is not None
        assert_allclose(
            result.log_likelihood,
            float(logistic_ref["loglik"]),
            rtol=1e-6,
            atol=1e-10,
        )

    def test_loglik_at_r_theta_matches_r(
        self, logistic_fit: Colr, logistic_ref: dict[str, np.ndarray]
    ) -> None:
        """mltpy's likelihood evaluated at R's θ reproduces R's logLik."""
        model = logistic_fit
        ll = log_likelihood(
            _theta_r_mltpy_layout(logistic_ref),
            model.basis,
            logistic_ref["y_train"],
            logistic_ref["x_train"].reshape(-1, 1),
            CensoringType.NONE,
            base_distribution="logistic",
            scaling=logistic_ref["x_s_train"].reshape(-1, 1),
        )
        assert_allclose(ll, float(logistic_ref["loglik"]), rtol=1e-10, atol=1e-10)

    def test_theta_matches_r(
        self, logistic_fit: Colr, logistic_ref: dict[str, np.ndarray]
    ) -> None:
        assert_allclose(
            logistic_fit.theta_,
            _theta_r_mltpy_layout(logistic_ref),
            rtol=1e-6,
            atol=1e-10,
        )

    def test_theta_matrix_matches_r(
        self, logistic_fit: Colr, logistic_ref: dict[str, np.ndarray]
    ) -> None:
        """Theta_ is R's column-major vec(Θ) reshaped to (p, q)."""
        p, q = _ORDER_Y + 1, _ORDER_X + 1
        theta_r = logistic_ref["theta"][: p * q].reshape(q, p).T
        assert_allclose(logistic_fit.Theta_, theta_r, rtol=1e-6, atol=1e-10)

    @pytest.mark.parametrize("what", ["distribution", "density"])
    def test_predict_matches_r(
        self,
        logistic_fit: Colr,
        logistic_ref: dict[str, np.ndarray],
        what: Literal["distribution", "density"],
    ) -> None:
        """predict() with the parallel X_scale_new= matches R row-wise."""
        grid = logistic_ref["grid"]
        y, x, x_s = grid[:, 0], grid[:, 1], grid[:, 2:3]
        got = logistic_fit.predict(y, X_new=x, X_scale_new=x_s, what=what)
        assert_allclose(got, logistic_ref[what], rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Convenience surface (synthetic data)
# ---------------------------------------------------------------------------


def _synthetic(
    n: int = 200, seed: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Exact responses from logit F = exp(0.2·x_s)·(2(y − 4.5) + 0.8(x − 0.5))."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    x_s = rng.normal(size=(n, 1))
    k = np.exp(0.2 * x_s[:, 0])
    y = 4.5 - 0.4 * (x - 0.5) + rng.logistic(size=n) / (2.0 * k)
    return y, x, x_s, (float(y.min() - 0.1), float(y.max() + 0.1))


def _colr(support: tuple[float, float], x_s: np.ndarray | None = None) -> Colr:
    return Colr(
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
) -> Colr:
    y, x, x_s, support = synthetic
    model = _colr(support, x_s)
    with pytest.warns(UserWarning, match="experimental"):
        model.fit(y, X=x)
    return model


@pytest.fixture(scope="module")
def fitted_unscaled(
    synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
) -> Colr:
    y, x, _, support = synthetic
    return _colr(support).fit(y, X=x)


def _analytic_h(
    model: Colr, y: np.ndarray, x: np.ndarray, x_s: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """h and ∂h/∂y from the tensor basis, Θ and γ: (Z·vec Θ)·exp(0.5·x_s·γ)."""
    assert isinstance(model.basis, InteractionBasis)
    assert model.Theta_ is not None
    vec_theta = model.Theta_.ravel()
    f = 1.0 if x_s is None else np.exp(0.5 * (x_s @ model.gamma_))
    h = (model.basis.evaluate(y, x) @ vec_theta) * f
    hp = (model.basis.derivative(y, x) @ vec_theta) * f
    return h, hp


def _dlogis(z: np.ndarray) -> np.ndarray:
    return expit(z) * expit(-z)


_Y_GRID = np.array([3.6, 4.0, 4.4, 4.8, 5.0, 3.8])
_X_GRID = np.array([0.1, 0.4, 0.9, 0.5, 0.2, 0.7])


class TestConvenienceSurface:
    def test_builds_scaled_interaction_model(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
    ) -> None:
        _, _, x_s, support = synthetic
        model = _colr(support, x_s)
        assert isinstance(model.basis, InteractionBasis)
        assert model.base_distribution == "logistic"
        assert model.censoring == CensoringType.NONE
        assert model.scaling is not None
        assert_allclose(model.scaling, x_s)

    def test_matches_base_class_path(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        fitted_synthetic: Colr,
    ) -> None:
        """Colr is a thin wrapper over the scaled InteractionBasis CTM."""
        y, x, x_s, support = synthetic
        ref = ConditionalTransformationModel(
            basis=InteractionBasis(
                y_basis=BernsteinBasis(order=_ORDER_Y, support=support),
                x_basis=_x_basis(),
            ),
            censoring=CensoringType.NONE,
            scaling=x_s,
            base_distribution="logistic",
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        with pytest.warns(UserWarning, match="experimental"):
            ref.fit(y, X=x)
        assert_allclose(fitted_synthetic.theta_, ref.theta_, rtol=1e-12, atol=0)

    def test_exposes_theta_and_gamma(self, fitted_synthetic: Colr) -> None:
        p, q = _ORDER_Y + 1, _ORDER_X + 1
        assert fitted_synthetic.Theta_ is not None
        assert fitted_synthetic.Theta_.shape == (p, q)
        assert fitted_synthetic.gamma_.shape == (1,)
        assert fitted_synthetic.theta_ is not None
        assert fitted_synthetic.theta_.shape == (p * q + 1,)
        # Θ columns are non-decreasing in y (column-wise monotonicity).
        assert np.all(np.diff(fitted_synthetic.Theta_, axis=0) >= -1e-8)

    @pytest.mark.parametrize("x_s_value", [-1.5, 0.0, 2.0])
    def test_distribution_matches_analytic(
        self, fitted_synthetic: Colr, x_s_value: float
    ) -> None:
        """Logistic link: F(y | x, x_s) = expit(h)."""
        x_s = np.full((_Y_GRID.size, 1), x_s_value)
        h, _ = _analytic_h(fitted_synthetic, _Y_GRID, _X_GRID, x_s)
        got = fitted_synthetic.predict(
            _Y_GRID, X_new=_X_GRID, X_scale_new=x_s, what="distribution"
        )
        assert_allclose(got, expit(h), rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("x_s_value", [-1.5, 0.0, 2.0])
    def test_density_matches_analytic(
        self, fitted_synthetic: Colr, x_s_value: float
    ) -> None:
        """Logistic link: f(y | x, x_s) = expit(h)·expit(−h)·h'."""
        x_s = np.full((_Y_GRID.size, 1), x_s_value)
        h, hp = _analytic_h(fitted_synthetic, _Y_GRID, _X_GRID, x_s)
        got = fitted_synthetic.predict(
            _Y_GRID, X_new=_X_GRID, X_scale_new=x_s, what="density"
        )
        assert_allclose(got, _dlogis(h) * hp, rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("x_s_value", [-1.5, 0.0, 2.0])
    def test_logodds_is_scaled_transformation(
        self, fitted_synthetic: Colr, x_s_value: float
    ) -> None:
        """Logistic link: logit F(y | x, x_s) = h, the scaled transformation."""
        x_s = np.full((_Y_GRID.size, 1), x_s_value)
        h, _ = _analytic_h(fitted_synthetic, _Y_GRID, _X_GRID, x_s)
        got = fitted_synthetic.predict(
            _Y_GRID, X_new=_X_GRID, X_scale_new=x_s, what="logodds"
        )
        assert_allclose(got, h, rtol=1e-10, atol=1e-12)

    @pytest.mark.filterwarnings("ignore:predict.*exceeds:UserWarning")
    @pytest.mark.parametrize("x_s_value", [-60.0, 60.0])
    def test_extreme_scaling_stays_finite(
        self, fitted_synthetic: Colr, x_s_value: float
    ) -> None:
        """|x_s| large enough to push h past the ±30 clip stays well-defined."""
        x_s = np.full((_Y_GRID.size, 1), x_s_value)
        cdf = fitted_synthetic.predict(
            _Y_GRID, X_new=_X_GRID, X_scale_new=x_s, what="distribution"
        )
        pdf = fitted_synthetic.predict(
            _Y_GRID, X_new=_X_GRID, X_scale_new=x_s, what="density"
        )
        assert np.all(np.isfinite(cdf)) and np.all((cdf >= 0.0) & (cdf <= 1.0))
        assert np.all(np.isfinite(pdf)) and np.all(pdf >= 0.0)

    @pytest.mark.parametrize("x_s_value", [-1.5, 0.0, 2.0])
    def test_quantile_inverts_distribution(
        self, fitted_synthetic: Colr, x_s_value: float
    ) -> None:
        """Quantiles honour X_scale_new: F(Q(p | x, x_s) | x, x_s) = p."""
        probs = np.linspace(0.2, 0.8, _X_GRID.size)
        x_s = np.full((_X_GRID.size, 1), x_s_value)
        q = fitted_synthetic.predict(
            probs, X_new=_X_GRID, X_scale_new=x_s, what="quantile"
        )
        cdf = fitted_synthetic.predict(
            q, X_new=_X_GRID, X_scale_new=x_s, what="distribution"
        )
        assert_allclose(cdf, probs, rtol=1e-6, atol=1e-8)

    def test_scaling_changes_distribution(self, fitted_synthetic: Colr) -> None:
        """Different x_s values give different CDFs at the same x."""
        lo = np.full((_Y_GRID.size, 1), -1.5)
        hi = np.full((_Y_GRID.size, 1), 2.0)
        f_lo = fitted_synthetic.predict(_Y_GRID, X_new=_X_GRID, X_scale_new=lo)
        f_hi = fitted_synthetic.predict(_Y_GRID, X_new=_X_GRID, X_scale_new=hi)
        assert np.max(np.abs(f_lo - f_hi)) > 1e-3

    @pytest.mark.parametrize("what", ["distribution", "density", "quantile"])
    def test_predict_requires_x_scale_new(
        self,
        fitted_synthetic: Colr,
        what: Literal["distribution", "density", "quantile"],
    ) -> None:
        y_new = np.full(_X_GRID.size, 0.5) if what == "quantile" else _Y_GRID
        with pytest.raises(ValueError, match="X_scale_new must be"):
            fitted_synthetic.predict(y_new, X_new=_X_GRID, what=what)

    def test_all_exact_censored_data_accepted(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        fitted_synthetic: Colr,
    ) -> None:
        """A CensoredData with no censored rows is exact data."""
        y, x, x_s, support = synthetic
        cd = CensoredData.right_censored(y, censored=np.zeros(y.size, dtype=bool))
        model = _colr(support, x_s)
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
        """Censored rows raise a clear error, not a NaN log-likelihood."""
        y, x, x_s, support = synthetic
        censored = np.zeros(y.size, dtype=bool)
        censored[::5] = True
        cd = CensoredData.right_censored(y, censored=censored)
        model = _colr(support, x_s if scaled else None)
        with pytest.raises(NotImplementedError, match="exact"):
            model.fit(cd, X=x)

    def test_fitted_transformation_not_available(self, fitted_synthetic: Colr) -> None:
        """No contiguous theta_b block exists on the interaction path."""
        with pytest.raises(NotImplementedError, match="InteractionBasis"):
            fitted_synthetic.fitted_transformation(_Y_GRID)

    def test_summary_tabulates_gamma(self, fitted_synthetic: Colr) -> None:
        text = fitted_synthetic.summary()
        assert "Model:        Colr" in text
        assert "Scaling coefficients:" in text


class TestInteractingOnly:
    """``Colr(interacting=...)`` without ``scaling=``: θ = vec_C(Θ)."""

    def test_exposes_theta_not_gamma(self, fitted_unscaled: Colr) -> None:
        assert fitted_unscaled.Theta_ is not None
        assert fitted_unscaled.Theta_.shape == (_ORDER_Y + 1, _ORDER_X + 1)
        assert fitted_unscaled.theta_ is not None
        assert fitted_unscaled.theta_.size == (_ORDER_Y + 1) * (_ORDER_X + 1)
        with pytest.raises(ValueError, match="gamma_ is undefined"):
            _ = fitted_unscaled.gamma_

    def test_no_experimental_warning(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
    ) -> None:
        """Only the scaled combination is flagged experimental."""
        y, x, _, support = synthetic
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _colr(support).fit(y, X=x)

    def test_matches_base_class_path(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        fitted_unscaled: Colr,
    ) -> None:
        y, x, _, support = synthetic
        ref = ConditionalTransformationModel(
            basis=InteractionBasis(
                y_basis=BernsteinBasis(order=_ORDER_Y, support=support),
                x_basis=_x_basis(),
            ),
            base_distribution="logistic",
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        ).fit(y, X=x)
        assert_allclose(fitted_unscaled.theta_, ref.theta_, rtol=1e-12, atol=0)

    @pytest.mark.parametrize("x_value", [0.1, 0.5, 0.9])
    def test_distribution_matches_analytic(
        self, fitted_unscaled: Colr, x_value: float
    ) -> None:
        x = np.full(_Y_GRID.size, x_value)
        h, _ = _analytic_h(fitted_unscaled, _Y_GRID, x, None)
        got = fitted_unscaled.predict(_Y_GRID, X_new=x, what="distribution")
        assert_allclose(got, expit(h), rtol=1e-10, atol=1e-14)

    def test_odds_vary_with_y(self, fitted_unscaled: Colr) -> None:
        """Non-proportional odds: the log-odds gap between x values depends on y."""
        y = np.array([3.8, 5.0])
        lo = fitted_unscaled.predict(y, X_new=np.full(2, 0.1), what="logodds")
        hi = fitted_unscaled.predict(y, X_new=np.full(2, 0.9), what="logodds")
        gap = hi - lo
        assert abs(gap[1] - gap[0]) > 1e-3

    def test_ordinal_x_basis(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
    ) -> None:
        """A stratified (ordinal x-basis) Colr fits with one Θ column per cut.

        ``OrdinalBasis(3)`` one-hot encodes the cut labels ``{1, 2}``, giving
        two Θ columns.
        """
        y, x, _, support = synthetic
        strata = np.where(x > 0.5, 2.0, 1.0)
        model = Colr(
            support=support,
            order=_ORDER_Y,
            interacting=OrdinalBasis(3),
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        ).fit(y, X=strata)
        assert model.Theta_ is not None
        assert model.Theta_.shape == (_ORDER_Y + 1, 2)


class TestBackwardCompatibility:
    def test_shift_path_unchanged_without_interacting(self) -> None:
        """interacting=None keeps the BernsteinBasis shift model."""
        model = Colr(support=(0.0, 1.0), order=3)
        assert isinstance(model.basis, BernsteinBasis)
        assert model.base_distribution == "logistic"
        assert model.censoring == CensoringType.NONE
