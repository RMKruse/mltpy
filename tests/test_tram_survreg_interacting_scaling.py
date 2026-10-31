"""``Survreg(interacting=..., scaling=...)`` — issue #110.

The parametric log-time survival model with an interacting (non-proportional)
transformation and an optional scaled baseline,

    F_Z⁻¹ F(t | x, x_s) = ([a(log t) ⊗ b(x)]ᵀ vec(Θ)) · exp(0.5 · x_s · γ),

reached through the ``Survreg`` convenience surface rather than
``ConditionalTransformationModel(basis=InteractionBasis(LogBernsteinBasis(...),
...), scaling=...)``.  ``F_Z`` is the family's link: weibull ↔ minimum extreme
value, lognormal ↔ normal, loglogistic ↔ logistic.  Parameter layout is
``theta_ = [vec_C(Θ) | γ]`` (ADR 0003, no β shift block), or ``vec_C(Θ)``
without ``scaling=``.  The interaction likelihood is exact-data only, so
censored rows are rejected with a clear error.

R reference data (``reference/stram_logbs_bs_<dist>_*``) is produced by
``reference/generate_reference.R`` via ``mlt::ctm(response = Bernstein-y with
log_first = TRUE, interacting = Bernstein-x, scaling = ~x_s)`` — the ``stram``
route confirmed in ADR 0003 — and cross-checked against the literal
``tram::Survreg(y | x ~ 1 | x_s, dist = <dist>)`` stram fit, whose γ is stored
separately.  γ is sign- and magnitude-aligned with mltpy (no flip).
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mltpy import (
    ConditionalTransformationModel,
    InteractionBasis,
    OptimizerConfig,
    log_likelihood,
)
from mltpy.basis import (
    BernsteinBasis,
    LegendreBasis,
    LogBernsteinBasis,
    OneHotBasis,
    OrdinalBasis,
    PolynomialBasis,
)
from mltpy.tram import Survreg, SurvregDistribution
from mltpy.variables import CensoredData, CensoringType

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"
_REF_FILES = [
    "y_train",
    "x_train",
    "x_s_train",
    "y_support",
    "theta",
    "gamma",
    "gamma_tram",
    "loglik",
    "grid",
    "survivor",
    "density",
    "hazard",
    "qgrid",
    "quantile",
]

# Survreg family → mltpy base distribution.
_BASE = {
    "weibull": "min_extreme_value",
    "lognormal": "normal",
    "loglogistic": "logistic",
}
_DISTS: list[SurvregDistribution] = ["weibull", "lognormal", "loglogistic"]

# Bases mirroring the R block: log-Bernstein-y order 1 (p = 2, tram::Survreg's
# affine log-time baseline) ⊗ Bernstein-x order 1 (q = 2) on x ∈ [0, 1].
_ORDER_Y = 1
_ORDER_X = 1

_EXPERIMENTAL = pytest.mark.filterwarnings(
    "ignore:scaling= with InteractionBasis is an experimental path"
)


def _x_basis() -> BernsteinBasis:
    return BernsteinBasis(order=_ORDER_X, support=(0.0, 1.0))


# ---------------------------------------------------------------------------
# R parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=_DISTS)
def survreg_ref(request: pytest.FixtureRequest) -> dict[str, object]:
    dist: str = request.param
    prefix = f"stram_logbs_bs_{dist}_"
    paths = {k: REF_DIR / f"{prefix}{k}.txt" for k in _REF_FILES}
    if not all(p.exists() for p in paths.values()):
        pytest.skip(
            f"{prefix}* fixtures not generated — run "
            "Rscript reference/generate_reference.R"
        )
    ref: dict[str, object] = {k: np.loadtxt(p) for k, p in paths.items()}
    ref["dist"] = dist
    return ref


def _arr(ref: dict[str, object], key: str) -> np.ndarray:
    value = ref[key]
    assert isinstance(value, np.ndarray)
    return value


def _theta_r_mltpy_layout(ref: dict[str, object]) -> np.ndarray:
    """R's coef(fit) reordered to mltpy's ``[vec_C(Θ) | γ]``.

    R orders vec(Θ) y-fastest (``Bs_i(y):Bs_j(x)``), i.e. column-major;
    mltpy's vec_C is row-major.
    """
    p, q = _ORDER_Y + 1, _ORDER_X + 1
    theta_r = _arr(ref, "theta")
    return np.concatenate([theta_r[: p * q].reshape(q, p).T.ravel(), theta_r[p * q :]])


@pytest.fixture(scope="module")
def survreg_fit(survreg_ref: dict[str, object]) -> Survreg:
    """Default (auglag) fit on the R data; the MLE is strictly interior."""
    a, b = _arr(survreg_ref, "y_support")
    dist = survreg_ref["dist"]
    assert dist in _BASE
    model = Survreg(
        support=(float(a), float(b)),
        distribution=dist,  # type: ignore[arg-type]
        order=_ORDER_Y,
        interacting=_x_basis(),
        scaling=_arr(survreg_ref, "x_s_train").reshape(-1, 1),
        optimizer_config=OptimizerConfig(random_state=0),
    )
    with pytest.warns(UserWarning, match="experimental"):
        model.fit(_arr(survreg_ref, "y_train"), X=_arr(survreg_ref, "x_train"))
    return model


class TestRParity:
    def test_converged(self, survreg_fit: Survreg) -> None:
        assert survreg_fit.result_ is not None
        assert survreg_fit.result_.converged

    def test_gamma_matches_r(
        self, survreg_fit: Survreg, survreg_ref: dict[str, object]
    ) -> None:
        gamma_r = np.atleast_1d(_arr(survreg_ref, "gamma"))
        assert_allclose(survreg_fit.gamma_, gamma_r, rtol=1e-6, atol=1e-10)

    def test_gamma_matches_tram_stram(
        self, survreg_fit: Survreg, survreg_ref: dict[str, object]
    ) -> None:
        """γ matches the literal ``tram::Survreg(y | x ~ 1 | x_s)`` stram fit.

        tram's response basis (1, log t) and interacting design (1, x) span
        the same spaces as the order-1 log-Bernstein-y and Bernstein-x bases,
        so the MLE is shared.
        """
        gamma_tram = np.atleast_1d(_arr(survreg_ref, "gamma_tram"))
        assert_allclose(survreg_fit.gamma_, gamma_tram, rtol=1e-6, atol=1e-10)

    def test_loglik_matches_r(
        self, survreg_fit: Survreg, survreg_ref: dict[str, object]
    ) -> None:
        result = survreg_fit.result_
        assert result is not None
        assert_allclose(
            result.log_likelihood,
            float(_arr(survreg_ref, "loglik")),
            rtol=1e-6,
            atol=1e-10,
        )

    def test_loglik_at_r_theta_matches_r(
        self, survreg_fit: Survreg, survreg_ref: dict[str, object]
    ) -> None:
        """mltpy's likelihood evaluated at R's θ reproduces R's logLik.

        Pins the log-time basis mapping (``log_first = TRUE`` on
        ``[log a, log b]``) and the 1/t Jacobian independently of the
        optimiser.
        """
        model = survreg_fit
        ll = log_likelihood(
            _theta_r_mltpy_layout(survreg_ref),
            model.basis,
            _arr(survreg_ref, "y_train"),
            _arr(survreg_ref, "x_train").reshape(-1, 1),
            CensoringType.NONE,
            base_distribution=model.base_distribution,
            scaling=_arr(survreg_ref, "x_s_train").reshape(-1, 1),
        )
        assert_allclose(ll, float(_arr(survreg_ref, "loglik")), rtol=1e-10, atol=1e-10)

    def test_theta_matches_r(
        self, survreg_fit: Survreg, survreg_ref: dict[str, object]
    ) -> None:
        assert_allclose(
            survreg_fit.theta_,
            _theta_r_mltpy_layout(survreg_ref),
            rtol=1e-6,
            atol=1e-10,
        )

    def test_theta_matrix_matches_r(
        self, survreg_fit: Survreg, survreg_ref: dict[str, object]
    ) -> None:
        """Theta_ is R's column-major vec(Θ) reshaped to (p, q)."""
        p, q = _ORDER_Y + 1, _ORDER_X + 1
        theta_r = _arr(survreg_ref, "theta")[: p * q].reshape(q, p).T
        assert_allclose(survreg_fit.Theta_, theta_r, rtol=1e-6, atol=1e-10)

    def test_survival_matches_r(
        self, survreg_fit: Survreg, survreg_ref: dict[str, object]
    ) -> None:
        """survival() with the parallel X_scale= matches R row-wise."""
        grid = _arr(survreg_ref, "grid")
        t, x, x_s = grid[:, 0], grid[:, 1], grid[:, 2:3]
        got = survreg_fit.survival(t, X=x, X_scale=x_s)
        assert_allclose(got, _arr(survreg_ref, "survivor"), rtol=1e-6, atol=1e-10)

    def test_hazard_matches_r(
        self, survreg_fit: Survreg, survreg_ref: dict[str, object]
    ) -> None:
        """hazard() with the parallel X_scale= matches R row-wise."""
        grid = _arr(survreg_ref, "grid")
        t, x, x_s = grid[:, 0], grid[:, 1], grid[:, 2:3]
        got = survreg_fit.hazard(t, X=x, X_scale=x_s)
        assert_allclose(got, _arr(survreg_ref, "hazard"), rtol=1e-6, atol=1e-10)

    def test_density_matches_r(
        self, survreg_fit: Survreg, survreg_ref: dict[str, object]
    ) -> None:
        """The density carries the 1/t Jacobian of the log-time basis."""
        grid = _arr(survreg_ref, "grid")
        t, x, x_s = grid[:, 0], grid[:, 1], grid[:, 2:3]
        got = survreg_fit.predict(t, X_new=x, X_scale_new=x_s, what="density")
        assert_allclose(got, _arr(survreg_ref, "density"), rtol=1e-6, atol=1e-10)

    def test_quantile_matches_r(
        self, survreg_fit: Survreg, survreg_ref: dict[str, object]
    ) -> None:
        """Quantiles invert on the original time scale, not on log t."""
        qgrid = _arr(survreg_ref, "qgrid")
        probs, x, x_s = qgrid[:, 0], qgrid[:, 1], qgrid[:, 2:3]
        got = survreg_fit.predict(probs, X_new=x, X_scale_new=x_s, what="quantile")
        assert_allclose(got, _arr(survreg_ref, "quantile"), rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Convenience surface (synthetic Weibull data)
# ---------------------------------------------------------------------------


def _synthetic(
    n: int = 200, seed: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Exact Weibull times: log[-log S] = exp(0.2·x_s)·(2(log t − 1) + 0.8(x − 0.5))."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    x_s = rng.normal(size=(n, 1))
    k = np.exp(0.2 * x_s[:, 0])
    g = np.log(-np.log(rng.uniform(size=n)))
    t = np.exp(1.0 - 0.4 * (x - 0.5) + g / (2.0 * k))
    return t, x, x_s, (float(t.min() * 0.9), float(t.max() * 1.1))


def _survreg(
    support: tuple[float, float],
    x_s: np.ndarray | None = None,
    distribution: SurvregDistribution = "weibull",
) -> Survreg:
    return Survreg(
        support=support,
        distribution=distribution,
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
) -> Survreg:
    t, x, x_s, support = synthetic
    model = _survreg(support, x_s)
    with pytest.warns(UserWarning, match="experimental"):
        model.fit(t, X=x)
    return model


@pytest.fixture(scope="module")
def fitted_unscaled(
    synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
) -> Survreg:
    t, x, _, support = synthetic
    return _survreg(support).fit(t, X=x)


def _analytic_h(
    model: Survreg, t: np.ndarray, x: np.ndarray, x_s: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """h and ∂h/∂t from the tensor basis, Θ and γ: (Z·vec Θ)·exp(0.5·x_s·γ)."""
    assert isinstance(model.basis, InteractionBasis)
    assert model.Theta_ is not None
    vec_theta = model.Theta_.ravel()
    f = 1.0 if x_s is None else np.exp(0.5 * (x_s @ model.gamma_))
    h = (model.basis.evaluate(t, x) @ vec_theta) * f
    hp = (model.basis.derivative(t, x) @ vec_theta) * f
    return h, hp


_T_GRID = np.array([1.2, 2.0, 2.7, 3.5, 4.5, 1.6])
_X_GRID = np.array([0.1, 0.4, 0.9, 0.5, 0.2, 0.7])


class TestConvenienceSurface:
    @pytest.mark.parametrize("distribution", _DISTS)
    def test_builds_scaled_interaction_model(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        distribution: SurvregDistribution,
    ) -> None:
        """The y-basis stays on log time; the family fixes the link."""
        _, _, x_s, support = synthetic
        model = _survreg(support, x_s, distribution)
        assert isinstance(model.basis, InteractionBasis)
        assert isinstance(model.basis.y_basis, LogBernsteinBasis)
        assert model.basis.y_basis.order == _ORDER_Y
        assert model.basis.y_basis.support == support
        assert model.base_distribution == _BASE[distribution]
        assert model.censoring == CensoringType.RIGHT
        assert model.scaling is not None
        assert_allclose(model.scaling, x_s)

    def test_matches_base_class_path(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        fitted_synthetic: Survreg,
    ) -> None:
        """Survreg is a thin wrapper over the scaled InteractionBasis CTM."""
        t, x, x_s, support = synthetic
        ref = ConditionalTransformationModel(
            basis=InteractionBasis(
                y_basis=LogBernsteinBasis(order=_ORDER_Y, support=support),
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

    def test_exposes_theta_and_gamma(self, fitted_synthetic: Survreg) -> None:
        p, q = _ORDER_Y + 1, _ORDER_X + 1
        assert fitted_synthetic.Theta_ is not None
        assert fitted_synthetic.Theta_.shape == (p, q)
        assert fitted_synthetic.gamma_.shape == (1,)
        assert fitted_synthetic.theta_ is not None
        assert fitted_synthetic.theta_.shape == (p * q + 1,)
        # Θ columns are non-decreasing in log t (column-wise monotonicity).
        assert np.all(np.diff(fitted_synthetic.Theta_, axis=0) >= -1e-8)

    def test_recovers_true_gamma(self, fitted_synthetic: Survreg) -> None:
        """The simulation uses γ = 0.4; n = 200 recovers it to ~2 SE."""
        assert abs(float(fitted_synthetic.gamma_[0]) - 0.4) < 0.25

    @pytest.mark.parametrize("x_s_value", [-1.5, 0.0, 2.0])
    def test_survival_matches_analytic(
        self, fitted_synthetic: Survreg, x_s_value: float
    ) -> None:
        """Weibull link: S(t | x, x_s) = exp(−exp(h))."""
        x_s = np.full((_T_GRID.size, 1), x_s_value)
        h, _ = _analytic_h(fitted_synthetic, _T_GRID, _X_GRID, x_s)
        got = fitted_synthetic.survival(_T_GRID, X=_X_GRID, X_scale=x_s)
        assert_allclose(got, np.exp(-np.exp(h)), rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("x_s_value", [-1.5, 0.0, 2.0])
    def test_hazard_matches_analytic(
        self, fitted_synthetic: Survreg, x_s_value: float
    ) -> None:
        """Weibull link: λ(t | x, x_s) = exp(h)·h', with h' ∝ 1/t."""
        x_s = np.full((_T_GRID.size, 1), x_s_value)
        h, hp = _analytic_h(fitted_synthetic, _T_GRID, _X_GRID, x_s)
        got = fitted_synthetic.hazard(_T_GRID, X=_X_GRID, X_scale=x_s)
        assert_allclose(got, np.exp(h) * hp, rtol=1e-10, atol=1e-14)

    @pytest.mark.filterwarnings("ignore:predict.*exceeds:UserWarning")
    @pytest.mark.parametrize("x_s_value", [-60.0, 60.0])
    def test_extreme_scaling_stays_finite(
        self, fitted_synthetic: Survreg, x_s_value: float
    ) -> None:
        """|x_s| large enough to push h past the ±30 clip stays well-defined."""
        x_s = np.full((_T_GRID.size, 1), x_s_value)
        surv = fitted_synthetic.survival(_T_GRID, X=_X_GRID, X_scale=x_s)
        haz = fitted_synthetic.hazard(_T_GRID, X=_X_GRID, X_scale=x_s)
        assert np.all(np.isfinite(surv)) and np.all((surv >= 0.0) & (surv <= 1.0))
        assert np.all(np.isfinite(haz)) and np.all(haz >= 0.0)

    @pytest.mark.parametrize("x_s_value", [-1.5, 0.0, 2.0])
    def test_quantile_inverts_distribution(
        self, fitted_synthetic: Survreg, x_s_value: float
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

    def test_scaling_changes_survival(self, fitted_synthetic: Survreg) -> None:
        """Different x_s values give different survival curves at the same x."""
        lo = np.full((_T_GRID.size, 1), -1.5)
        hi = np.full((_T_GRID.size, 1), 2.0)
        s_lo = fitted_synthetic.survival(_T_GRID, X=_X_GRID, X_scale=lo)
        s_hi = fitted_synthetic.survival(_T_GRID, X=_X_GRID, X_scale=hi)
        assert np.max(np.abs(s_lo - s_hi)) > 1e-3

    @pytest.mark.parametrize("method", ["survival", "hazard"])
    def test_domain_methods_require_x_scale(
        self, fitted_synthetic: Survreg, method: Literal["survival", "hazard"]
    ) -> None:
        with pytest.raises(ValueError, match="X_scale_new must be"):
            getattr(fitted_synthetic, method)(_T_GRID, X=_X_GRID)

    @pytest.mark.parametrize("what", ["distribution", "density", "quantile"])
    def test_predict_requires_x_scale_new(
        self,
        fitted_synthetic: Survreg,
        what: Literal["distribution", "density", "quantile"],
    ) -> None:
        y_new = np.full(_X_GRID.size, 0.5) if what == "quantile" else _T_GRID
        with pytest.raises(ValueError, match="X_scale_new must be"):
            fitted_synthetic.predict(y_new, X_new=_X_GRID, what=what)

    def test_all_exact_censored_data_accepted(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        fitted_synthetic: Survreg,
    ) -> None:
        """A CensoredData with no censored rows is exact data."""
        t, x, x_s, support = synthetic
        cd = CensoredData.right_censored(t, censored=np.zeros(t.size, dtype=bool))
        model = _survreg(support, x_s)
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
        t, x, x_s, support = synthetic
        censored = np.zeros(t.size, dtype=bool)
        censored[::5] = True
        cd = CensoredData.right_censored(t, censored=censored)
        model = _survreg(support, x_s if scaled else None)
        with pytest.raises(NotImplementedError, match="exact"):
            model.fit(cd, X=x)

    def test_fitted_transformation_not_available(
        self, fitted_synthetic: Survreg
    ) -> None:
        """No contiguous theta_b block exists on the interaction path."""
        with pytest.raises(NotImplementedError, match="InteractionBasis"):
            fitted_synthetic.fitted_transformation(_T_GRID)

    def test_summary_tabulates_gamma(self, fitted_synthetic: Survreg) -> None:
        text = fitted_synthetic.summary()
        assert "Model:        Survreg" in text
        assert "Scaling coefficients:" in text

    def test_repr(self, fitted_synthetic: Survreg) -> None:
        text = repr(fitted_synthetic)
        assert text.startswith(f"Survreg(order={_ORDER_Y}, ")
        assert "fitted=True" in text


class TestInteractingOnly:
    """``Survreg(interacting=...)`` without ``scaling=``: θ = vec_C(Θ)."""

    def test_exposes_theta_not_gamma(self, fitted_unscaled: Survreg) -> None:
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
        t, x, _, support = synthetic
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _survreg(support).fit(t, X=x)

    def test_matches_base_class_path(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        fitted_unscaled: Survreg,
    ) -> None:
        t, x, _, support = synthetic
        ref = ConditionalTransformationModel(
            basis=InteractionBasis(
                y_basis=LogBernsteinBasis(order=_ORDER_Y, support=support),
                x_basis=_x_basis(),
            ),
            censoring=CensoringType.RIGHT,
            base_distribution="min_extreme_value",
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        ).fit(t, X=x)
        assert_allclose(fitted_unscaled.theta_, ref.theta_, rtol=1e-12, atol=0)

    @pytest.mark.parametrize("x_value", [0.1, 0.5, 0.9])
    def test_survival_matches_analytic(
        self, fitted_unscaled: Survreg, x_value: float
    ) -> None:
        x = np.full(_T_GRID.size, x_value)
        h, _ = _analytic_h(fitted_unscaled, _T_GRID, x, None)
        got = fitted_unscaled.survival(_T_GRID, X=x)
        assert_allclose(got, np.exp(-np.exp(h)), rtol=1e-10, atol=1e-14)

    def test_survival_rejects_x_scale(self, fitted_unscaled: Survreg) -> None:
        x_s = np.zeros((_T_GRID.size, 1))
        with pytest.raises(ValueError, match="X_scale_new must be None"):
            fitted_unscaled.survival(_T_GRID, X=_X_GRID, X_scale=x_s)

    def test_hazards_not_proportional(self, fitted_unscaled: Survreg) -> None:
        """Non-proportional hazards: the log-hazard gap between x values varies."""
        t = np.array([1.2, 4.5])
        lo = fitted_unscaled.hazard(t, X=np.full(2, 0.1))
        hi = fitted_unscaled.hazard(t, X=np.full(2, 0.9))
        gap = np.log(hi) - np.log(lo)
        assert abs(gap[1] - gap[0]) > 1e-3

    @pytest.mark.parametrize("distribution", ["lognormal", "loglogistic"])
    def test_other_families_fit(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        distribution: SurvregDistribution,
    ) -> None:
        t, x, _, support = synthetic
        model = _survreg(support, None, distribution).fit(t, X=x)
        assert model.result_ is not None
        assert np.isfinite(model.result_.log_likelihood)
        surv = model.survival(_T_GRID, X=_X_GRID)
        assert np.all((surv > 0.0) & (surv < 1.0))

    @pytest.mark.parametrize(
        ("x_basis", "n_cols"),
        [(OrdinalBasis(3), 2), (OneHotBasis(K=2), 2)],
        ids=["ordinal", "onehot"],
    )
    def test_stratified_x_basis(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
        x_basis: OrdinalBasis | OneHotBasis,
        n_cols: int,
    ) -> None:
        """A stratified Survreg (tram's ``y | s ~ 1``) gets one Θ column per stratum.

        ``OrdinalBasis(3)`` one-hot encodes the cut labels ``{1, 2}`` and
        ``OneHotBasis(K=2)`` the labels ``{0, 1}``; both give two Θ columns.
        """
        t, x, _, support = synthetic
        strata = (x > 0.5).astype(float)
        if isinstance(x_basis, OrdinalBasis):
            strata = strata + 1.0
        model = Survreg(
            support=support,
            order=_ORDER_Y,
            interacting=x_basis,
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        ).fit(t, X=strata)
        assert model.Theta_ is not None
        assert model.Theta_.shape == (_ORDER_Y + 1, n_cols)


class TestErrors:
    @pytest.mark.parametrize(
        "x_basis",
        [
            PolynomialBasis(order=1, support=(0.0, 1.0)),
            LegendreBasis(order=1, support=(0.0, 1.0)),
        ],
        ids=["polynomial", "legendre"],
    )
    def test_non_partition_of_unity_x_basis_rejected(
        self, x_basis: PolynomialBasis | LegendreBasis
    ) -> None:
        """Column-wise monotonicity needs a non-negative partition of unity."""
        with pytest.raises(ValueError, match="partition of unity"):
            Survreg(support=(0.1, 10.0), interacting=x_basis)  # type: ignore[arg-type]

    def test_invalid_distribution_rejected(self) -> None:
        with pytest.raises(ValueError, match="distribution="):
            Survreg(
                support=(0.1, 10.0),
                distribution="exponential",  # type: ignore[arg-type]
                interacting=_x_basis(),
            )

    @pytest.mark.parametrize("support", [(0.0, 10.0), (-1.0, 10.0)])
    def test_non_positive_support_rejected(self, support: tuple[float, float]) -> None:
        """The log-time y-basis needs a strictly positive support."""
        with pytest.raises(ValueError, match="strictly positive"):
            Survreg(support=support, interacting=_x_basis())

    def test_interacting_without_x_rejected(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
    ) -> None:
        """The interacting covariate is required at fit time."""
        t, _, _, support = synthetic
        with pytest.raises(ValueError, match="requires X"):
            _survreg(support).fit(t)

    def test_empty_y_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one observation"):
            _survreg((0.1, 10.0)).fit(np.empty(0), X=np.empty(0))

    def test_time_outside_support_rejected(
        self,
        synthetic: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]],
    ) -> None:
        t, x, _, support = synthetic
        model = _survreg((support[0], float(np.median(t))))
        with pytest.raises(ValueError, match="outside support"):
            model.fit(t, X=x)


class TestBackwardCompatibility:
    def test_shift_path_unchanged_without_interacting(self) -> None:
        """interacting=None keeps the LogBernsteinBasis shift model."""
        model = Survreg(support=(0.1, 10.0), order=3)
        assert isinstance(model.basis, LogBernsteinBasis)
        assert model.base_distribution == "min_extreme_value"
        assert model.censoring == CensoringType.RIGHT

    def test_scaling_keyword_still_accepted(self) -> None:
        x_s = np.zeros((5, 1))
        model = Survreg(support=(0.1, 10.0), order=1, scaling=x_s)
        assert isinstance(model.basis, LogBernsteinBasis)
        assert model.scaling is not None
