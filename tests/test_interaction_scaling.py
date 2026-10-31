"""Scaling terms combined with the tensor-product InteractionBasis.

Covers the *non-proportional, heteroskedastic* CTM ratified in
``docs/adr/0003-scaling-with-interaction.md`` (issue #103): an
``InteractionBasis`` baseline ``(a(y) ⊗ b(x))ᵀ vec(Θ)`` scaled by
``exp(0.5 · x_s · γ)``, for exact data and the normal base distribution.

Parameter layout on this path is ``theta_ = [vec_C(Θ) | γ]`` (no additive
``β`` shift block — see ADR 0003 Decision 1).
"""

from __future__ import annotations

import pathlib
import warnings
from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import approx_fprime
from scipy.stats import norm

from mltpy import (
    MLT,
    ConditionalTransformationModel,
    InteractionBasis,
    InterceptBasis,
    OptimizerConfig,
    hessian,
    log_likelihood,
    negative_log_likelihood,
    score_matrix,
)
from mltpy.basis import BernsteinBasis
from mltpy.tram import _TramModel


def _scaled_interaction_problem(
    n: int = 200, seed: int = 42
) -> tuple[InteractionBasis, np.ndarray, np.ndarray, np.ndarray]:
    """Bernstein-y ⊗ Bernstein-x baseline with a single scaling column."""
    rng = np.random.default_rng(seed)
    y_basis = BernsteinBasis(order=3, support=(0.0, 5.0))
    x_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
    ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
    y = rng.uniform(0.1, 4.9, size=n)
    X = rng.uniform(0.05, 0.95, size=n)
    x_s = rng.normal(size=(n, 1))
    return ib, y, X, x_s


class TestScaledInteractionFit:
    def test_fit_exposes_theta_and_gamma(self) -> None:
        """End-to-end tracer: fit exact/normal data, read Θ and γ back."""
        ib, y, X, x_s = _scaled_interaction_problem()
        model = ConditionalTransformationModel(
            basis=ib,
            scaling=x_s,
            base_distribution="normal",
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        with pytest.warns(UserWarning, match="experimental"):
            model.fit(y, X)

        assert model.is_fitted_
        p, q = ib.n_y_params, ib.n_x_params
        assert model.Theta_ is not None
        assert model.Theta_.shape == (p, q)
        # gamma_ raises (NotFittedError / ValueError) rather than returning
        # None; by this point the fit succeeded, so it is the trailing γ block.
        assert model.gamma_.shape == (1,)
        assert model.theta_.shape == (p * q + 1,)
        assert np.isfinite(model.result_.log_likelihood)


class TestReductionToShift:
    def test_intercept_xbasis_matches_scaled_shift(self) -> None:
        """An InterceptBasis x-basis (q=1) collapses the interacting baseline
        to ``h_0(y)``, so a scaled InteractionBasis must reproduce the
        log-likelihood and γ of a scaled shift ``MLT`` with no covariates
        (ADR 0003 Decision 6, secondary check)."""
        rng = np.random.default_rng(7)
        n, order, support = 250, 4, (0.0, 5.0)
        y = rng.uniform(0.1, 4.9, size=n)
        x_s = rng.normal(size=(n, 1))

        shift = MLT(
            order=order,
            support=support,
            scaling=x_s,
            base_distribution="normal",
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        shift.fit(y)

        ib = InteractionBasis(
            y_basis=BernsteinBasis(order=order, support=support),
            x_basis=InterceptBasis(support=(0.0, 1.0)),
        )
        inter = ConditionalTransformationModel(
            basis=ib,
            scaling=x_s,
            base_distribution="normal",
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        with pytest.warns(UserWarning, match="experimental"):
            inter.fit(y, X=np.full((n, 1), 0.5))

        assert_allclose(
            inter.result_.log_likelihood,
            shift.result_.log_likelihood,
            rtol=1e-5,
            atol=1e-8,
        )
        assert_allclose(inter.gamma_, shift.gamma_, rtol=1e-4, atol=1e-6)
        # Θ has a single x-column equal to the shift baseline coefficients.
        assert_allclose(
            inter.Theta_[:, 0], shift.theta_[: order + 1], rtol=1e-4, atol=1e-6
        )


class TestScaledInteractionGradient:
    def test_analytic_gradient_matches_finite_difference(self) -> None:
        """Analytic ∂NLL/∂θ over θ = [vec_C(Θ) | γ] matches approx_fprime,
        including the γ block (acceptance criterion 3)."""
        ib, y, X, x_s = _scaled_interaction_problem(n=60, seed=9)
        p, q = ib.n_y_params, ib.n_x_params
        rng = np.random.default_rng(123)

        # Feasible, strictly monotone Θ columns + a non-zero γ so the scale
        # factor f = exp(0.5·x_s·γ) is genuinely exercised.
        Theta = np.outer(np.linspace(-1.0, 1.0, p), np.ones(q))
        Theta = Theta + 0.05 * rng.standard_normal((p, q))
        for j in range(q):
            Theta[:, j] = np.maximum.accumulate(Theta[:, j]) + 1e-3 * np.arange(p)
        gamma = np.array([0.4])
        theta = np.concatenate([Theta.ravel(), gamma])

        _, grad_ana = negative_log_likelihood(
            theta, ib, y, X=X, gradient=True, scaling=x_s
        )

        def nll(t: np.ndarray) -> float:
            return float(
                negative_log_likelihood(t, ib, y, X=X, gradient=False, scaling=x_s)
            )

        grad_fd = approx_fprime(theta, nll, 1e-6)
        assert_allclose(grad_ana, grad_fd, rtol=1e-5, atol=1e-6)


class TestZeroScalingReduction:
    def test_zero_scaling_columns_match_pure_interaction(self) -> None:
        """All-zero scaling columns give f = exp(0) = 1, so the scaled fit
        must reproduce the pure InteractionBasis Θ and log-likelihood
        (ADR 0003 Decision 6, secondary check 1)."""
        ib, y, X, _ = _scaled_interaction_problem(n=200, seed=42)
        cfg = OptimizerConfig(solver="auglag", random_state=0)

        pure = ConditionalTransformationModel(
            basis=ib, base_distribution="normal", optimizer_config=cfg
        )
        pure.fit(y, X)

        x_s_zero = np.zeros((y.size, 1))
        scaled = ConditionalTransformationModel(
            basis=ib,
            scaling=x_s_zero,
            base_distribution="normal",
            optimizer_config=cfg,
        )
        with pytest.warns(UserWarning, match="experimental"):
            scaled.fit(y, X)

        assert_allclose(
            scaled.result_.log_likelihood,
            pure.result_.log_likelihood,
            rtol=1e-7,
            atol=1e-9,
        )
        assert_allclose(scaled.Theta_, pure.Theta_, rtol=1e-6, atol=1e-8)
        # γ is unidentified by all-zero columns and must stay at its init.
        assert_allclose(scaled.gamma_, np.zeros(1), atol=1e-8)


class TestExperimentalWarning:
    def test_fit_emits_experimental_warning(self) -> None:
        """fit() on scaling + InteractionBasis warns (ADR 0003 Decision 7,
        mirroring R tram's stram 'highly experimental' warning)."""
        ib, y, X, x_s = _scaled_interaction_problem(n=80, seed=1)
        model = ConditionalTransformationModel(
            basis=ib,
            scaling=x_s,
            base_distribution="normal",
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        with pytest.warns(UserWarning, match="experimental"):
            model.fit(y, X)

    def test_pure_interaction_fit_does_not_warn(self) -> None:
        """The experimental warning fires only on the scaled path."""
        ib, y, X, _ = _scaled_interaction_problem(n=80, seed=1)
        model = ConditionalTransformationModel(
            basis=ib,
            base_distribution="normal",
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            model.fit(y, X)


def _fit_intercept_reduction_pair(
    base_distribution: str = "normal",
) -> tuple[MLT, ConditionalTransformationModel]:
    """Fit a scaled shift ``MLT`` (no covariates) and the equivalent scaled
    ``InteractionBasis`` with an ``InterceptBasis`` x-basis on the same data.

    The interacting baseline collapses to ``h_0(y)`` when ``q = 1``, so the
    two models describe the same distribution and every ``predict`` target
    of the interaction model must match the (independently implemented)
    scaled-shift predict path.
    """
    rng = np.random.default_rng(7)
    n, order, support = 250, 4, (0.0, 5.0)
    y = rng.uniform(0.1, 4.9, size=n)
    x_s = rng.normal(size=(n, 1))
    cfg = OptimizerConfig(solver="auglag", random_state=0)

    shift = MLT(
        order=order,
        support=support,
        scaling=x_s,
        base_distribution=base_distribution,
        optimizer_config=cfg,
    )
    shift.fit(y)

    ib = InteractionBasis(
        y_basis=BernsteinBasis(order=order, support=support),
        x_basis=InterceptBasis(support=(0.0, 1.0)),
    )
    inter = ConditionalTransformationModel(
        basis=ib,
        scaling=x_s,
        base_distribution=base_distribution,
        optimizer_config=cfg,
    )
    with pytest.warns(UserWarning, match="experimental"):
        inter.fit(y, X=np.full((n, 1), 0.5))
    return shift, inter


@pytest.fixture(scope="module")
def reduction_pair() -> tuple[MLT, ConditionalTransformationModel]:
    return _fit_intercept_reduction_pair()


# Prediction grid: response values spanning the support, paired row-wise with
# non-trivial scaling-design values so f = exp(0.5·x_s·γ) varies per row.
_Y_GRID = np.linspace(0.3, 4.7, 9)
_XS_GRID = np.linspace(-1.5, 1.5, 9)[:, None]


class TestScaledInteractionPredict:
    @pytest.mark.parametrize(
        "what",
        [
            "trafo",
            "distribution",
            "logdistribution",
            "survivor",
            "logsurvivor",
            "density",
            "logdensity",
            "hazard",
            "loghazard",
            "cumhazard",
            "logcumhazard",
            "odds",
            "logodds",
        ],
    )
    def test_matches_scaled_shift_reduction(
        self,
        reduction_pair: tuple[MLT, ConditionalTransformationModel],
        what: str,
    ) -> None:
        """With an InterceptBasis x-basis, predict(X_scale_new=...) on the
        scaled interaction model equals the scaled-shift prediction."""
        shift, inter = reduction_pair
        m = _Y_GRID.size
        expected = shift.predict(_Y_GRID, X_scale_new=_XS_GRID, what=what)
        got = inter.predict(
            _Y_GRID, X_new=np.full((m, 1), 0.5), X_scale_new=_XS_GRID, what=what
        )
        assert_allclose(got, expected, rtol=1e-4, atol=1e-6)

    def test_quantile_matches_scaled_shift_reduction(
        self, reduction_pair: tuple[MLT, ConditionalTransformationModel]
    ) -> None:
        """Quantile inversion divides the target by f_i, so the interaction
        quantile must equal the scaled-shift quantile under the reduction."""
        shift, inter = reduction_pair
        probs = np.linspace(0.1, 0.9, 9)
        expected = shift.predict(probs, X_scale_new=_XS_GRID, what="quantile")
        got = inter.predict(
            probs,
            X_new=np.full((probs.size, 1), 0.5),
            X_scale_new=_XS_GRID,
            what="quantile",
        )
        assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


_LINKS = ["normal", "logistic", "min_extreme_value"]


@pytest.fixture(scope="module", params=_LINKS)
def scaled_bs_bs_model(
    request: pytest.FixtureRequest,
) -> ConditionalTransformationModel:
    """Bernstein-y ⊗ Bernstein-x scaled fit, one per base distribution."""
    ib, y, X, x_s = _scaled_interaction_problem(n=200, seed=42)
    model = ConditionalTransformationModel(
        basis=ib,
        scaling=x_s,
        base_distribution=request.param,
        optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
    )
    with pytest.warns(UserWarning, match="experimental"):
        model.fit(y, X)
    return model


class TestScaledInteractionPredictInteracting:
    @pytest.mark.parametrize("x_val, xs_val", [(0.2, -1.2), (0.5, 0.0), (0.8, 1.5)])
    def test_quantile_distribution_round_trip(
        self,
        scaled_bs_bs_model: ConditionalTransformationModel,
        x_val: float,
        xs_val: float,
    ) -> None:
        """F(Q(p | x, x_s) | x, x_s) = p with a non-trivial X_scale_new."""
        probs = np.linspace(0.1, 0.9, 9)
        m = probs.size
        X_new = np.full((m, 1), x_val)
        X_scale_new = np.full((m, 1), xs_val)
        q = scaled_bs_bs_model.predict(
            probs, X_new=X_new, X_scale_new=X_scale_new, what="quantile"
        )
        p_back = scaled_bs_bs_model.predict(
            q, X_new=X_new, X_scale_new=X_scale_new, what="distribution"
        )
        assert_allclose(p_back, probs, atol=1e-6)

    @pytest.mark.parametrize("x_val, xs_val", [(0.2, -1.2), (0.8, 1.5)])
    def test_density_integrates_to_cdf_mass(
        self,
        scaled_bs_bs_model: ConditionalTransformationModel,
        x_val: float,
        xs_val: float,
    ) -> None:
        """∫_a^b f(y | x, x_s) dy = F(b | x, x_s) − F(a | x, x_s).

        The bounded Bernstein support truncates the base distribution, so the
        integral is compared to the CDF mass over the support rather than to
        1: heavy-tailed links leave a few percent of mass outside ``[a, b]``
        when f = exp(0.5·x_s·γ) < 1 flattens h.  Equality pins the density
        as the exact y-derivative of the scaled CDF (h' carries f too)."""
        a, b = scaled_bs_bs_model.basis.support
        grid = np.linspace(a, b, 4001)
        m = grid.size
        X_new = np.full((m, 1), x_val)
        X_scale_new = np.full((m, 1), xs_val)
        dens = scaled_bs_bs_model.predict(
            grid, X_new=X_new, X_scale_new=X_scale_new, what="density"
        )
        cdf = scaled_bs_bs_model.predict(
            grid, X_new=X_new, X_scale_new=X_scale_new, what="distribution"
        )
        mass = cdf[-1] - cdf[0]
        assert_allclose(np.trapezoid(dens, grid), mass, rtol=1e-5)

    def test_mismatched_x_scale_columns_raise(
        self, scaled_bs_bs_model: ConditionalTransformationModel
    ) -> None:
        y_new = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="X_scale_new has 2 columns"):
            scaled_bs_bs_model.predict(
                y_new, X_new=np.full((2, 1), 0.5), X_scale_new=np.zeros((2, 2))
            )

    def test_missing_x_scale_raises(
        self, scaled_bs_bs_model: ConditionalTransformationModel
    ) -> None:
        with pytest.raises(ValueError, match="X_scale_new must be provided"):
            scaled_bs_bs_model.predict(
                np.array([1.0]), X_new=np.full((1, 1), 0.5), what="quantile"
            )


REF_DIR = pathlib.Path(__file__).parent.parent / "reference"


class TestStramRParity:
    """R-parity ratchet for the scaled-interaction (stram) path — issue #103.

    Fixtures are generated by ``reference/generate_reference.R`` under the
    prefix ``stram_bs_bs_normal_*`` (mlt::ctm with response = Bernstein-y,
    interacting = Bernstein-x, scaling = linear x_s, Normal base).  γ is
    sign-aligned with mltpy, so no flip is applied (ADR 0003 Decision 6).
    Materialised in #125.  The MLE has one active monotonicity row (the middle
    Θ column is flat in its first two entries), so the R fit is polished with
    mlt's projected ``spg`` optimiser rather than an unconstrained BFGS step.
    """

    def test_gamma_and_loglik_match_r(self) -> None:
        model, (gamma_R, ll_R) = _fit_stram_fixture("gamma", "loglik")
        assert_allclose(model.gamma_, np.atleast_1d(gamma_R), rtol=1e-6, atol=1e-10)
        assert_allclose(
            model.result_.log_likelihood, float(ll_R), rtol=1e-6, atol=1e-10
        )

    def test_theta_matches_r(self) -> None:
        model, (theta_R,) = _fit_stram_fixture("theta")
        p, q = 3, 3
        # mlt orders vec(Θ) with the y-basis index fastest (column-major on
        # the (p, q) matrix); the trailing entry is γ.
        Theta_R = theta_R[: p * q].reshape(p, q, order="F")
        assert_allclose(model.Theta_, Theta_R, rtol=1e-6, atol=1e-10)
        assert_allclose(model.gamma_, theta_R[p * q :], rtol=1e-6, atol=1e-10)


def _fit_stram_fixture(
    *extra: str,
) -> tuple[ConditionalTransformationModel, list[np.ndarray]]:
    """Fit mltpy on the ``stram_bs_bs_normal_*`` R fixture (single x_s column).

    Skips the calling test unless the shared inputs and every
    ``stram_bs_bs_normal_<name>.txt`` named in ``extra`` exist; returns the
    fitted model together with the loaded ``extra`` arrays, in order.
    """
    prefix = "stram_bs_bs_normal_"
    names = ["y_train", "x_train", "x_s_train", "y_support", *extra]
    paths = [REF_DIR / f"{prefix}{name}.txt" for name in names]
    if not all(p.exists() for p in paths):
        pytest.skip(
            f"{prefix}* fixtures not generated — run "
            "Rscript reference/generate_reference.R"
        )
    y, x, x_s, support, *loaded = (np.loadtxt(p) for p in paths)

    # Mirror the R bases: Bernstein-y order 2 (p=3) ⊗ Bernstein-x order 2 (q=3).
    ib = InteractionBasis(
        y_basis=BernsteinBasis(order=2, support=tuple(support)),
        x_basis=BernsteinBasis(order=2, support=(0.0, 1.0)),
    )
    model = ConditionalTransformationModel(
        basis=ib,
        scaling=x_s.reshape(-1, 1),
        base_distribution="normal",
        optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
    )
    with pytest.warns(UserWarning, match="experimental"):
        model.fit(y, X=x.reshape(-1, 1))
    return model, loaded


# ---------------------------------------------------------------------------
# γ uncertainty on the scaled-interaction path — issue #105 (ADR 0003 Dec. 4)
# ---------------------------------------------------------------------------


def _feasible_scaled_theta(
    ib: InteractionBasis, gamma: np.ndarray, seed: int = 123
) -> np.ndarray:
    """``[vec_C(Θ) | γ]`` with strictly monotone Θ columns and a non-zero γ,
    so the scale factor ``f = exp(0.5·x_s·γ)`` is genuinely exercised."""
    p, q = ib.n_y_params, ib.n_x_params
    rng = np.random.default_rng(seed)
    Theta = np.outer(np.linspace(-1.0, 1.0, p), np.ones(q))
    Theta = Theta + 0.05 * rng.standard_normal((p, q))
    for j in range(q):
        Theta[:, j] = np.maximum.accumulate(Theta[:, j]) + 1e-3 * np.arange(p)
    return np.concatenate([Theta.ravel(), gamma])


def _central_fd(
    fun: Callable[[np.ndarray], np.ndarray | float], x: np.ndarray, step: float
) -> np.ndarray:
    """Central-difference Jacobian of ``fun`` at ``x`` (test reference only)."""
    cols = []
    for k in range(x.size):
        e = np.zeros_like(x)
        e[k] = step
        cols.append((np.asarray(fun(x + e)) - np.asarray(fun(x - e))) / (2 * step))
    return np.stack(cols, axis=-1)


@pytest.fixture(scope="module", params=[1, 2], ids=["q_s=1", "q_s=2"])
def scaled_interaction_fit(
    request: pytest.FixtureRequest,
) -> ConditionalTransformationModel:
    """A fitted scaled InteractionBasis model with ``q_s`` scaling columns."""
    q_s = request.param
    ib, y, X, _ = _scaled_interaction_problem(n=200, seed=42)
    x_s = np.random.default_rng(5).normal(size=(y.size, q_s))
    model = ConditionalTransformationModel(
        basis=ib,
        scaling=x_s,
        base_distribution="normal",
        optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
    )
    with pytest.warns(UserWarning, match="experimental"):
        model.fit(y, X)
    return model


class TestScaledInteractionVcov:
    def test_vcov_spans_theta_and_gamma(
        self, scaled_interaction_fit: ConditionalTransformationModel
    ) -> None:
        """vcov() is the finite, symmetric ``(p·q + q_s)²`` inverse observed
        information over ``[vec_C(Θ) | γ]`` (acceptance criterion 1)."""
        model = scaled_interaction_fit
        ib = model.basis
        assert isinstance(ib, InteractionBasis)
        q_s = model.gamma_.size
        m = ib.n_y_params * ib.n_x_params + q_s

        V = model.vcov()
        assert V.shape == (m, m)
        assert np.all(np.isfinite(V))
        assert_allclose(V, V.T, rtol=0, atol=1e-10)
        # The γ block is a proper covariance: positive definite.
        assert np.all(np.linalg.eigvalsh(V[-q_s:, -q_s:]) > 0)

    def test_gamma_wald_se_and_ci(
        self, scaled_interaction_fit: ConditionalTransformationModel
    ) -> None:
        """Wald SEs for γ are the trailing ``q_s`` diagonal entries of vcov();
        confint() and sandwich_se() report finite γ intervals/SEs from the
        same machinery (acceptance criterion 2)."""
        model = scaled_interaction_fit
        q_s = model.gamma_.size

        se = model.standard_errors()[-q_s:]
        assert np.all(np.isfinite(se)) and np.all(se > 0)
        assert_allclose(se, np.sqrt(np.diag(model.vcov())[-q_s:]), rtol=1e-12)

        ci = model.confint(level=0.95)[-q_s:]
        z = norm.ppf(0.975)
        assert_allclose(ci[:, 0], model.gamma_ - z * se, rtol=1e-10, atol=1e-12)
        assert_allclose(ci[:, 1], model.gamma_ + z * se, rtol=1e-10, atol=1e-12)

        sand = model.sandwich_se()[-q_s:]
        assert np.all(np.isfinite(sand)) and np.all(sand > 0)

    def test_gamma_wald_test_against_zero(
        self, scaled_interaction_fit: ConditionalTransformationModel
    ) -> None:
        """A single-row γ contrast reproduces the squared Wald z of the γ SE."""
        model = scaled_interaction_fit
        R = np.zeros((1, model.theta_.size))
        R[0, -1] = 1.0
        res = model.wald_test(R)
        se = model.standard_errors()[-1]
        assert_allclose(res.statistic, (model.gamma_[-1] / se) ** 2, rtol=1e-10)
        assert res.df == 1


class TestScaledInteractionDerivatives:
    @pytest.mark.parametrize("q_s", [1, 2])
    def test_hessian_matches_jacobian_of_analytic_gradient(self, q_s: int) -> None:
        """The finite-difference interaction Hessian threads ``scaling`` into
        its NLL closure: it matches the central-difference Jacobian of the
        *analytic* gradient, including the γ rows and the Θ–γ cross-block."""
        ib, y, X, _ = _scaled_interaction_problem(n=60, seed=9)
        x_s = np.random.default_rng(3).normal(size=(y.size, q_s))
        theta = _feasible_scaled_theta(ib, np.linspace(0.4, -0.3, q_s))

        H = hessian(theta, ib, y, X=X, scaling=x_s)

        def grad(t: np.ndarray) -> np.ndarray:
            _, g = negative_log_likelihood(t, ib, y, X=X, gradient=True, scaling=x_s)
            return g

        H_ref = _central_fd(grad, theta, 1e-6)
        assert H.shape == (theta.size, theta.size)
        assert_allclose(H, H_ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
    @pytest.mark.parametrize("q_s", [1, 2])
    def test_score_gamma_columns_match_per_observation_fd(
        self, q_s: int, weighted: bool
    ) -> None:
        """Each γ column of score_matrix is ``w_i · ∂ℓ_i/∂γ``, matched against
        a central difference of the single-observation log-likelihood
        (acceptance criterion 3)."""
        ib, y, X, _ = _scaled_interaction_problem(n=40, seed=11)
        x_s = np.random.default_rng(4).normal(size=(y.size, q_s))
        theta = _feasible_scaled_theta(ib, np.linspace(0.4, -0.3, q_s))
        w = np.random.default_rng(6).uniform(0.5, 2.0, y.size) if weighted else None

        S = score_matrix(theta, ib, y, X=X, weights=w, scaling=x_s)
        n_theta = ib.n_y_params * ib.n_x_params
        assert S.shape == (y.size, n_theta + q_s)

        S_ref = np.empty((y.size, q_s))
        for i in range(y.size):
            row = slice(i, i + 1)

            def ll_i(g: np.ndarray, row: slice = row) -> float:
                t = np.concatenate([theta[:n_theta], g])
                return log_likelihood(t, ib, y[row], X=X[row], scaling=x_s[row])

            S_ref[i] = _central_fd(ll_i, theta[n_theta:], 1e-6)
        if w is not None:
            S_ref = w[:, None] * S_ref
        assert_allclose(S[:, n_theta:], S_ref, rtol=1e-6, atol=1e-8)

    def test_fitted_estfun_gamma_columns_sum_to_zero(
        self, scaled_interaction_fit: ConditionalTransformationModel
    ) -> None:
        """At the MLE the γ score columns sum to zero (γ is unconstrained, so
        no KKT multiplier absorbs its gradient)."""
        model = scaled_interaction_fit
        q_s = model.gamma_.size
        assert_allclose(model.estfun()[:, -q_s:].sum(axis=0), 0.0, atol=1e-4)


class _ScaledInteractionTram(_TramModel):
    """Exact-data, normal-base tram over an InteractionBasis.

    Stand-in for the forthcoming ``BoxCox(interacting=..., scaling=...)``
    (#107) so the shared :meth:`_TramModel.summary` is exercised on the
    normal-base scaled-interaction path; the public ``Coxph`` route is
    covered in ``tests/test_tram_coxph_interacting_scaling.py``.
    """

    def __init__(self, basis: InteractionBasis, scaling: np.ndarray | None) -> None:
        ConditionalTransformationModel.__init__(
            self,
            basis=basis,
            base_distribution="normal",
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
            scaling=scaling,
        )
        self._order = basis.order
        self._support = basis.support


class TestScaledInteractionSummary:
    @pytest.mark.parametrize("q_s", [1, 2])
    def test_summary_tabulates_gamma_with_wald_se(self, q_s: int) -> None:
        """summary() renders a *Scaling coefficients* block with γ and its
        Wald SE; Θ stays untabulated (constrained nuisance), matching the
        shift path's treatment of theta_b."""
        ib, y, X, _ = _scaled_interaction_problem(n=200, seed=42)
        x_s = np.random.default_rng(5).normal(size=(y.size, q_s))
        model = _ScaledInteractionTram(ib, x_s)
        with pytest.warns(UserWarning, match="experimental"):
            model.fit(y, X)

        s = model.summary()
        assert "Scaling coefficients:" in s
        block = s.split("Scaling coefficients:")[1].strip("\n").splitlines()
        # Header + one row per γ; no Θ rows leak into the table.
        assert len(block) == 1 + q_s
        se = model.standard_errors()[-q_s:]
        for j, row in enumerate(block[1:]):
            fields = row.split()
            assert fields[0] == f"X{j + 1}"
            assert_allclose(float(fields[1]), model.gamma_[j], atol=5e-5)
            assert_allclose(float(fields[2]), se[j], atol=5e-5)

    def test_unscaled_interaction_summary_has_no_coefficient_table(self) -> None:
        """Without scaling there is no γ (and no flat β) block to tabulate."""
        ib, y, X, _ = _scaled_interaction_problem(n=120, seed=42)
        model = _ScaledInteractionTram(ib, None)
        model.fit(y, X)
        s = model.summary()
        assert "Coefficients:" not in s
        assert "Scaling coefficients:" not in s


class TestStramGammaVcovRParity:
    """R-parity ratchet for γ's covariance on the stram path — issue #105.

    ``stram_bs_bs_normal_gamma_vcov.txt`` is R's ``vcov(fit)`` restricted to
    the scaling coefficients (``reference/generate_reference.R``).  R's
    ``vcov.mlt`` is ``solve(Hessian)`` and ignores the fixture's one active
    monotonicity row, as does mltpy's default ``inv(H)``; mltpy's Hessian is
    finite-difference on this path (ADR 0003 Decision 4), hence rtol=1e-4 —
    the bound the shift-path vcov parity tests also use.  Materialised in #125.
    """

    def test_gamma_vcov_matches_r(self) -> None:
        model, (V_gamma_R,) = _fit_stram_fixture("gamma_vcov")
        # Single scaling column, so the γγ block is 1×1.
        assert_allclose(
            model.vcov()[-1:, -1:], V_gamma_R.reshape(1, 1), rtol=1e-4, atol=1e-8
        )
