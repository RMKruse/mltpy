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

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import approx_fprime

from mltpy import (
    MLT,
    ConditionalTransformationModel,
    InteractionBasis,
    InterceptBasis,
    OptimizerConfig,
    negative_log_likelihood,
)
from mltpy.basis import BernsteinBasis


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


REF_DIR = pathlib.Path(__file__).parent.parent / "reference"


class TestStramRParity:
    """R-parity ratchet for the scaled-interaction (stram) path — issue #103.

    Fixtures are generated by ``reference/generate_reference.R`` under the
    prefix ``stram_bs_bs_normal_*`` (mlt::ctm with response = Bernstein-y,
    interacting = Bernstein-x, scaling = linear x_s, Normal base).  γ is
    sign-aligned with mltpy, so no flip is applied (ADR 0003 Decision 6).
    Skips until the fixtures have been materialised under R.
    """

    def test_gamma_and_loglik_match_r(self) -> None:
        prefix = "stram_bs_bs_normal_"
        required = [
            REF_DIR / f"{prefix}y_train.txt",
            REF_DIR / f"{prefix}x_train.txt",
            REF_DIR / f"{prefix}x_s_train.txt",
            REF_DIR / f"{prefix}y_support.txt",
            REF_DIR / f"{prefix}gamma.txt",
            REF_DIR / f"{prefix}loglik.txt",
        ]
        if not all(p.exists() for p in required):
            pytest.skip(
                f"{prefix}* fixtures not generated — run "
                "Rscript reference/generate_reference.R"
            )

        y = np.loadtxt(required[0])
        x = np.loadtxt(required[1])
        x_s = np.atleast_2d(np.loadtxt(required[2])).reshape(-1, 1)
        support = tuple(np.loadtxt(required[3]))
        gamma_R = np.atleast_1d(np.loadtxt(required[4]))
        ll_R = float(np.loadtxt(required[5]))

        # Mirror the R bases: Bernstein-y order 2 (p=3) ⊗ Bernstein-x order 2 (q=3).
        ib = InteractionBasis(
            y_basis=BernsteinBasis(order=2, support=support),
            x_basis=BernsteinBasis(order=2, support=(0.0, 1.0)),
        )
        model = ConditionalTransformationModel(
            basis=ib,
            scaling=x_s,
            base_distribution="normal",
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        with pytest.warns(UserWarning, match="experimental"):
            model.fit(y, X=x.reshape(-1, 1))

        assert_allclose(model.gamma_, gamma_R, rtol=1e-6, atol=1e-10)
        assert_allclose(model.result_.log_likelihood, ll_R, rtol=1e-6, atol=1e-10)
