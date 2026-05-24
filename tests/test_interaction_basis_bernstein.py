"""Tests for InteractionBasis with Bernstein y-basis and Bernstein x-basis — issue #64.

Scope of issue #64 (per the issue body):
- Design matrix + likelihood path only.
- Monotonicity enforcement and R parity ship in issue #65.

Acceptance criteria covered here:
1. ``InteractionBasis(Bernstein, Bernstein).evaluate(...)`` matches the
   hand-computed row-Kronecker for small cases.
2. Analytical derivative ``∂/∂y`` matches a finite-difference reference to
   ``rtol=1e-6``.
3. Negative log-likelihood and its gradient match finite-difference
   references for a small synthetic problem.
4. An unconstrained fit (or fit with diagnostic-only monotonicity check)
   converges and is reproducible with a fixed ``random_state``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymlt import (
    ConditionalTransformationModel,
    InteractionBasis,
    OptimizerConfig,
)
from pymlt.basis import BernsteinBasis
from pymlt.likelihood import negative_log_likelihood

# ---------------------------------------------------------------------------
# 1. Hand-computed row-Kronecker for the Bernstein × Bernstein design.
# ---------------------------------------------------------------------------


class TestBernsteinBernsteinEvaluate:
    def test_shape(self) -> None:
        y_basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        x_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
        y = np.array([0.2, 0.5, 0.9])
        X = np.array([0.1, 0.5, 0.7])
        design = ib.evaluate(y, X)
        # p = 4, q = 3, n = 3 ⇒ shape (3, 12)
        assert design.shape == (3, 12)

    def test_row_kron(self) -> None:
        """design[i] = kron(a(y_i), b(x_i)) for every row."""
        y_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        x_basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)

        y = np.array([0.15, 0.55, 0.85])
        X = np.array([0.10, 0.50, 0.90])
        design = ib.evaluate(y, X)

        A = y_basis.evaluate(y)  # (3, 3)
        B = x_basis.evaluate(X)  # (3, 4)
        for i in range(3):
            expected_row = np.kron(A[i], B[i])
            assert_allclose(design[i], expected_row, rtol=1e-14)

    def test_partition_of_unity_in_x(self) -> None:
        """Summing across x-blocks gives back a(y) — Bernstein-x is a PoU."""
        y_basis = BernsteinBasis(order=3, support=(0.0, 5.0))
        x_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
        y = np.array([1.0, 2.5, 4.0])
        X = np.array([0.2, 0.4, 0.8])
        design = ib.evaluate(y, X)  # (3, p*q)
        p, q = 4, 3
        # row i reshaped to (p, q) — each row of the (p, q) matrix sums to
        # a(y_i)[k] · sum_j(b(x_i)[j]) == a(y_i)[k] · 1.
        rows = design.reshape(3, p, q)
        a_recovered = rows.sum(axis=2)
        a_expected = y_basis.evaluate(y)
        assert_allclose(a_recovered, a_expected, rtol=1e-12)

    def test_integrate_matches_row_kron(self) -> None:
        y_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        x_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
        y = np.array([0.3, 0.7])
        X = np.array([0.25, 0.75])
        integral = ib.integrate(y, X)

        iA = y_basis.integrate(y)  # (2, 3)
        B = x_basis.evaluate(X)  # (2, 3)
        for i in range(2):
            assert_allclose(integral[i], np.kron(iA[i], B[i]), rtol=1e-14)


# ---------------------------------------------------------------------------
# 2. Analytical derivative matches finite-difference reference.
# ---------------------------------------------------------------------------


class TestBernsteinBernsteinDerivative:
    def test_derivative_row_kron(self) -> None:
        y_basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        x_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
        y = np.array([0.2, 0.6])
        X = np.array([0.3, 0.7])
        d_design = ib.derivative(y, X)

        dA = y_basis.derivative(y, order=1)
        B = x_basis.evaluate(X)
        for i in range(2):
            assert_allclose(d_design[i], np.kron(dA[i], B[i]), rtol=1e-14)

    def test_derivative_matches_finite_difference(self) -> None:
        """Central differences on evaluate(y, X) match analytical derivative."""
        y_basis = BernsteinBasis(order=4, support=(0.0, 1.0))
        x_basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)

        rng = np.random.default_rng(11)
        y = rng.uniform(0.1, 0.9, size=6)
        X = rng.uniform(0.05, 0.95, size=6)

        eps = 1e-6
        plus = ib.evaluate(y + eps, X)
        minus = ib.evaluate(y - eps, X)
        fd = (plus - minus) / (2.0 * eps)

        ana = ib.derivative(y, X)
        assert_allclose(ana, fd, rtol=1e-6, atol=1e-8)

    def test_evaluate_with_derivative_consistent(self) -> None:
        y_basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        x_basis = BernsteinBasis(order=4, support=(0.0, 1.0))
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)

        rng = np.random.default_rng(2)
        y = rng.uniform(0.05, 0.95, size=5)
        X = rng.uniform(0.05, 0.95, size=5)
        design, d_design = ib.evaluate_with_derivative(y, X)
        assert_allclose(design, ib.evaluate(y, X))
        assert_allclose(d_design, ib.derivative(y, X))


# ---------------------------------------------------------------------------
# 3. Likelihood path: NLL and its gradient match finite differences.
# ---------------------------------------------------------------------------


def _make_problem(
    n: int = 60, seed: int = 5
) -> tuple[InteractionBasis, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_basis = BernsteinBasis(order=3, support=(0.0, 5.0))
    x_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
    ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
    y = rng.uniform(0.1, 4.9, size=n)
    X = rng.uniform(0.05, 0.95, size=n)
    return ib, y, X


class TestBernsteinBernsteinLikelihood:
    def test_nll_value_finite(self) -> None:
        ib, y, X = _make_problem()
        p, q = ib.n_y_params, ib.n_x_params
        # Monotone, feasible θ: each column = linspace(-2, 2, p)
        Theta = np.outer(np.linspace(-2.0, 2.0, p), np.ones(q))
        theta = Theta.ravel()
        nll = negative_log_likelihood(theta, ib, y, X=X, gradient=False)
        assert np.isfinite(nll)

    def test_grad_matches_finite_difference(self) -> None:
        """Analytical ∂NLL/∂θ matches central differences to rtol=1e-5."""
        ib, y, X = _make_problem(n=40, seed=9)
        p, q = ib.n_y_params, ib.n_x_params
        rng = np.random.default_rng(123)
        # Start from feasible θ then perturb mildly
        Theta = np.outer(np.linspace(-1.0, 1.0, p), np.ones(q))
        Theta = Theta + 0.05 * rng.standard_normal(Theta.shape)
        # Re-monotonise columns to stay strictly feasible (h' > 0 needed for NLL).
        for j in range(q):
            Theta[:, j] = np.maximum.accumulate(Theta[:, j])
            # Ensure strict monotonicity by adding a tiny ramp
            Theta[:, j] = Theta[:, j] + 1e-3 * np.arange(p)
        theta = Theta.ravel()

        nll_ana, grad_ana = negative_log_likelihood(theta, ib, y, X=X, gradient=True)

        eps = 1e-6
        grad_fd = np.zeros_like(theta)
        for k in range(theta.size):
            tp = theta.copy()
            tp[k] += eps
            tm = theta.copy()
            tm[k] -= eps
            f_plus = negative_log_likelihood(tp, ib, y, X=X, gradient=False)
            f_minus = negative_log_likelihood(tm, ib, y, X=X, gradient=False)
            grad_fd[k] = (f_plus - f_minus) / (2.0 * eps)

        assert np.isfinite(nll_ana)
        assert_allclose(grad_ana, grad_fd, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. End-to-end fit: convergence + reproducibility with fixed random_state.
# ---------------------------------------------------------------------------


class TestBernsteinBernsteinFit:
    def test_fit_runs_and_returns_theta(self) -> None:
        ib, y, X = _make_problem(n=200, seed=42)
        model = ConditionalTransformationModel(
            basis=ib,
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        model.fit(y, X)
        assert model.is_fitted_
        assert model.theta_ is not None
        assert model.Theta_ is not None
        p, q = ib.n_y_params, ib.n_x_params
        assert model.Theta_.shape == (p, q)
        assert model.theta_.shape == (p * q,)

    def test_fit_reproducible(self) -> None:
        """Same random_state ⇒ identical θ."""
        ib1, y, X = _make_problem(n=150, seed=42)
        ib2, _, _ = _make_problem(n=150, seed=42)

        cfg = OptimizerConfig(solver="auglag", random_state=0)
        m1 = ConditionalTransformationModel(basis=ib1, optimizer_config=cfg)
        m1.fit(y, X)
        m2 = ConditionalTransformationModel(basis=ib2, optimizer_config=cfg)
        m2.fit(y, X)

        assert m1.theta_ is not None
        assert m2.theta_ is not None
        assert_allclose(m1.theta_, m2.theta_, rtol=0, atol=0)

    def test_fit_predicts_valid_cdf_and_pdf(self) -> None:
        ib, y, X = _make_problem(n=200, seed=42)
        model = ConditionalTransformationModel(
            basis=ib,
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        model.fit(y, X)

        cdf = model.predict(y, X, what="distribution")
        pdf = model.predict(y, X, what="density")
        assert cdf.shape == (y.size,)
        assert pdf.shape == (y.size,)
        assert np.all((cdf >= 0.0) & (cdf <= 1.0))
        assert np.all(pdf > 0.0)

    def test_fit_columns_monotone(self) -> None:
        """Each column of Θ must satisfy D @ Θ[:, j] ≥ 0 after fit.

        This is the diagnostic from the issue body — the closed-form
        Kronecker constraint enforces it already (Bernstein-x is a PoU),
        but we assert it explicitly here.
        """
        ib, y, X = _make_problem(n=200, seed=42)
        model = ConditionalTransformationModel(
            basis=ib,
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        model.fit(y, X)
        Theta = model.Theta_
        assert Theta is not None
        p = ib.n_y_params
        D = np.diff(np.eye(p), axis=0)  # (p-1, p)
        diffs = D @ Theta  # (p-1, q)
        assert np.all(diffs >= -1e-8), (
            f"Column-wise monotonicity violated; min diff = {diffs.min()}"
        )


# ---------------------------------------------------------------------------
# 5. Public API export.
# ---------------------------------------------------------------------------


def test_interaction_basis_bernstein_bernstein_accepted() -> None:
    """No ValueError when both bases are BernsteinBasis."""
    ib = InteractionBasis(
        y_basis=BernsteinBasis(order=3, support=(0.0, 1.0)),
        x_basis=BernsteinBasis(order=2, support=(0.0, 1.0)),
    )
    assert ib.n_params == 4 * 3
    assert ib.n_y_params == 4
    assert ib.n_x_params == 3


def test_unsupported_x_basis_still_rejected() -> None:
    """Defensive: Polynomial / Legendre x-bases must still raise."""
    from pymlt.basis import LegendreBasis, PolynomialBasis

    with pytest.raises(ValueError, match="non-negative and a partition"):
        InteractionBasis(
            y_basis=BernsteinBasis(order=2, support=(0.0, 1.0)),
            x_basis=PolynomialBasis(order=2, support=(0.0, 1.0)),
        )
    with pytest.raises(ValueError, match="non-negative and a partition"):
        InteractionBasis(
            y_basis=BernsteinBasis(order=2, support=(0.0, 1.0)),
            x_basis=LegendreBasis(order=2, support=(0.0, 1.0)),
        )
