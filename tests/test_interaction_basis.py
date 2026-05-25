"""Tests for InteractionBasis (Bernstein-y ⊠ one-hot-x) — issue #63.

RED phase: all tests should fail until the implementation is complete.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mltpy import (
    MLT,
    ConditionalTransformationModel,
    InteractionBasis,
    OptimizerConfig,
)
from mltpy.basis import BernsteinBasis, OneHotBasis

# ---------------------------------------------------------------------------
# OneHotBasis unit tests
# ---------------------------------------------------------------------------


class TestOneHotBasis:
    def test_evaluate_basic(self) -> None:
        basis = OneHotBasis(K=3)
        x = np.array([0, 1, 2], dtype=float)
        B = basis.evaluate(x)
        assert B.shape == (3, 3)
        expected = np.eye(3)
        assert_allclose(B, expected)

    def test_evaluate_repeated(self) -> None:
        basis = OneHotBasis(K=2)
        x = np.array([0, 0, 1, 0, 1], dtype=float)
        B = basis.evaluate(x)
        assert B.shape == (5, 2)
        assert_allclose(B[:, 0], [1, 1, 0, 1, 0])
        assert_allclose(B[:, 1], [0, 0, 1, 0, 1])

    def test_partition_of_unity(self) -> None:
        basis = OneHotBasis(K=4)
        rng = np.random.default_rng(42)
        x = rng.integers(0, 4, size=20).astype(float)
        B = basis.evaluate(x)
        assert_allclose(B.sum(axis=1), np.ones(20))

    def test_nonnegative(self) -> None:
        basis = OneHotBasis(K=3)
        x = np.array([0, 1, 2, 0, 1], dtype=float)
        B = basis.evaluate(x)
        assert np.all(B >= 0)

    def test_order_property(self) -> None:
        for K in (2, 3, 5):
            assert OneHotBasis(K=K).order == K - 1

    def test_n_params(self) -> None:
        assert OneHotBasis(K=3).order + 1 == 3

    def test_invalid_K(self) -> None:
        with pytest.raises(ValueError, match="K must be >= 2"):
            OneHotBasis(K=1)

    def test_invalid_label_out_of_range(self) -> None:
        basis = OneHotBasis(K=3)
        with pytest.raises(ValueError):
            basis.evaluate(np.array([0, 1, 3], dtype=float))  # 3 is out of range

    def test_invalid_negative_label(self) -> None:
        basis = OneHotBasis(K=3)
        with pytest.raises(ValueError):
            basis.evaluate(np.array([-1, 0, 1], dtype=float))

    def test_empty_input(self) -> None:
        basis = OneHotBasis(K=3)
        B = basis.evaluate(np.array([], dtype=float))
        assert B.shape == (0, 3)

    def test_non_integer_label(self) -> None:
        basis = OneHotBasis(K=3)
        with pytest.raises(ValueError):
            basis.evaluate(np.array([0.5, 1.0], dtype=float))


# ---------------------------------------------------------------------------
# InteractionBasis unit tests (evaluate, derivative, evaluate_with_derivative)
# ---------------------------------------------------------------------------


class TestInteractionBasisEvaluate:
    """Hand-computed expected values for small n, K, p cases."""

    def test_shape_bernstein_onehot(self) -> None:
        y_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        x_basis = OneHotBasis(K=3)
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
        y = np.array([0.25, 0.5, 0.75])
        X = np.array([0, 1, 2], dtype=float)
        design = ib.evaluate(y, X)
        assert design.shape == (3, 3 * 3)  # n=3, p*q=3*3=9

    def test_evaluate_is_row_kron(self) -> None:
        """design[i] = kron(a(y_i), b(x_i)) for each row i."""
        y_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        x_basis = OneHotBasis(K=2)
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
        y = np.array([0.3, 0.7])
        X = np.array([0, 1], dtype=float)
        design = ib.evaluate(y, X)

        A = y_basis.evaluate(y)  # (2, 3)
        B = x_basis.evaluate(X)  # (2, 2)
        for i in range(2):
            expected_row = np.kron(A[i], B[i])
            assert_allclose(design[i], expected_row, rtol=1e-14)

    def test_derivative_is_row_kron_of_dA_and_B(self) -> None:
        """d_design[i] = kron(da(y_i)/dy, b(x_i)) for each row."""
        y_basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        x_basis = OneHotBasis(K=2)
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
        y = np.array([0.2, 0.8])
        X = np.array([1, 0], dtype=float)
        d_design = ib.derivative(y, X)

        dA = y_basis.derivative(y, order=1)  # (2, 4)
        B = x_basis.evaluate(X)  # (2, 2)
        for i in range(2):
            expected_row = np.kron(dA[i], B[i])
            assert_allclose(d_design[i], expected_row, rtol=1e-14)

    def test_evaluate_with_derivative_consistent(self) -> None:
        y_basis = BernsteinBasis(order=3, support=(0.0, 1.0))
        x_basis = OneHotBasis(K=3)
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
        y = np.linspace(0.1, 0.9, 5)
        X = np.array([0, 1, 2, 0, 1], dtype=float)
        design, d_design = ib.evaluate_with_derivative(y, X)
        assert_allclose(design, ib.evaluate(y, X))
        assert_allclose(d_design, ib.derivative(y, X))

    def test_stratum_k_has_zero_other_columns(self) -> None:
        """For obs in stratum k, non-k blocks of the design row are zero."""
        y_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        x_basis = OneHotBasis(K=3)
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
        p, q = 3, 3  # y params, x params
        y = np.array([0.5])
        X = np.array([1], dtype=float)  # stratum 1
        design = ib.evaluate(y, X)  # shape (1, 9)
        # Stratum 1 → columns [1*p : 2*p] are non-zero; others are zero
        row = design[0].reshape(p, q)
        assert_allclose(row[:, 0], np.zeros(p))  # stratum 0 block
        assert_allclose(row[:, 2], np.zeros(p))  # stratum 2 block
        assert not np.all(row[:, 1] == 0.0)  # stratum 1 block non-zero

    def test_integrate_is_row_kron_of_integral_and_B(self) -> None:
        """integrate(y, X)[i] = kron(integral_a(y_i), b(x_i))."""
        y_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
        x_basis = OneHotBasis(K=2)
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
        y = np.array([0.3, 0.7])
        X = np.array([0, 1], dtype=float)
        integral = ib.integrate(y, X)

        iA = y_basis.integrate(y)  # (2, 3)
        B = x_basis.evaluate(X)  # (2, 2)
        for i in range(2):
            expected_row = np.kron(iA[i], B[i])
            assert_allclose(integral[i], expected_row, rtol=1e-14)

    def test_n_params(self) -> None:
        ib = InteractionBasis(
            y_basis=BernsteinBasis(order=3, support=(0.0, 1.0)),
            x_basis=OneHotBasis(K=4),
        )
        assert ib.n_params == 4 * 4  # (3+1) * 4
        assert ib.n_y_params == 4
        assert ib.n_x_params == 4


# ---------------------------------------------------------------------------
# Constraints for InteractionBasis
# ---------------------------------------------------------------------------


class TestInteractionBasisConstraints:
    def test_constraint_matrix_shape(self) -> None:
        from mltpy.constraints import build_constraint_matrices_interaction

        ib = InteractionBasis(
            y_basis=BernsteinBasis(order=3, support=(0.0, 1.0)),
            x_basis=OneHotBasis(K=2),
        )
        cm = build_constraint_matrices_interaction(ib)
        p, q = 4, 2
        # (p-1)*q inequality rows, (p*q) columns
        assert cm.A_ineq.shape == ((p - 1) * q, p * q)
        assert cm.b_ineq.shape == ((p - 1) * q,)
        assert_allclose(cm.b_ineq, 0.0)

    def test_constraint_feasibility(self) -> None:
        """Non-decreasing columns of Theta satisfy (D⊗I_q)@vec(Theta) >= 0."""
        from mltpy.constraints import build_constraint_matrices_interaction

        ib = InteractionBasis(
            y_basis=BernsteinBasis(order=2, support=(0.0, 1.0)),
            x_basis=OneHotBasis(K=3),
        )
        cm = build_constraint_matrices_interaction(ib)
        p, q = 3, 3
        # Theta = linspace in each column — should be feasible
        Theta = np.outer(np.linspace(0, 1, p), np.ones(q))
        theta = Theta.ravel()
        assert np.all(cm.A_ineq @ theta >= -1e-14)

    def test_constraint_infeasible(self) -> None:
        from mltpy.constraints import build_constraint_matrices_interaction

        ib = InteractionBasis(
            y_basis=BernsteinBasis(order=2, support=(0.0, 1.0)),
            x_basis=OneHotBasis(K=2),
        )
        cm = build_constraint_matrices_interaction(ib)
        p, q = 3, 2
        # Theta has a decreasing column — infeasible
        Theta = np.zeros((p, q))
        Theta[:, 0] = [1.0, 0.5, 0.0]  # decreasing
        theta = Theta.ravel()
        assert np.any(cm.A_ineq @ theta < 0)

    def test_unsupported_x_basis_raises(self) -> None:
        from mltpy.basis import PolynomialBasis

        with pytest.raises(ValueError, match="non-negative and a partition"):
            InteractionBasis(
                y_basis=BernsteinBasis(order=2, support=(0.0, 1.0)),
                x_basis=PolynomialBasis(order=2, support=(0.0, 1.0)),
            )


# ---------------------------------------------------------------------------
# End-to-end stratified fit tests
# ---------------------------------------------------------------------------


def _make_stratified_data(
    n_per_stratum: int = 100,
    K: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, BernsteinBasis]:
    """Generate y, X for a K-stratum normal transformation model."""
    rng = np.random.default_rng(seed)
    support = (0.0, 10.0)
    basis = BernsteinBasis(order=3, support=support)

    y_list, x_list = [], []
    for k in range(K):
        y_k = rng.uniform(support[0] + 0.1, support[1] - 0.1, size=n_per_stratum)
        y_list.append(y_k)
        x_list.append(np.full(n_per_stratum, k, dtype=float))

    y = np.concatenate(y_list)
    X = np.concatenate(x_list)
    return y, X, basis


class TestStratifiedFitPropertyTest:
    """Property test: stratified model == K independent MLT fits."""

    @pytest.mark.parametrize("K", [2, 3])
    def test_stratified_matches_independent_fits(self, K: int) -> None:
        """Fit one InteractionBasis model and K independent MLTs on same data.

        Coefficients for each stratum must match to rtol=1e-4.
        """
        rng = np.random.default_rng(7)
        n = 80
        support = (0.0, 5.0)
        y_basis = BernsteinBasis(order=3, support=support)

        # Draw y uniformly; assign strata evenly
        y = rng.uniform(support[0] + 0.05, support[1] - 0.05, size=n * K)
        X_labels = np.tile(np.arange(K), n).astype(float)

        # Stratified model
        x_basis = OneHotBasis(K=K)
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
        config = OptimizerConfig(solver="auglag", random_state=0)
        strat_model = ConditionalTransformationModel(
            basis=ib,
            optimizer_config=config,
        )
        strat_model.fit(y, X_labels)

        assert strat_model.Theta_ is not None
        p, q = y_basis.order + 1, K
        assert strat_model.Theta_.shape == (p, q)

        # K independent MLT models
        for k in range(K):
            mask = X_labels == k
            y_k = y[mask]
            indep = MLT(
                order=y_basis.order,
                support=y_basis.support,
                optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
            )
            indep.fit(y_k)
            theta_strat_k = strat_model.Theta_[:, k]
            assert indep.theta_ is not None
            theta_indep_k = indep.theta_[:p]
            assert_allclose(theta_strat_k, theta_indep_k, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# predict(type="cdf") / predict(type="pdf")
# ---------------------------------------------------------------------------


class TestStratifiedPredict:
    def test_predict_cdf_shape(self) -> None:
        n, K = 60, 2
        support = (0.0, 5.0)
        rng = np.random.default_rng(13)
        y = rng.uniform(support[0] + 0.1, support[1] - 0.1, size=n)
        X = (np.arange(n) % K).astype(float)

        y_basis = BernsteinBasis(order=3, support=support)
        x_basis = OneHotBasis(K=K)
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
        model = ConditionalTransformationModel(
            basis=ib,
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        model.fit(y, X)

        cdf = model.predict(y, X, what="distribution")
        assert cdf.shape == (n,)
        assert np.all(cdf >= 0.0)
        assert np.all(cdf <= 1.0)

    def test_predict_pdf_positive(self) -> None:
        n, K = 60, 2
        support = (0.0, 5.0)
        rng = np.random.default_rng(17)
        y = rng.uniform(support[0] + 0.1, support[1] - 0.1, size=n)
        X = (np.arange(n) % K).astype(float)

        y_basis = BernsteinBasis(order=3, support=support)
        x_basis = OneHotBasis(K=K)
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
        model = ConditionalTransformationModel(
            basis=ib,
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        model.fit(y, X)

        pdf = model.predict(y, X, what="density")
        assert pdf.shape == (n,)
        assert np.all(pdf > 0.0)

    def test_predict_held_out(self) -> None:
        """Predictions on held-out data with different X labels."""
        n_train, K = 80, 3
        support = (0.0, 5.0)
        rng = np.random.default_rng(99)
        y_train = rng.uniform(support[0] + 0.1, support[1] - 0.1, size=n_train)
        X_train = (np.arange(n_train) % K).astype(float)

        y_basis = BernsteinBasis(order=3, support=support)
        x_basis = OneHotBasis(K=K)
        ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
        model = ConditionalTransformationModel(
            basis=ib,
            optimizer_config=OptimizerConfig(solver="auglag", random_state=0),
        )
        model.fit(y_train, X_train)

        y_test = rng.uniform(support[0] + 0.1, support[1] - 0.1, size=10)
        X_test = (np.arange(10) % K).astype(float)
        cdf = model.predict(y_test, X_test, what="distribution")
        assert cdf.shape == (10,)
        assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0)


# ---------------------------------------------------------------------------
# Public API export
# ---------------------------------------------------------------------------


def test_one_hot_basis_in_public_api() -> None:
    import mltpy

    assert hasattr(mltpy, "OneHotBasis")
    from mltpy import OneHotBasis as OHB

    b = OHB(K=3)
    assert b.order == 2
