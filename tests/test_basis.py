"""Tests for pymlt.basis — Bernstein basis, derivatives, and integration."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pymlt.basis import BernsteinBasis

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_basis(order: int = 4, support: tuple = (0.0, 1.0)) -> BernsteinBasis:
    return BernsteinBasis(order=order, support=support)


def linspace_in(basis: BernsteinBasis, n: int = 50) -> np.ndarray:
    a, b = basis.support
    return np.linspace(a, b, n)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestBernsteinBasisConstruction:
    def test_valid(self):
        b = BernsteinBasis(order=3, support=(0.0, 2.0))
        assert b.order == 3
        assert b.support == (0.0, 2.0)

    def test_negative_order_raises(self):
        with pytest.raises(ValueError, match="order"):
            BernsteinBasis(order=-1, support=(0.0, 1.0))

    def test_zero_order_ok(self):
        b = BernsteinBasis(order=0, support=(0.0, 1.0))
        assert b.order == 0

    def test_support_reversed_raises(self):
        with pytest.raises(ValueError, match="support"):
            BernsteinBasis(order=3, support=(1.0, 0.0))

    def test_support_equal_raises(self):
        with pytest.raises(ValueError, match="support"):
            BernsteinBasis(order=3, support=(1.0, 1.0))

    @pytest.mark.parametrize(
        "support",
        [
            (-np.inf, 1.0),
            (0.0, np.inf),
            (-np.inf, np.inf),
            (np.nan, 1.0),
            (0.0, np.nan),
        ],
    )
    def test_non_finite_support_raises(self, support):
        with pytest.raises(ValueError, match="finite"):
            BernsteinBasis(order=3, support=support)


# ---------------------------------------------------------------------------
# evaluate() — shape, partition of unity, boundary, non-negativity
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_shape(self):
        b = make_basis(order=5)
        y = np.linspace(0, 1, 20)
        M = b.evaluate(y)
        assert M.shape == (20, 6)

    def test_partition_of_unity(self):
        b = make_basis(order=6, support=(-2.0, 3.0))
        y = linspace_in(b, 100)
        M = b.evaluate(y)
        np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-12)

    def test_non_negativity(self):
        b = make_basis(order=4)
        y = linspace_in(b, 200)
        assert np.all(b.evaluate(y) >= -1e-15)

    def test_boundary_left(self):
        b = make_basis(order=4, support=(0.5, 2.5))
        M = b.evaluate(np.array([0.5]))
        expected = np.zeros(5)
        expected[0] = 1.0
        np.testing.assert_allclose(M[0], expected, atol=1e-12)

    def test_boundary_right(self):
        b = make_basis(order=4, support=(0.5, 2.5))
        M = b.evaluate(np.array([2.5]))
        expected = np.zeros(5)
        expected[-1] = 1.0
        np.testing.assert_allclose(M[0], expected, atol=1e-12)

    def test_order_zero_is_all_ones(self):
        b = BernsteinBasis(order=0, support=(0.0, 1.0))
        y = np.array([0.0, 0.3, 0.7, 1.0])
        M = b.evaluate(y)
        assert M.shape == (4, 1)
        np.testing.assert_allclose(M, 1.0, atol=1e-12)

    def test_scalar_input_works(self):
        b = make_basis(order=3)
        M = b.evaluate(np.array([0.5]))
        assert M.shape == (1, 4)

    @pytest.mark.parametrize("y", [np.array([-0.1, 0.5]), np.array([0.5, 1.1])])
    def test_out_of_support_raises(self, y):
        b = make_basis(order=3)
        with pytest.raises(ValueError, match="outside support"):
            b.evaluate(y)

    def test_2d_input_raises(self):
        b = make_basis(order=3)
        with pytest.raises(ValueError, match="1-D"):
            b.evaluate(np.array([[0.2, 0.4], [0.6, 0.8]]))

    @given(
        order=st.integers(1, 15),
        a=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
        width=st.floats(0.01, 200, allow_nan=False, allow_infinity=False),
        n=st.integers(1, 40),
    )
    @settings(max_examples=300)
    def test_pou_any_support_and_order(self, order, a, width, n):
        b = BernsteinBasis(order=order, support=(a, a + width))
        y = np.linspace(a, a + width, n)
        row_sums = b.evaluate(y).sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# derivative() — shape, analytical vs finite difference, edge cases
# ---------------------------------------------------------------------------


class TestDerivative:
    def test_shape_order1(self):
        b = make_basis(order=5)
        y = linspace_in(b, 30)
        D = b.derivative(y, order=1)
        assert D.shape == (30, 6)

    def test_shape_order2(self):
        b = make_basis(order=5)
        y = linspace_in(b, 30)
        D = b.derivative(y, order=2)
        assert D.shape == (30, 6)

    def test_invalid_order_raises(self):
        b = make_basis(order=3)
        with pytest.raises(ValueError, match="order"):
            b.derivative(np.array([0.5]), order=3)

    def test_order0_basis_has_zero_derivative(self):
        b = BernsteinBasis(order=0, support=(0.0, 1.0))
        D = b.derivative(np.array([0.3, 0.7]), order=1)
        assert D.shape == (2, 1)
        np.testing.assert_allclose(D, 0.0, atol=1e-15)

    def test_order1_basis_has_zero_second_derivative(self):
        b = BernsteinBasis(order=1, support=(0.0, 1.0))
        D2 = b.derivative(np.array([0.3, 0.7]), order=2)
        assert D2.shape == (2, 2)
        np.testing.assert_allclose(D2, 0.0, atol=1e-15)

    @pytest.mark.parametrize("y", [np.array([-0.1, 0.5]), np.array([0.5, 1.1])])
    def test_out_of_support_raises(self, y):
        b = make_basis(order=3)
        with pytest.raises(ValueError, match="outside support"):
            b.derivative(y, order=1)

    def test_2d_input_raises(self):
        b = make_basis(order=3)
        with pytest.raises(ValueError, match="1-D"):
            b.derivative(np.array([[0.2, 0.4], [0.6, 0.8]]))

    def test_derivative1_vs_finite_difference(self):
        """Analytical 1st derivative matches central finite difference."""
        b = make_basis(order=5, support=(0.0, 3.0))
        y = np.linspace(0.05, 2.95, 50)
        h = 1e-5
        fd = (b.evaluate(y + h) - b.evaluate(y - h)) / (2 * h)
        anal = b.derivative(y, order=1)
        np.testing.assert_allclose(anal, fd, atol=1e-5)

    def test_derivative2_vs_finite_difference(self):
        """Analytical 2nd derivative matches central finite difference."""
        b = make_basis(order=6, support=(0.0, 4.0))
        y = np.linspace(0.1, 3.9, 40)
        h = 1e-4
        fd2 = (b.evaluate(y + h) - 2 * b.evaluate(y) + b.evaluate(y - h)) / h**2
        anal = b.derivative(y, order=2)
        np.testing.assert_allclose(anal, fd2, atol=1e-4)

    def test_derivatives_sum_to_zero(self):
        """Sum of first-derivative rows = 0 (Partition of Unity differentiated)."""
        b = make_basis(order=7, support=(-1.0, 2.0))
        y = linspace_in(b, 60)
        D = b.derivative(y, order=1)
        np.testing.assert_allclose(D.sum(axis=1), 0.0, atol=1e-11)

    @given(
        order=st.integers(2, 12),
        a=st.floats(-50, 50, allow_nan=False, allow_infinity=False),
        width=st.floats(0.1, 100, allow_nan=False, allow_infinity=False),
        seed=st.integers(0, 2**31 - 1),
    )
    @settings(max_examples=200)
    def test_derivative_positive_for_ascending_theta(self, order, a, width, seed):
        """h'(y) = derivative(y) @ theta >= 0 when theta is non-decreasing."""
        rng = np.random.default_rng(seed)
        # Build non-decreasing theta via cumulative sum of non-negative increments
        theta = np.cumsum(rng.uniform(0.0, 2.0, size=order + 1)) - (order + 1)
        b = BernsteinBasis(order=order, support=(a, a + width))
        y = np.linspace(a, a + width, 30)
        h_prime = b.derivative(y, order=1) @ theta
        assert np.all(h_prime >= -1e-10), (
            f"Monotonicity violated: min h' = {h_prime.min():.3e}"
        )


# ---------------------------------------------------------------------------
# evaluate_with_derivative() — equivalence, shapes, edge cases
# ---------------------------------------------------------------------------


class TestEvaluateWithDerivative:
    @pytest.mark.parametrize("order", [1, 3, 5, 8])
    def test_equivalence(self, order):
        """evaluate_with_derivative returns same results as separate calls."""
        b = make_basis(order=order)
        y = linspace_in(b, 40)
        B_ref = b.evaluate(y)
        dB_ref = b.derivative(y, order=1)
        B, dB = b.evaluate_with_derivative(y)
        np.testing.assert_array_equal(B, B_ref)
        np.testing.assert_array_equal(dB, dB_ref)

    def test_shapes(self):
        b = make_basis(order=6)
        y = linspace_in(b, 25)
        B, dB = b.evaluate_with_derivative(y)
        assert B.shape == (25, 7)
        assert dB.shape == (25, 7)

    def test_order0_edge_case(self):
        b = BernsteinBasis(order=0, support=(0.0, 1.0))
        y = np.array([0.25, 0.5, 0.75])
        B, dB = b.evaluate_with_derivative(y)
        assert B.shape == (3, 1)
        assert dB.shape == (3, 1)
        np.testing.assert_allclose(dB, 0.0, atol=1e-15)

    @pytest.mark.parametrize("y", [np.array([-0.1, 0.5]), np.array([0.5, 1.1])])
    def test_out_of_support_raises(self, y):
        b = make_basis(order=3)
        with pytest.raises(ValueError, match="outside support"):
            b.evaluate_with_derivative(y)


# ---------------------------------------------------------------------------
# integrate() — shape, value at full domain, monotonicity
# ---------------------------------------------------------------------------


class TestIntegrate:
    def test_shape(self):
        b = make_basis(order=4)
        y = linspace_in(b, 20)
        integ = b.integrate(y)
        assert integ.shape == (20, 5)

    def test_zero_at_lower_bound(self):
        b = make_basis(order=4, support=(1.0, 3.0))
        integ = b.integrate(np.array([1.0]))
        np.testing.assert_allclose(integ, 0.0, atol=1e-12)

    def test_full_domain_integral(self):
        """∫_a^b B_{i,k}(y) dy = (b−a)/(k+1) for all i."""
        b = make_basis(order=5, support=(0.5, 2.5))
        a, bval = b.support
        integ = b.integrate(np.array([bval]))  # shape (1, 6)
        expected = (bval - a) / (b.order + 1)
        np.testing.assert_allclose(integ[0], expected, atol=1e-12)

    def test_integral_sum_equals_support_width(self):
        """Sum of column integrals over [a,b] = (b-a)."""
        b = make_basis(order=6, support=(-1.0, 3.0))
        a, bval = b.support
        integ = b.integrate(np.array([bval]))
        np.testing.assert_allclose(integ.sum(), bval - a, atol=1e-12)

    def test_integrate_monotone_in_y(self):
        """Running integral of constant 1 (sum of all B_i) is monotone in y."""
        b = make_basis(order=5, support=(0.0, 2.0))
        y = linspace_in(b, 50)
        running_total = b.integrate(y).sum(axis=1)  # = ∫_a^y 1 dy = y - a
        diffs = np.diff(running_total)
        assert np.all(diffs >= -1e-14)

    def test_integrate_matches_sum_of_basis(self):
        """∫_a^y sum_i B_{i,k}(s) ds = y − a (since partition of unity)."""
        b = make_basis(order=5, support=(0.0, 2.0))
        y = linspace_in(b, 40)
        a, _ = b.support
        integral_sum = b.integrate(y).sum(axis=1)
        np.testing.assert_allclose(integral_sum, y - a, atol=1e-11)

    @pytest.mark.parametrize("y", [np.array([-0.1, 0.5]), np.array([0.5, 1.1])])
    def test_out_of_support_raises(self, y):
        b = make_basis(order=3)
        with pytest.raises(ValueError, match="outside support"):
            b.integrate(y)

    def test_2d_input_raises(self):
        b = make_basis(order=3)
        with pytest.raises(ValueError, match="1-D"):
            b.integrate(np.array([[0.2, 0.4], [0.6, 0.8]]))

    @given(
        order=st.integers(1, 10),
        a=st.floats(-20, 20, allow_nan=False, allow_infinity=False),
        width=st.floats(0.05, 50, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=150)
    def test_integrate_monotone_for_any_basis(self, order, a, width):
        """Running integral (summed over columns) is non-decreasing."""
        b = BernsteinBasis(order=order, support=(a, a + width))
        y = np.linspace(a, a + width, 25)
        running = b.integrate(y).sum(axis=1)
        assert np.all(np.diff(running) >= -1e-12)


# ---------------------------------------------------------------------------
# Reference .npy comparison (skip if file absent)
# ---------------------------------------------------------------------------


def test_reference_npy(tmp_path):
    """Compare evaluate() against R basefun::Bernstein_basis reference values.

    Reference is produced by reference/generate_reference.R and stored as
    a plain-text 11x5 matrix in reference/bernstein_reference.txt (ascending
    column order, matching pymlt's convention).
    """
    import pathlib

    ref_path = (
        pathlib.Path(__file__).parent.parent / "reference" / "bernstein_reference.txt"
    )
    if not ref_path.exists():
        pytest.skip(
            "reference/bernstein_reference.txt not yet generated — "
            "run Rscript reference/generate_reference.R"
        )

    ref = np.loadtxt(ref_path)
    y = np.linspace(0.0, 1.0, ref.shape[0])
    b = BernsteinBasis(order=ref.shape[1] - 1, support=(0.0, 1.0))
    M = b.evaluate(y)
    np.testing.assert_allclose(M, ref, atol=1e-10)
