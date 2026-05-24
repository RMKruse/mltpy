"""Tests for pymlt.basis — Bernstein basis, derivatives, and integration."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pymlt.basis import (
    BernsteinBasis,
    InterceptBasis,
    LegendreBasis,
    LogBasis,
    PolynomialBasis,
)

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
# Assembled (B, dB) content-keyed cache (#95)
# ---------------------------------------------------------------------------


class TestAssembledCache:
    def test_repeated_equal_content_y_returns_same_objects(self):
        """evaluate_with_derivative memoizes the assembled (B, dB) on y content:
        a fresh-but-equal array (the censored ``cd.exact[mask]`` slice pattern)
        hits the cache and returns the *identical* B and dB objects."""
        b = make_basis(order=4)
        y1 = np.linspace(0.1, 0.9, 20)
        y2 = y1.copy()  # distinct object, identical content
        assert y1 is not y2
        B1, dB1 = b.evaluate_with_derivative(y1)
        B2, dB2 = b.evaluate_with_derivative(y2)
        assert B2 is B1
        assert dB2 is dB1

    def test_returned_arrays_are_read_only(self):
        """Cached (B, dB) are read-only so an errant in-place write — which
        would corrupt every later cache consumer — fails loudly."""
        b = make_basis(order=5)
        y = np.linspace(0.0, 1.0, 15)
        B, dB = b.evaluate_with_derivative(y)
        assert not B.flags.writeable
        assert not dB.flags.writeable
        with pytest.raises(ValueError):
            B[0, 0] = 1.0
        with pytest.raises(ValueError):
            dB[0, 0] = 1.0

    def test_order0_derivative_is_read_only(self):
        """The k=0 branch builds dB from np.zeros; it must be read-only too."""
        b = BernsteinBasis(order=0, support=(0.0, 1.0))
        _, dB = b.evaluate_with_derivative(np.array([0.2, 0.5, 0.8]))
        assert not dB.flags.writeable

    def test_evaluate_warms_shared_assembled_cache(self):
        """evaluate participates in the same assembled cache: calling it warms
        the (B, dB) entry so a subsequent evaluate_with_derivative is a full
        hit (no dB rebuild)."""
        from pymlt.basis import _bernstein_assembled_cache

        b = make_basis(order=4)
        y = np.linspace(0.1, 0.9, 18)
        key = (b.order, b.support, np.ascontiguousarray(y, dtype=float).tobytes())
        _bernstein_assembled_cache.pop(key, None)
        B_eval = b.evaluate(y)
        assert key in _bernstein_assembled_cache
        # And the warmed entry is what evaluate_with_derivative returns.
        B_pair, _ = b.evaluate_with_derivative(y.copy())
        assert B_pair is B_eval

    def test_evaluate_returns_read_only(self):
        b = make_basis(order=4)
        B = b.evaluate(np.linspace(0.0, 1.0, 10))
        assert not B.flags.writeable

    def test_cache_is_bounded(self):
        """Distinct y across many fits cannot grow the cache without limit."""
        from pymlt.basis import _BERNSTEIN_ASSEMBLED_CACHE_MAXSIZE as maxsize
        from pymlt.basis import _bernstein_assembled_cache as cache

        cache.clear()
        b = make_basis(order=3)
        for i in range(maxsize + 5):
            # Distinct length → distinct content, all within [0, 1].
            b.evaluate_with_derivative(np.linspace(0.0, 1.0, 10 + i))
        assert len(cache) == maxsize

    def test_lru_recency_protects_accessed_entry(self):
        """A cache hit refreshes recency (move_to_end), so a recently-touched
        old entry survives eviction while the next-oldest is dropped."""
        from pymlt.basis import _BERNSTEIN_ASSEMBLED_CACHE_MAXSIZE as maxsize
        from pymlt.basis import _bernstein_assembled_cache as cache

        cache.clear()
        b = make_basis(order=3)

        def key_for(y: np.ndarray) -> tuple:
            return (b.order, b.support, np.ascontiguousarray(y, dtype=float).tobytes())

        ys = [np.linspace(0.0, 1.0, 10 + i) for i in range(maxsize)]
        for y in ys:
            b.evaluate_with_derivative(y)
        # Cache is full.  Touch the oldest entry → moves it to most-recent.
        b.evaluate_with_derivative(ys[0].copy())
        # Insert one fresh entry → eviction drops the now-oldest (ys[1]).
        b.evaluate_with_derivative(np.linspace(0.0, 1.0, 10 + maxsize))
        assert key_for(ys[0]) in cache  # protected by the refresh
        assert key_for(ys[1]) not in cache  # evicted instead

    def test_keys_distinguish_content_order_support(self):
        """No false hits: differing y content, order, or support never collide."""
        y = np.linspace(0.1, 0.9, 12)
        # Different content → different cached object.
        b = make_basis(order=4)
        B1, _ = b.evaluate_with_derivative(y)
        B2, _ = b.evaluate_with_derivative(np.linspace(0.1, 0.9, 13))
        assert B1 is not B2
        # Different order at equal y/support → distinct shapes, no collision.
        b5 = BernsteinBasis(order=5, support=(0.0, 1.0))
        B5, _ = b5.evaluate_with_derivative(y.copy())
        assert B1.shape[1] == 5 and B5.shape[1] == 6
        # Different support at equal y/order → distinct normalisation & scale.
        b_wide = BernsteinBasis(order=4, support=(-1.0, 2.0))
        _, dB_unit = b.evaluate_with_derivative(y.copy())
        _, dB_wide = b_wide.evaluate_with_derivative(y.copy())
        assert not np.allclose(dB_unit, dB_wide)


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


# ===========================================================================
# PolynomialBasis
# ===========================================================================


class TestPolynomialBasisConstruction:
    def test_valid(self):
        b = PolynomialBasis(order=3, support=(0.0, 2.0))
        assert b.order == 3
        assert b.support == (0.0, 2.0)

    def test_negative_order_raises(self):
        with pytest.raises(ValueError, match="order"):
            PolynomialBasis(order=-1, support=(0.0, 1.0))

    def test_zero_order_ok(self):
        b = PolynomialBasis(order=0, support=(0.0, 1.0))
        assert b.order == 0

    def test_support_reversed_raises(self):
        with pytest.raises(ValueError, match="support"):
            PolynomialBasis(order=3, support=(1.0, 0.0))

    def test_non_finite_support_raises(self):
        with pytest.raises(ValueError, match="finite"):
            PolynomialBasis(order=3, support=(0.0, np.inf))


class TestPolynomialBasisEvaluate:
    def test_shape(self):
        b = PolynomialBasis(order=4, support=(0.0, 1.0))
        y = np.linspace(0, 1, 20)
        assert b.evaluate(y).shape == (20, 5)

    def test_boundary_left_is_unit_vector(self):
        b = PolynomialBasis(order=4, support=(0.0, 2.0))
        M = b.evaluate(np.array([0.0]))
        expected = np.zeros(5)
        expected[0] = 1.0  # t=0: [1, 0, 0, 0, 0]
        np.testing.assert_allclose(M[0], expected, atol=1e-12)

    def test_boundary_right_is_all_ones(self):
        b = PolynomialBasis(order=3, support=(0.0, 2.0))
        M = b.evaluate(np.array([2.0]))
        np.testing.assert_allclose(M[0], [1.0, 1.0, 1.0, 1.0], atol=1e-12)

    def test_midpoint(self):
        b = PolynomialBasis(order=3, support=(0.0, 2.0))
        M = b.evaluate(np.array([1.0]))  # t = 0.5
        np.testing.assert_allclose(M[0], [1.0, 0.5, 0.25, 0.125], atol=1e-12)

    def test_empty_input(self):
        b = PolynomialBasis(order=2, support=(0.0, 1.0))
        assert b.evaluate(np.array([])).shape == (0, 3)

    def test_out_of_support_raises(self):
        b = PolynomialBasis(order=2, support=(0.0, 1.0))
        with pytest.raises(ValueError, match="outside support"):
            b.evaluate(np.array([1.5]))

    def test_order_zero_returns_ones(self):
        b = PolynomialBasis(order=0, support=(0.0, 1.0))
        M = b.evaluate(np.linspace(0, 1, 10))
        np.testing.assert_allclose(M, 1.0, atol=1e-12)


class TestPolynomialBasisDerivative:
    def test_shape(self):
        b = PolynomialBasis(order=4, support=(0.0, 1.0))
        D = b.derivative(np.linspace(0, 1, 20), order=1)
        assert D.shape == (20, 5)

    def test_first_column_zero(self):
        b = PolynomialBasis(order=4, support=(0.0, 3.0))
        D = b.derivative(np.linspace(0, 3, 20), order=1)
        np.testing.assert_allclose(D[:, 0], 0.0, atol=1e-12)

    def test_second_column_is_scale(self):
        # d/dy [y/(b-a)] = 1/(b-a)
        b = PolynomialBasis(order=2, support=(0.0, 4.0))
        D = b.derivative(np.linspace(0, 4, 20), order=1)
        np.testing.assert_allclose(D[:, 1], 1.0 / 4.0, atol=1e-12)

    def test_derivative_vs_finite_difference(self):
        b = PolynomialBasis(order=5, support=(0.0, 3.0))
        y = np.linspace(0.1, 2.9, 40)
        h = 1e-5
        fd = (b.evaluate(y + h) - b.evaluate(y - h)) / (2 * h)
        np.testing.assert_allclose(b.derivative(y, order=1), fd, atol=1e-5)

    def test_second_derivative_vs_finite_difference(self):
        b = PolynomialBasis(order=5, support=(0.0, 3.0))
        y = np.linspace(0.1, 2.9, 30)
        h = 1e-4
        fd2 = (b.evaluate(y + h) - 2 * b.evaluate(y) + b.evaluate(y - h)) / h**2
        np.testing.assert_allclose(b.derivative(y, order=2), fd2, atol=1e-4)

    def test_invalid_order_raises(self):
        b = PolynomialBasis(order=3, support=(0.0, 1.0))
        with pytest.raises(ValueError, match="order"):
            b.derivative(np.array([0.5]), order=0)


class TestPolynomialBasisIntegrate:
    def test_shape(self):
        b = PolynomialBasis(order=3, support=(0.0, 2.0))
        assert b.integrate(np.linspace(0, 2, 10)).shape == (10, 4)

    def test_zero_at_lower_bound(self):
        b = PolynomialBasis(order=3, support=(1.0, 3.0))
        np.testing.assert_allclose(b.integrate(np.array([1.0])), 0.0, atol=1e-12)

    def test_constant_basis_integral_equals_width(self):
        # ∫_0^b 1 dy = b
        b = PolynomialBasis(order=0, support=(0.0, 5.0))
        result = b.integrate(np.array([5.0]))
        np.testing.assert_allclose(result[0, 0], 5.0, atol=1e-12)

    def test_linear_basis_integral(self):
        # ∫_0^2 (y/2) dy = [y²/4]_0^2 = 1; result is (b-a)*t²/2 = 2*(1/2)=1
        b = PolynomialBasis(order=1, support=(0.0, 2.0))
        result = b.integrate(np.array([2.0]))  # t=1
        np.testing.assert_allclose(result[0], [2.0, 1.0], atol=1e-12)

    def test_integrate_vs_numerical(self):
        from scipy.integrate import quad

        b = PolynomialBasis(order=4, support=(0.0, 3.0))
        y_target = 2.0
        B_y = b.integrate(np.array([y_target]))[0]
        for col in range(5):
            numerical, _ = quad(
                lambda y: b.evaluate(np.array([y]))[0, col], 0.0, y_target
            )
            np.testing.assert_allclose(B_y[col], numerical, rtol=1e-6, atol=1e-10)


# ===========================================================================
# LegendreBasis
# ===========================================================================


class TestLegendreBasisConstruction:
    def test_valid(self):
        b = LegendreBasis(order=3, support=(-1.0, 1.0))
        assert b.order == 3

    def test_negative_order_raises(self):
        with pytest.raises(ValueError, match="order"):
            LegendreBasis(order=-1, support=(0.0, 1.0))

    def test_support_reversed_raises(self):
        with pytest.raises(ValueError, match="support"):
            LegendreBasis(order=2, support=(1.0, 0.0))

    def test_non_finite_support_raises(self):
        with pytest.raises(ValueError, match="finite"):
            LegendreBasis(order=2, support=(-np.inf, 1.0))


class TestLegendreBasisEvaluate:
    def test_shape(self):
        b = LegendreBasis(order=4, support=(0.0, 2.0))
        assert b.evaluate(np.linspace(0, 2, 15)).shape == (15, 5)

    def test_p0_is_one(self):
        b = LegendreBasis(order=0, support=(0.0, 1.0))
        M = b.evaluate(np.linspace(0, 1, 10))
        np.testing.assert_allclose(M[:, 0], 1.0, atol=1e-12)

    def test_p1_is_linear(self):
        # P_1(t) = t where t = 2*(y-a)/(b-a) - 1
        b = LegendreBasis(order=1, support=(0.0, 2.0))
        y = np.linspace(0, 2, 11)
        t = 2 * y / 2 - 1  # t ∈ [-1, 1]
        M = b.evaluate(y)
        np.testing.assert_allclose(M[:, 1], t, atol=1e-12)

    def test_p2_is_quadratic(self):
        # P_2(t) = (3t² − 1)/2
        b = LegendreBasis(order=2, support=(-1.0, 1.0))
        y = np.linspace(-1, 1, 11)
        t = y  # support is [-1,1] so t = y
        M = b.evaluate(y)
        expected = (3 * t**2 - 1) / 2
        np.testing.assert_allclose(M[:, 2], expected, atol=1e-12)

    def test_orthogonality(self):
        """∫_{-1}^{1} P_m(t) P_n(t) dt ≈ 2/(2n+1) δ_{mn}."""
        b = LegendreBasis(order=4, support=(-1.0, 1.0))
        y = np.linspace(-1, 1, 2000)
        M = b.evaluate(y)
        dy = y[1] - y[0]
        gram = M.T @ M * dy
        expected = np.diag([2 / (2 * n + 1) for n in range(5)])
        np.testing.assert_allclose(gram, expected, atol=5e-3)

    def test_empty_input(self):
        b = LegendreBasis(order=3, support=(0.0, 1.0))
        assert b.evaluate(np.array([])).shape == (0, 4)

    def test_out_of_support_raises(self):
        b = LegendreBasis(order=2, support=(0.0, 1.0))
        with pytest.raises(ValueError, match="outside support"):
            b.evaluate(np.array([1.5]))


class TestLegendreBasisDerivative:
    def test_shape(self):
        b = LegendreBasis(order=3, support=(0.0, 2.0))
        assert b.derivative(np.linspace(0, 2, 20)).shape == (20, 4)

    def test_p0_derivative_is_zero(self):
        b = LegendreBasis(order=0, support=(0.0, 1.0))
        D = b.derivative(np.linspace(0, 1, 10))
        np.testing.assert_allclose(D[:, 0], 0.0, atol=1e-12)

    def test_p1_derivative_is_constant(self):
        # P_1(t) = t, d/dy P_1(t) = dt/dy = 2/(b-a)
        b = LegendreBasis(order=1, support=(0.0, 4.0))
        D = b.derivative(np.linspace(0, 4, 20))
        np.testing.assert_allclose(D[:, 1], 2 / 4, atol=1e-12)

    def test_derivative_vs_finite_difference(self):
        b = LegendreBasis(order=5, support=(0.0, 3.0))
        y = np.linspace(0.1, 2.9, 40)
        h = 1e-5
        fd = (b.evaluate(y + h) - b.evaluate(y - h)) / (2 * h)
        np.testing.assert_allclose(b.derivative(y, order=1), fd, atol=1e-5)

    def test_invalid_order_raises(self):
        b = LegendreBasis(order=3, support=(0.0, 1.0))
        with pytest.raises(ValueError, match="order"):
            b.derivative(np.array([0.5]), order=3)


class TestLegendreBasisIntegrate:
    def test_shape(self):
        b = LegendreBasis(order=3, support=(0.0, 2.0))
        assert b.integrate(np.linspace(0, 2, 10)).shape == (10, 4)

    def test_zero_at_lower_bound(self):
        b = LegendreBasis(order=3, support=(1.0, 3.0))
        np.testing.assert_allclose(b.integrate(np.array([1.0])), 0.0, atol=1e-12)

    def test_p0_integral_equals_width(self):
        # ∫_0^b P_0 dy = b - 0 = b
        b = LegendreBasis(order=0, support=(0.0, 3.0))
        result = b.integrate(np.array([3.0]))
        np.testing.assert_allclose(result[0, 0], 3.0, atol=1e-12)

    def test_integrate_vs_numerical(self):
        from scipy.integrate import quad

        b = LegendreBasis(order=4, support=(0.0, 3.0))
        y_target = 2.0
        B_y = b.integrate(np.array([y_target]))[0]
        for col in range(5):
            numerical, _ = quad(
                lambda y: b.evaluate(np.array([y]))[0, col], 0.0, y_target
            )
            np.testing.assert_allclose(B_y[col], numerical, rtol=1e-6, atol=1e-10)


# ===========================================================================
# LogBasis
# ===========================================================================


class TestLogBasisConstruction:
    def test_valid(self):
        b = LogBasis(support=(0.5, 5.0))
        assert b.order == 0
        assert b.support == (0.5, 5.0)

    def test_non_positive_lower_bound_raises(self):
        with pytest.raises(ValueError, match="positive"):
            LogBasis(support=(0.0, 5.0))

    def test_negative_lower_bound_raises(self):
        with pytest.raises(ValueError, match="positive"):
            LogBasis(support=(-1.0, 5.0))

    def test_support_reversed_raises(self):
        with pytest.raises(ValueError, match="support"):
            LogBasis(support=(5.0, 1.0))

    def test_non_finite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            LogBasis(support=(1.0, np.inf))


class TestLogBasisEvaluate:
    def test_shape(self):
        b = LogBasis(support=(0.5, 5.0))
        assert b.evaluate(np.array([1.0, 2.0, 3.0])).shape == (3, 1)

    def test_log_one_is_zero(self):
        b = LogBasis(support=(0.5, 5.0))
        np.testing.assert_allclose(b.evaluate(np.array([1.0])), [[0.0]], atol=1e-12)

    def test_log_e_is_one(self):
        b = LogBasis(support=(0.5, 5.0))
        np.testing.assert_allclose(b.evaluate(np.array([np.e])), [[1.0]], atol=1e-12)

    def test_empty_input(self):
        b = LogBasis(support=(0.5, 5.0))
        assert b.evaluate(np.array([])).shape == (0, 1)

    def test_out_of_support_raises(self):
        b = LogBasis(support=(1.0, 5.0))
        with pytest.raises(ValueError, match="outside support"):
            b.evaluate(np.array([6.0]))


class TestLogBasisDerivative:
    def test_shape(self):
        b = LogBasis(support=(0.5, 5.0))
        assert b.derivative(np.array([1.0, 2.0])).shape == (2, 1)

    def test_derivative_at_one(self):
        b = LogBasis(support=(0.5, 5.0))
        # d/dy log(y) = 1/y; at y=1 → 1
        np.testing.assert_allclose(b.derivative(np.array([1.0])), [[1.0]], atol=1e-12)

    def test_derivative_vs_finite_difference(self):
        b = LogBasis(support=(0.5, 5.0))
        y = np.linspace(0.6, 4.9, 30)
        h = 1e-6
        fd = (b.evaluate(y + h) - b.evaluate(y - h)) / (2 * h)
        np.testing.assert_allclose(b.derivative(y, order=1), fd, atol=1e-5)

    def test_invalid_order_raises(self):
        b = LogBasis(support=(0.5, 5.0))
        with pytest.raises(ValueError, match="order"):
            b.derivative(np.array([1.0]), order=2)


class TestLogBasisIntegrate:
    def test_shape(self):
        b = LogBasis(support=(0.5, 5.0))
        assert b.integrate(np.array([1.0, 2.0])).shape == (2, 1)

    def test_zero_at_lower_bound(self):
        b = LogBasis(support=(1.0, 5.0))
        np.testing.assert_allclose(b.integrate(np.array([1.0])), [[0.0]], atol=1e-12)

    def test_integrate_vs_numerical(self):
        from scipy.integrate import quad

        b = LogBasis(support=(1.0, 5.0))
        y_target = 3.0
        result = b.integrate(np.array([y_target]))[0, 0]
        numerical, _ = quad(lambda y: np.log(y), 1.0, y_target)
        np.testing.assert_allclose(result, numerical, rtol=1e-8, atol=1e-12)


# ===========================================================================
# InterceptBasis
# ===========================================================================


class TestInterceptBasisConstruction:
    def test_valid(self):
        b = InterceptBasis(support=(0.0, 5.0))
        assert b.order == 0
        assert b.support == (0.0, 5.0)

    def test_support_reversed_raises(self):
        with pytest.raises(ValueError, match="support"):
            InterceptBasis(support=(5.0, 0.0))

    def test_non_finite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            InterceptBasis(support=(0.0, np.inf))


class TestInterceptBasisEvaluate:
    def test_shape(self):
        b = InterceptBasis(support=(0.0, 5.0))
        assert b.evaluate(np.array([1.0, 2.0, 3.0])).shape == (3, 1)

    def test_all_ones(self):
        b = InterceptBasis(support=(0.0, 5.0))
        M = b.evaluate(np.linspace(0, 5, 20))
        np.testing.assert_allclose(M, 1.0, atol=1e-12)

    def test_empty_input(self):
        b = InterceptBasis(support=(0.0, 5.0))
        assert b.evaluate(np.array([])).shape == (0, 1)

    def test_out_of_support_raises(self):
        b = InterceptBasis(support=(0.0, 5.0))
        with pytest.raises(ValueError, match="outside support"):
            b.evaluate(np.array([6.0]))


class TestInterceptBasisDerivative:
    def test_all_zeros(self):
        b = InterceptBasis(support=(0.0, 5.0))
        D = b.derivative(np.linspace(0, 5, 20))
        np.testing.assert_allclose(D, 0.0, atol=1e-12)

    def test_shape(self):
        b = InterceptBasis(support=(0.0, 5.0))
        assert b.derivative(np.array([1.0, 2.0])).shape == (2, 1)

    def test_invalid_order_raises(self):
        b = InterceptBasis(support=(0.0, 5.0))
        with pytest.raises(ValueError, match="order"):
            b.derivative(np.array([1.0]), order=3)


class TestInterceptBasisIntegrate:
    def test_shape(self):
        b = InterceptBasis(support=(0.0, 5.0))
        assert b.integrate(np.array([1.0, 2.0])).shape == (2, 1)

    def test_zero_at_lower_bound(self):
        b = InterceptBasis(support=(0.0, 5.0))
        np.testing.assert_allclose(b.integrate(np.array([0.0])), [[0.0]], atol=1e-12)

    def test_integral_equals_y_minus_a(self):
        b = InterceptBasis(support=(1.0, 5.0))
        y = np.array([1.5, 2.0, 3.0, 5.0])
        result = b.integrate(y)
        np.testing.assert_allclose(result[:, 0], y - 1.0, atol=1e-12)

    def test_integrate_vs_numerical(self):
        from scipy.integrate import quad

        b = InterceptBasis(support=(0.0, 5.0))
        y_target = 3.0
        result = b.integrate(np.array([y_target]))[0, 0]
        numerical, _ = quad(lambda _: 1.0, 0.0, y_target)
        np.testing.assert_allclose(result, numerical, rtol=1e-8)
