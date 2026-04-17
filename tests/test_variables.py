"""Tests for pymlt.variables — variable types and CensoredData."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pymlt.variables import (
    CensoredData,
    NumericVariable,
    OrderedVariable,
    SurvivalVariable,
)

# ---------------------------------------------------------------------------
# NumericVariable
# ---------------------------------------------------------------------------

class TestNumericVariable:
    def test_basic_creation(self):
        v = NumericVariable("y", support=(0.0, 1.0))
        assert v.name == "y"
        assert v.support == (0.0, 1.0)
        assert v.bounds is None
        assert v.log_first is False

    def test_with_bounds(self):
        v = NumericVariable("y", support=(-5.0, 5.0), bounds=(-3.0, 3.0))
        assert v.bounds == (-3.0, 3.0)

    def test_with_log_first(self):
        v = NumericVariable("y", support=(0.0, 10.0), log_first=True)
        assert v.log_first is True

    def test_support_equal_raises(self):
        with pytest.raises(ValueError, match="support"):
            NumericVariable("y", support=(1.0, 1.0))

    def test_support_reversed_raises(self):
        with pytest.raises(ValueError, match="support"):
            NumericVariable("y", support=(2.0, 1.0))

    def test_bounds_reversed_raises(self):
        with pytest.raises(ValueError, match="bounds"):
            NumericVariable("y", support=(0.0, 10.0), bounds=(5.0, 1.0))

    def test_bounds_equal_raises(self):
        with pytest.raises(ValueError, match="bounds"):
            NumericVariable("y", support=(0.0, 10.0), bounds=(3.0, 3.0))

    @given(
        a=st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
        width=st.floats(1e-6, 1e6, allow_nan=False, allow_infinity=False),
    )
    def test_valid_support_never_raises(self, a, width):
        v = NumericVariable("y", support=(a, a + width))
        assert v.support[0] < v.support[1]


# ---------------------------------------------------------------------------
# OrderedVariable
# ---------------------------------------------------------------------------

class TestOrderedVariable:
    def test_basic_creation(self):
        v = OrderedVariable("grade", levels=["low", "medium", "high"])
        assert v.name == "grade"
        assert v.levels == ["low", "medium", "high"]
        assert v.n_levels == 3

    def test_two_levels_ok(self):
        v = OrderedVariable("x", levels=["a", "b"])
        assert v.n_levels == 2

    def test_one_level_raises(self):
        with pytest.raises(ValueError, match="at least 2 levels"):
            OrderedVariable("x", levels=["only"])

    def test_zero_levels_raises(self):
        with pytest.raises(ValueError, match="at least 2 levels"):
            OrderedVariable("x", levels=[])

    def test_duplicate_levels_raises(self):
        with pytest.raises(ValueError, match="unique"):
            OrderedVariable("x", levels=["a", "b", "a"])

    @given(st.lists(st.text(min_size=1), min_size=2, unique=True))
    def test_valid_levels_never_raises(self, levels):
        v = OrderedVariable("x", levels=levels)
        assert v.n_levels == len(levels)


# ---------------------------------------------------------------------------
# SurvivalVariable
# ---------------------------------------------------------------------------

class TestSurvivalVariable:
    def test_default_support(self):
        v = SurvivalVariable("t")
        assert v.support[0] == 0.0
        assert v.support[1] == float("inf")

    def test_finite_support(self):
        v = SurvivalVariable("t", support=(0.0, 100.0))
        assert v.support == (0.0, 100.0)

    def test_negative_support_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            SurvivalVariable("t", support=(-1.0, 10.0))

    def test_reversed_support_raises(self):
        with pytest.raises(ValueError, match="a < b"):
            SurvivalVariable("t", support=(5.0, 2.0))

    def test_equal_support_raises(self):
        with pytest.raises(ValueError, match="a < b"):
            SurvivalVariable("t", support=(1.0, 1.0))


# ---------------------------------------------------------------------------
# CensoredData — constructors
# ---------------------------------------------------------------------------

class TestCensoredDataFromExact:
    def test_basic(self):
        y = np.array([1.0, 2.0, 3.0])
        cd = CensoredData.from_exact(y)
        assert cd.n == 3
        assert cd.n_exact == 3
        assert cd.n_censored == 0
        np.testing.assert_array_equal(cd.exact, y)
        np.testing.assert_array_equal(cd.lower, y)
        np.testing.assert_array_equal(cd.upper, y)

    def test_masks(self):
        cd = CensoredData.from_exact(np.array([0.5, 1.5]))
        assert cd.is_exact_mask.all()
        assert not cd.is_right_censored_mask.any()
        assert not cd.is_left_censored_mask.any()
        assert not cd.is_interval_censored_mask.any()

    def test_input_not_mutated(self):
        y = np.array([1.0, 2.0])
        cd = CensoredData.from_exact(y)
        y[0] = 99.0
        assert cd.exact[0] == 1.0

    @given(arrays(float, st.integers(1, 50), elements=st.floats(-1e6, 1e6, allow_nan=False)))
    def test_all_exact_for_any_finite_array(self, y):
        cd = CensoredData.from_exact(y)
        assert cd.is_exact_mask.all()
        assert cd.n_exact == len(y)


class TestCensoredDataRightCensored:
    def test_no_censoring(self):
        y = np.array([1.0, 2.0, 3.0])
        censored = np.array([False, False, False])
        cd = CensoredData.right_censored(y, censored)
        assert cd.n_exact == 3
        assert cd.n_censored == 0
        assert cd.is_exact_mask.all()

    def test_all_censored(self):
        y = np.array([1.0, 2.0, 3.0])
        censored = np.array([True, True, True])
        cd = CensoredData.right_censored(y, censored)
        assert cd.n_exact == 0
        assert cd.n_censored == 3
        assert cd.is_right_censored_mask.all()
        np.testing.assert_array_equal(cd.lower, y)
        assert np.all(np.isinf(cd.upper))

    def test_mixed(self):
        y = np.array([1.0, 2.0, 3.0])
        censored = np.array([False, True, False])
        cd = CensoredData.right_censored(y, censored)
        assert cd.n_exact == 2
        assert cd.n_censored == 1
        assert cd.is_exact_mask[0] and not cd.is_exact_mask[1] and cd.is_exact_mask[2]
        assert cd.is_right_censored_mask[1]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            CensoredData.right_censored(np.array([1.0, 2.0]), np.array([True]))


class TestCensoredDataLeftCensored:
    def test_all_censored(self):
        y = np.array([1.0, 2.0])
        censored = np.array([True, True])
        cd = CensoredData.left_censored(y, censored)
        assert cd.n_exact == 0
        assert cd.is_left_censored_mask.all()
        assert np.all(np.isinf(cd.lower) & (cd.lower < 0))
        np.testing.assert_array_equal(cd.upper, y)

    def test_mixed(self):
        y = np.array([1.0, 2.0, 3.0])
        censored = np.array([True, False, True])
        cd = CensoredData.left_censored(y, censored)
        assert cd.n_exact == 1
        assert cd.is_exact_mask[1]
        assert cd.is_left_censored_mask[0] and cd.is_left_censored_mask[2]


class TestCensoredDataIntervalCensored:
    def test_basic(self):
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([1.0, 2.0, 3.0])
        cd = CensoredData.interval_censored(lower, upper)
        assert cd.n_exact == 0
        assert cd.n_censored == 3
        assert cd.is_interval_censored_mask.all()

    def test_equal_bounds_allowed(self):
        # degenerate interval = point mass
        cd = CensoredData.interval_censored(np.array([1.0]), np.array([1.0]))
        assert cd.is_interval_censored_mask[0]

    def test_inverted_bounds_raises(self):
        with pytest.raises(ValueError, match="lower must be <= upper"):
            CensoredData.interval_censored(np.array([2.0]), np.array([1.0]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            CensoredData.interval_censored(np.array([1.0, 2.0]), np.array([3.0]))


# ---------------------------------------------------------------------------
# CensoredData — direct construction / validation
# ---------------------------------------------------------------------------

class TestCensoredDataValidation:
    def test_length_mismatch_lower_raises(self):
        with pytest.raises(ValueError, match="same length"):
            CensoredData(
                exact=np.array([1.0, 2.0]),
                lower=np.array([1.0]),
                upper=np.array([1.0, 2.0]),
            )

    def test_length_mismatch_upper_raises(self):
        with pytest.raises(ValueError, match="same length"):
            CensoredData(
                exact=np.array([1.0, 2.0]),
                lower=np.array([1.0, 2.0]),
                upper=np.array([2.0]),
            )

    def test_lower_greater_than_upper_raises(self):
        with pytest.raises(ValueError, match="lower must be <= upper"):
            CensoredData(
                exact=np.array([np.nan]),
                lower=np.array([3.0]),
                upper=np.array([1.0]),
            )

    def test_trunc_lower_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="trunc_lower"):
            CensoredData(
                exact=np.array([1.0, 2.0]),
                lower=np.array([1.0, 2.0]),
                upper=np.array([1.0, 2.0]),
                trunc_lower=np.array([0.0]),
            )

    def test_trunc_upper_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="trunc_upper"):
            CensoredData(
                exact=np.array([1.0, 2.0]),
                lower=np.array([1.0, 2.0]),
                upper=np.array([1.0, 2.0]),
                trunc_upper=np.array([5.0]),
            )

    def test_truncation_stored_correctly(self):
        cd = CensoredData(
            exact=np.array([1.0, 2.0]),
            lower=np.array([1.0, 2.0]),
            upper=np.array([1.0, 2.0]),
            trunc_lower=np.array([0.0, 0.5]),
            trunc_upper=np.array([3.0, 4.0]),
        )
        np.testing.assert_array_equal(cd.trunc_lower, [0.0, 0.5])
        np.testing.assert_array_equal(cd.trunc_upper, [3.0, 4.0])


# ---------------------------------------------------------------------------
# CensoredData — mask exclusivity (property-based)
# ---------------------------------------------------------------------------

@given(
    y=arrays(float, st.integers(1, 30), elements=st.floats(-1e3, 1e3, allow_nan=False)),
    censored=arrays(bool, st.integers(1, 30), elements=st.booleans()),
)
@settings(max_examples=200)
def test_right_censored_masks_are_mutually_exclusive(y, censored):
    if len(y) != len(censored):
        return  # skip incompatible shapes generated by hypothesis
    cd = CensoredData.right_censored(y, censored)
    # Each observation belongs to exactly one category
    total = (
        cd.is_exact_mask.astype(int)
        + cd.is_right_censored_mask.astype(int)
        + cd.is_left_censored_mask.astype(int)
        + cd.is_interval_censored_mask.astype(int)
    )
    assert (total == 1).all(), "masks must be mutually exclusive and exhaustive"


@given(
    y=arrays(float, st.integers(1, 30), elements=st.floats(-1e3, 1e3, allow_nan=False)),
)
def test_from_exact_masks_sum_to_one(y):
    cd = CensoredData.from_exact(y)
    total = (
        cd.is_exact_mask.astype(int)
        + cd.is_right_censored_mask.astype(int)
        + cd.is_left_censored_mask.astype(int)
        + cd.is_interval_censored_mask.astype(int)
    )
    assert (total == 1).all()
