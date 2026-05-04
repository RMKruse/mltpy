"""Tests for Polr — proportional-odds ordinal regression."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pymlt
from pymlt import OrderedVariable, OrdinalBasis, Polr
from pymlt.variables import CensoredData

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"


# ---------------------------------------------------------------------------
# OrderedVariable
# ---------------------------------------------------------------------------


class TestOrderedVariable:
    def test_K_property(self):
        v = OrderedVariable(("a", "b", "c"))
        assert v.K == 3

    def test_requires_two_levels(self):
        with pytest.raises(ValueError, match="at least 2 levels"):
            OrderedVariable(("only-one",))

    def test_unique_levels_required(self):
        with pytest.raises(ValueError, match="unique"):
            OrderedVariable(("a", "b", "a"))

    def test_encode_decode_round_trip(self):
        v = OrderedVariable(("low", "mid", "high"))
        labels = ["high", "low", "mid", "high", "low"]
        codes = v.encode(labels)
        np.testing.assert_array_equal(codes, [3, 1, 2, 3, 1])
        np.testing.assert_array_equal(v.decode(codes), labels)

    def test_unknown_label_raises(self):
        v = OrderedVariable(("low", "high"))
        with pytest.raises(ValueError, match="not in the OrderedVariable levels"):
            v.encode(["low", "middle", "high"])

    def test_decode_out_of_range_raises(self):
        v = OrderedVariable(("a", "b"))
        with pytest.raises(ValueError, match=r"Codes must be in"):
            v.decode(np.array([0, 1], dtype=np.intp))
        with pytest.raises(ValueError, match=r"Codes must be in"):
            v.decode(np.array([1, 3], dtype=np.intp))

    def test_from_labels_inferred_levels(self):
        labels = ["b", "a", "c", "b", "a"]
        var, cd = OrderedVariable.from_labels(labels)
        # Inferred levels are sorted unique values.
        assert var.levels == ("a", "b", "c")
        assert isinstance(cd, CensoredData)
        # Bounds for level 'a' (code 1) → (-inf, 1]; 'c' (code 3) → (2, +inf]
        assert cd.lower[0] == 1.0  # 'b' → (1, 2]
        assert cd.upper[0] == 2.0
        assert cd.lower[1] == -np.inf  # 'a' → (-inf, 1]
        assert cd.upper[1] == 1.0
        assert cd.lower[2] == 2.0  # 'c' → (2, +inf)
        assert cd.upper[2] == np.inf

    def test_from_labels_explicit_levels(self):
        labels = ["mid", "low", "high"]
        var, cd = OrderedVariable.from_labels(labels, levels=("low", "mid", "high"))
        assert var.levels == ("low", "mid", "high")
        # mid → code 2 → (1, 2]
        assert cd.lower[0] == 1.0
        assert cd.upper[0] == 2.0


# ---------------------------------------------------------------------------
# OrdinalBasis
# ---------------------------------------------------------------------------


class TestOrdinalBasis:
    def test_K_must_be_at_least_two(self):
        with pytest.raises(ValueError, match="K must be >= 2"):
            OrdinalBasis(K=1)

    def test_order_and_support(self):
        b = OrdinalBasis(K=4)
        assert b.order == 2  # K - 2; n_params = order + 1 = K - 1 = 3
        assert b.support == (0.0, 4.0)

    def test_evaluate_one_hot(self):
        b = OrdinalBasis(K=4)
        # Cut positions 1..3 → one-hot rows of length 3
        m = b.evaluate(np.array([1, 2, 3, 1]))
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(m, expected)

    def test_evaluate_rejects_out_of_range(self):
        b = OrdinalBasis(K=3)  # cut positions in {1, 2}
        with pytest.raises(ValueError, match=r"cut positions must be in"):
            b.evaluate(np.array([0, 1]))
        with pytest.raises(ValueError, match=r"cut positions must be in"):
            b.evaluate(np.array([1, 3]))

    def test_evaluate_rejects_non_integer(self):
        b = OrdinalBasis(K=3)
        with pytest.raises(ValueError, match="integer cut positions"):
            b.evaluate(np.array([1.5]))

    def test_derivative_is_zero(self):
        b = OrdinalBasis(K=4)
        d = b.derivative(np.array([1, 2, 3]))
        np.testing.assert_array_equal(d, np.zeros((3, 3)))

    def test_integrate_not_implemented(self):
        b = OrdinalBasis(K=3)
        with pytest.raises(NotImplementedError):
            b.integrate(np.array([1.0]))


# ---------------------------------------------------------------------------
# Polr — basic fitting
# ---------------------------------------------------------------------------


def _make_synthetic_polr(
    n: int = 200, K: int = 3, q: int = 2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    levels = tuple(f"L{k + 1}" for k in range(K))
    probs = np.full(K, 1.0 / K)
    y = rng.choice(np.array(levels), size=n, p=probs)
    X = rng.standard_normal((n, q))
    return y, X, levels


class TestPolrSmoke:
    def test_fit_and_basic_attributes(self):
        y, X, levels = _make_synthetic_polr(seed=1)
        m = Polr(levels=levels).fit(y, X)
        assert m.is_fitted_
        assert m.cutpoints_.shape == (len(levels) - 1,)
        assert m.coef_.shape == (X.shape[1],)
        assert np.all(np.diff(m.cutpoints_) >= -1e-9)  # non-decreasing

    def test_predict_proba_shape_and_simplex(self):
        y, X, levels = _make_synthetic_polr(seed=2)
        m = Polr(levels=levels).fit(y, X)
        probs = m.predict_proba(X[:10])
        assert probs.shape == (10, len(levels))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_predict_class_returns_known_labels(self):
        y, X, levels = _make_synthetic_polr(seed=3)
        m = Polr(levels=levels).fit(y, X)
        cls = m.predict_class(X[:5])
        assert all(c in levels for c in cls)

    def test_K2_binary_case(self):
        rng = np.random.default_rng(11)
        y = rng.choice(("no", "yes"), size=150, p=[0.4, 0.6])
        X = rng.standard_normal((150, 2))
        m = Polr(levels=("no", "yes")).fit(y, X)
        # K=2 ⇒ exactly 1 cutpoint, q=2 ⇒ 2 coefficients
        assert m.cutpoints_.shape == (1,)
        assert m.coef_.shape == (2,)
        probs = m.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_no_covariates_fit(self):
        y, _, levels = _make_synthetic_polr(q=0, seed=4)
        m = Polr(levels=levels).fit(y)
        assert m.coef_.shape == (0,)
        # predict_proba with no X → single row of marginal class probabilities.
        probs = m.predict_proba()
        assert probs.shape == (1, len(levels))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_disabled(self):
        y, X, levels = _make_synthetic_polr(seed=5)
        m = Polr(levels=levels).fit(y, X)
        with pytest.raises(NotImplementedError, match="predict_proba"):
            m.predict()

    def test_unfitted_levels_raises(self):
        m = Polr()
        with pytest.raises(pymlt.NotFittedError):
            _ = m.levels_

    def test_inferred_levels_from_pandas_categorical(self):
        pd = pytest.importorskip("pandas")
        y = pd.Categorical(
            ["high", "low", "mid", "high"],
            categories=["low", "mid", "high"],
            ordered=True,
        )
        m = Polr().fit(y)
        assert m.levels_ == ("low", "mid", "high")

    def test_predict_proba_requires_X_when_fitted_with_X(self):
        y, X, levels = _make_synthetic_polr(seed=6)
        m = Polr(levels=levels).fit(y, X)
        with pytest.raises(ValueError, match="X must be provided"):
            m.predict_proba()

    def test_predict_proba_X_shape_mismatch(self):
        y, X, levels = _make_synthetic_polr(seed=7)
        m = Polr(levels=levels).fit(y, X)
        with pytest.raises(ValueError, match="covariate coefficients"):
            m.predict_proba(np.zeros((4, X.shape[1] + 1)))

    def test_invalid_distribution_raises(self):
        with pytest.raises(ValueError):
            Polr(distribution="invalid")  # type: ignore[arg-type]


class TestPolrSummary:
    def test_unfitted_summary(self):
        s = Polr().summary()
        assert "fitted=False" in s

    def test_fitted_summary_contains_cutpoints_and_coefs(self):
        y, X, levels = _make_synthetic_polr(seed=8)
        m = Polr(levels=levels).fit(y, X)
        s = m.summary()
        assert "Cutpoints" in s
        assert "Coefficients" in s
        # Cut name format: "L1|L2"
        assert "|" in s


# ---------------------------------------------------------------------------
# R reference comparison
# ---------------------------------------------------------------------------


_POLR_REFS_AVAILABLE = (REFERENCE_DIR / "polr_logistic_theta.txt").exists()
_POLR_DISTRIBUTIONS = [
    ("logistic", "logistic"),
    ("probit", "normal"),
    ("cloglog", "min_extreme_value"),
]


@pytest.mark.skipif(
    not _POLR_REFS_AVAILABLE,
    reason="Polr R reference data not generated; run Rscript reference/generate_reference.R",
)
class TestPolrReference:
    """Validate Polr against R tram::Polr output for three link functions."""

    @pytest.fixture
    def fixture_data(self) -> dict:
        y_labels = (REFERENCE_DIR / "polr_y.txt").read_text().strip().splitlines()
        X_flat = np.loadtxt(REFERENCE_DIR / "polr_X.txt")
        n = len(y_labels)
        q = X_flat.size // n
        X = X_flat.reshape(n, q)
        levels = ("low", "mid", "high")
        return {"y": y_labels, "X": X, "n": n, "K": len(levels), "levels": levels}

    @pytest.mark.parametrize(("r_label", "py_dist"), _POLR_DISTRIBUTIONS)
    def test_cutpoints_match_r(self, fixture_data, r_label, py_dist):
        theta_r = np.loadtxt(REFERENCE_DIR / f"polr_{r_label}_theta.txt")
        K = fixture_data["K"]
        # In R coef(.., with_baseline=TRUE) the first K-1 entries are the
        # cutpoints (raw, no sign flip).
        cutpoints_r = theta_r[: K - 1]
        m = Polr(levels=fixture_data["levels"], distribution=py_dist).fit(
            fixture_data["y"], fixture_data["X"]
        )
        np.testing.assert_allclose(m.cutpoints_, cutpoints_r, atol=1e-3)

    @pytest.mark.parametrize(("r_label", "py_dist"), _POLR_DISTRIBUTIONS)
    def test_coef_matches_r_with_sign_flip(self, fixture_data, r_label, py_dist):
        theta_r = np.loadtxt(REFERENCE_DIR / f"polr_{r_label}_theta.txt")
        K = fixture_data["K"]
        # R parameterises h - X·β; pymlt uses h + X·β.  Negate to compare.
        beta_r = theta_r[K - 1 :]
        m = Polr(levels=fixture_data["levels"], distribution=py_dist).fit(
            fixture_data["y"], fixture_data["X"]
        )
        np.testing.assert_allclose(m.coef_, -beta_r, atol=1e-3)

    @pytest.mark.parametrize(("r_label", "py_dist"), _POLR_DISTRIBUTIONS)
    def test_loglik_matches_r(self, fixture_data, r_label, py_dist):
        ll_r = float(np.loadtxt(REFERENCE_DIR / f"polr_{r_label}_loglik.txt"))
        m = Polr(levels=fixture_data["levels"], distribution=py_dist).fit(
            fixture_data["y"], fixture_data["X"]
        )
        assert m.result_ is not None
        np.testing.assert_allclose(m.result_.log_likelihood, ll_r, atol=1e-4)

    @pytest.mark.parametrize(("r_label", "py_dist"), _POLR_DISTRIBUTIONS)
    def test_predict_proba_matches_r(self, fixture_data, r_label, py_dist):
        proba_r_flat = np.loadtxt(REFERENCE_DIR / f"polr_{r_label}_proba.txt")
        n, K = fixture_data["n"], fixture_data["K"]
        proba_r = proba_r_flat.reshape(n, K)
        m = Polr(levels=fixture_data["levels"], distribution=py_dist).fit(
            fixture_data["y"], fixture_data["X"]
        )
        proba_py = m.predict_proba(fixture_data["X"])
        np.testing.assert_allclose(proba_py, proba_r, atol=1e-4)
