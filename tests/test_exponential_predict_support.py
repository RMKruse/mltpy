"""Predict-side support check for the exponential link (issue #157, ADR 0005).

Under ``base_distribution="exponential"`` the fit enforces the lower-support-end
rule ``h(a|x_i) ≥ 0`` for every *training* row only.  ``predict`` at a new ``x``
with ``h(a|x) < 0`` therefore leaves the support of the standard exponential:
the CDF near ``a`` is negative.  ADR 0005 consequence 4: warn with a
``UserWarning`` naming the offending rows and return the raw values (no
clamping at ``max(h, 0)``, no raising).

Follow-up note — scaled path: ``scaling=`` with the exponential link fits
only without a shift block so far (#114).  There ``h(a|x) = θ_b[0]·f(x) ≥ 0``
for every ``x``, so the warning cannot fire
(``tests/test_exponential_scaling_no_shift.py`` checks it stays silent under
extrapolation).  The predict-side check already uses the scaled form
``θ_b[0]·exp(0.5·x_s·γ) + x_d·β + offset``; a scaled, inside-the-hull case
should be added once the shift + scale slice (#115, map #117) makes that fit
possible.

Fixture
-------
Left-censored data from ``h(y|x) = 2y + 1 − x`` on support ``[0, 3]``.  With
exact or right-censored rows the exponential shift log-likelihood depends on
``β`` only through ``−Σ x_i β``, which pins every ``h(a|x_i)`` to 0 and ``β`` to
0; left-censored rows reward a larger ``h`` and make ``β`` identifiable.  The
fit lands at ``θ_b[0] ≈ 1.04``, ``β ≈ −0.87``: ``h(a|x) ≥ 0`` on the training
range ``x ∈ [0, 1]`` and ``h(a|2) ≈ −0.69``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.typing import NDArray

from mltpy.basis import BernsteinBasis
from mltpy.model import ConditionalTransformationModel
from mltpy.variables import CensoredData, CensoringType

ExpoFit = tuple[ConditionalTransformationModel, NDArray[np.float64]]

_SUPPORT = (0.0, 3.0)


@pytest.fixture(scope="module")
def expo_fit() -> ExpoFit:
    """Exponential-link shift model with a non-zero ``β`` (see module notes)."""
    rng = np.random.default_rng(1)
    n = 200
    x = rng.uniform(0.0, 1.0, n)
    e = rng.exponential(1.0, n)
    y = np.clip((e - (1.0 - x)) / 2.0, 0.0, 2.99)
    c = 0.25
    cens = y <= c
    data = CensoredData.left_censored(np.where(cens, c, y), cens)
    model = ConditionalTransformationModel(
        BernsteinBasis(order=3, support=_SUPPORT),
        censoring=CensoringType.LEFT,
        base_distribution="exponential",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data, X=x[:, None])
    return model, x


# Rows 1 and 3 extrapolate past x ≈ 1.2, where h(a|x) < 0.
_X_NEW = np.array([[0.5], [2.0], [0.2], [3.0]])
_BAD_ROWS = "[1, 3]"


_ALL_WHAT = [
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
    "quantile",
]


def _support_warnings(rec: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in rec if "h(a|x) < 0" in str(w.message)]


def _y_for(what: str, m: int) -> np.ndarray:
    """Probabilities for ``quantile``, response values otherwise."""
    return np.full(m, 0.5) if what == "quantile" else np.full(m, 1.0)


class TestExponentialSupportWarning:
    @pytest.mark.parametrize("what", _ALL_WHAT)
    def test_extrapolation_warns_naming_rows(
        self, expo_fit: ExpoFit, what: str
    ) -> None:
        model, _ = expo_fit
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model.predict(_y_for(what, 4), X_new=_X_NEW, what=what)
        msgs = _support_warnings(rec)
        assert len(msgs) == 1
        assert _BAD_ROWS in msgs[0]
        hits = [w for w in rec if "h(a|x) < 0" in str(w.message)]
        assert issubclass(hits[0].category, UserWarning)
        # stacklevel points at the caller of predict(), i.e. this test file.
        assert hits[0].filename == __file__

    @pytest.mark.parametrize("what", _ALL_WHAT)
    def test_no_warning_at_training_rows(self, expo_fit: ExpoFit, what: str) -> None:
        """Inside the training hull h(a|x) ≥ 0 up to solver tolerance."""
        model, x = expo_fit
        X_train = x[:, None]
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model.predict(_y_for(what, x.size), X_new=X_train, what=what)
        assert _support_warnings(rec) == []

    @pytest.mark.parametrize("x_bad", [2.0, 3.0, 10.0])
    def test_values_returned_unclamped(self, expo_fit: ExpoFit, x_bad: float) -> None:
        """At y = a the CDF is 1 − exp(−h(a|x)) < 0: raw, not clamped to 0."""
        model, _ = expo_fit
        a = _SUPPORT[0]
        y_new = np.array([a, a])
        X_new = np.array([[0.5], [x_bad]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cdf = model.predict(y_new, X_new=X_new, what="distribution")
            trafo = model.predict(y_new, X_new=X_new, what="trafo")
        assert trafo[1] < 0.0
        assert cdf[1] < 0.0
        np.testing.assert_allclose(cdf, -np.expm1(-trafo), rtol=1e-12)
        # The in-hull row is unaffected by sharing a call with a bad row.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cdf_ok = model.predict(y_new[:1], X_new=X_new[:1], what="distribution")
        np.testing.assert_array_equal(cdf[:1], cdf_ok)

    @pytest.mark.parametrize("what", ["distribution", "quantile"])
    def test_offset_enters_h_at_lower_end(self, expo_fit: ExpoFit, what: str) -> None:
        """A negative offset can push an in-hull row off the support."""
        model, _ = expo_fit
        X_new = np.array([[0.5], [0.5]])
        offset = np.array([0.0, -5.0])
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model.predict(_y_for(what, 2), X_new=X_new, what=what, offset_new=offset)
        msgs = _support_warnings(rec)
        assert len(msgs) == 1
        assert "[1]" in msgs[0]

    def test_many_rows_are_truncated(self, expo_fit: ExpoFit) -> None:
        model, _ = expo_fit
        X_new = np.full((25, 1), 3.0)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model.predict(np.full(25, 1.0), X_new=X_new)
        (msg,) = _support_warnings(rec)
        assert "25 row(s)" in msg
        assert "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, …]" in msg

    def test_empty_input_does_not_warn(self, expo_fit: ExpoFit) -> None:
        model, _ = expo_fit
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = model.predict(np.empty(0), X_new=np.empty((0, 1)))
        assert out.shape == (0,)

    @pytest.mark.parametrize("dist", ["normal", "logistic", "min_extreme_value"])
    def test_other_links_do_not_check(self, dist: str) -> None:
        """The support check is specific to the exponential link."""
        rng = np.random.default_rng(3)
        x = rng.uniform(0.0, 1.0, 80)
        y = np.clip(1.0 + x + rng.normal(0.0, 0.3, 80), 0.05, 2.95)
        model = ConditionalTransformationModel(
            BernsteinBasis(order=3, support=_SUPPORT), base_distribution=dist
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y, X=x[:, None])
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model.predict(np.full(2, 1.0), X_new=np.array([[-50.0], [50.0]]))
        assert _support_warnings(rec) == []

    @pytest.mark.parametrize("seed", [31, 2])
    def test_active_support_rows_round_off_does_not_warn(self, seed: int) -> None:
        """Exact data pins every h(a|x_i) to 0; solver round-off is ~−1e-9.

        On these fixtures most training rows land at ``h(a|x_i) ∈ [−7e-9, 0)``
        because all support rows are active (cf. ``TestExponentialWithCovariates``
        in ``test_model.py``).  A strict ``< 0`` check would warn on in-sample
        prediction; the ``1e-7`` feasibility tolerance must absorb it.
        """
        rng = np.random.default_rng(seed)
        y = rng.uniform(0.05, 0.95, 100)
        X = rng.normal(0.0, 0.5, (100, 2))
        model = ConditionalTransformationModel(
            BernsteinBasis(order=3, support=(0.0, 1.0)),
            base_distribution="exponential",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y, X=X)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model.predict(y, X_new=X)
        assert _support_warnings(rec) == []


def _saturation_warnings(rec: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in rec if "predict(what='quantile')" in str(w.message)]


class TestQuantileSupportAtom:
    """For p < F(a|x) = 1 − exp(−h(a|x)) the quantile is exactly a (ADR 0005)."""

    @pytest.mark.parametrize("p", [0.01, 0.1, 0.3])
    def test_atom_quantile_is_lower_end_with_atom_wording(
        self, expo_fit: ExpoFit, p: float
    ) -> None:
        model, _ = expo_fit
        a = _SUPPORT[0]
        X_new = np.array([[0.5]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_a = model.predict(np.array([a]), X_new=X_new, what="distribution")[0]
        assert p < f_a  # fixture sanity: F(a|0.5) ≈ 0.46
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            q = model.predict(np.array([p]), X_new=X_new, what="quantile")
        np.testing.assert_allclose(q, a, atol=1e-8)
        (msg,) = _saturation_warnings(rec)
        assert "support atom" in msg
        assert "F(a|x) = 1 − exp(−h(a|x))" in msg
        assert "asymptotic limit" not in msg
        hit = next(w for w in rec if "support atom" in str(w.message))
        assert hit.filename == __file__

    def test_quantile_above_atom_is_not_saturated(self, expo_fit: ExpoFit) -> None:
        model, _ = expo_fit
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            q = model.predict(np.array([0.6]), X_new=np.array([[0.5]]), what="quantile")
        assert _saturation_warnings(rec) == []
        assert _SUPPORT[0] < q[0] < _SUPPORT[1]

    def test_upper_saturation_keeps_endpoint_wording(self, expo_fit: ExpoFit) -> None:
        """Clipping at θ_b[-1] is still a basis-endpoint artefact."""
        model, _ = expo_fit
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model.predict(
                np.array([1.0 - 1e-12]), X_new=np.array([[0.5]]), what="quantile"
            )
        (msg,) = _saturation_warnings(rec)
        assert "support atom" not in msg
        assert "asymptotic limit" in msg

    def test_mixed_rows_report_both_sides(self, expo_fit: ExpoFit) -> None:
        model, _ = expo_fit
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model.predict(
                np.array([0.1, 1.0 - 1e-12]),
                X_new=np.array([[0.5], [0.5]]),
                what="quantile",
            )
        msgs = _saturation_warnings(rec)
        assert len(msgs) == 2
        assert any("support atom" in m for m in msgs)
        assert any("asymptotic limit" in m for m in msgs)

    def test_non_exponential_lower_saturation_keeps_wording(self) -> None:
        rng = np.random.default_rng(3)
        y = rng.uniform(0.5, 2.5, 80)
        model = ConditionalTransformationModel(
            BernsteinBasis(order=3, support=_SUPPORT)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(y)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model.predict(np.array([1e-12]), what="quantile")
        (msg,) = _saturation_warnings(rec)
        assert "support atom" not in msg
        assert "asymptotic limit" in msg


def _atom_hits(rec: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    """Recorded support-atom quantile warnings (issue #157 wording)."""
    return [w for w in rec if "support atom" in str(w.message)]


class TestSimulateSupportAtom:
    """``simulate()`` hits the atom at ``a`` by design (issue #159).

    Draws ``u ~ Uniform(0, 1)`` with ``u < F(a|x)`` invert to the support atom
    ``a``; that is correct sampling, so ``simulate()`` must not report it.  The
    genuine warnings (``h(a|x) < 0`` and the upper-end bracket clip) still pass
    through, attributed to the caller of ``simulate()``.
    """

    _N = 20_000

    @pytest.mark.parametrize("x0", [0.0, 0.5, 1.0])
    def test_atom_draws_do_not_warn(self, expo_fit: ExpoFit, x0: float) -> None:
        model, _ = expo_fit
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model.simulate(200, X=np.full((200, 1), x0), random_state=0)
        assert _atom_hits(rec) == []

    @pytest.mark.parametrize("x0", [0.0, 0.5, 1.0])
    def test_atom_frequency_matches_cdf_at_lower_end(
        self, expo_fit: ExpoFit, x0: float
    ) -> None:
        """Draws equal ``a`` with probability ``F(a|x) = 1 − exp(−h(a|x))``."""
        model, _ = expo_fit
        a = _SUPPORT[0]
        X = np.full((self._N, 1), x0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_a = model.predict(np.array([a]), X_new=X[:1], what="distribution")[0]
            draws = model.simulate(self._N, X=X, random_state=7)
        assert f_a > 0.1  # fixture sanity: a non-trivial atom
        freq = np.mean(np.abs(draws - a) < 1e-8)
        se = np.sqrt(f_a * (1.0 - f_a) / self._N)
        assert abs(freq - f_a) < 5.0 * se

    @pytest.mark.parametrize("x_bad", [2.0, 3.0])
    def test_off_support_warning_passes_through(
        self, expo_fit: ExpoFit, x_bad: float
    ) -> None:
        model, _ = expo_fit
        X = np.array([[0.5], [x_bad], [0.5]])
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model.simulate(3, X=X, random_state=0)
        hits = [w for w in rec if "h(a|x) < 0" in str(w.message)]
        assert len(hits) == 1
        assert "[1]" in str(hits[0].message)
        assert issubclass(hits[0].category, UserWarning)
        assert hits[0].filename == __file__

    @pytest.mark.parametrize("x0", [0.0, 0.5])
    def test_upper_saturation_warning_passes_through(
        self, expo_fit: ExpoFit, x0: float
    ) -> None:
        """``1 − F(b|x) ≈ 5e-4``: a large seeded sample clips at ``θ_b[-1]``."""
        model, _ = expo_fit
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model.simulate(self._N, X=np.full((self._N, 1), x0), random_state=3)
        hits = [w for w in rec if "asymptotic limit" in str(w.message)]
        assert len(hits) == 1
        assert hits[0].filename == __file__
        assert _atom_hits(rec) == []

    def test_empty_draw_is_silent(self, expo_fit: ExpoFit) -> None:
        model, _ = expo_fit
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            draws = model.simulate(0, X=np.empty((0, 1)), random_state=0)
        assert draws.shape == (0,)

    def test_atom_draws_pass_error_filter(self, expo_fit: ExpoFit) -> None:
        """A user ``error`` filter must not turn routine atom draws into errors."""
        model, _ = expo_fit
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            draws = model.simulate(50, X=np.full((50, 1), 0.5), random_state=0)
        assert np.any(np.abs(draws - _SUPPORT[0]) < 1e-8)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_predict_on_same_draws_still_warns(
        self, expo_fit: ExpoFit, seed: int
    ) -> None:
        """An explicit quantile ``predict`` keeps the #157 atom warning."""
        model, _ = expo_fit
        X = np.full((50, 1), 0.5)
        u = np.random.default_rng(seed).uniform(size=50)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model.predict(u, X_new=X, what="quantile")
        (hit,) = _atom_hits(rec)
        assert hit.filename == __file__
        assert "point(s)" in str(hit.message)
