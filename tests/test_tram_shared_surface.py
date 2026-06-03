"""Behaviour tests for the shared _TramModel / _SurvivalMixin surface.

These pin the public behaviour that the tram convenience subclasses share
after hoisting the previously copy-pasted gamma_, feature_names_scaling_,
fitted_transformation, survival and hazard members. They exercise the
classes through their public API only, so they survive the internal
move from per-subclass copies to the shared base/mixin.
"""

from __future__ import annotations

import numpy as np
import pytest

from mltpy.tram import Coxph, Lehmann
from mltpy.variables import CensoredData


def _survival_data(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    times = rng.exponential(scale=0.4, size=n).clip(0.01, 0.99)
    censored = rng.random(n) < 0.3
    return times, censored


# ---------------------------------------------------------------------------
# fitted_transformation is available on every shift-basis tram model
# ---------------------------------------------------------------------------


class TestFittedTransformationShared:
    def test_coxph_exposes_fitted_transformation(self):
        times, censored = _survival_data()
        cd = CensoredData.right_censored(times, censored)
        model = Coxph(support=(0.01, 1.0)).fit(cd)
        grid = np.linspace(0.05, 0.95, 30)
        h = model.fitted_transformation(grid)
        assert h.shape == (30,)
        # baseline transformation is monotone non-decreasing in y
        assert np.all(np.diff(h) >= -1e-8)


# ---------------------------------------------------------------------------
# gamma_ is defined on every _TramModel; non-scaling models raise ValueError
# ---------------------------------------------------------------------------


class TestGammaShared:
    def test_lehmann_gamma_raises_value_error_when_no_scaling(self):
        times, censored = _survival_data()
        cd = CensoredData.right_censored(times, censored)
        model = Lehmann(support=(0.01, 1.0)).fit(cd)
        with pytest.raises(ValueError, match="scaling="):
            _ = model.gamma_

    def test_lehmann_feature_names_scaling_raises_when_no_scaling(self):
        times, censored = _survival_data()
        cd = CensoredData.right_censored(times, censored)
        model = Lehmann(support=(0.01, 1.0)).fit(cd)
        with pytest.raises(ValueError, match="scaling="):
            _ = model.feature_names_scaling_


# ---------------------------------------------------------------------------
# survival()/hazard() share one X_scale-aware signature (_SurvivalMixin)
# ---------------------------------------------------------------------------


class TestSurvivalMixinSignature:
    def test_lehmann_survival_accepts_and_rejects_x_scale(self):
        times, censored = _survival_data()
        cd = CensoredData.right_censored(times, censored)
        model = Lehmann(support=(0.01, 1.0)).fit(cd)
        grid = np.linspace(0.05, 0.95, 10)
        # X_scale= is now an accepted keyword (was an unexpected-kwarg TypeError)
        # and, since Lehmann has no scaling, a non-None value is rejected.
        with pytest.raises(ValueError, match="scaling="):
            model.survival(grid, X_scale=np.ones((10, 1)))

    def test_lehmann_survival_default_still_works(self):
        times, censored = _survival_data()
        cd = CensoredData.right_censored(times, censored)
        model = Lehmann(support=(0.01, 1.0)).fit(cd)
        grid = np.linspace(0.05, 0.95, 10)
        s = model.survival(grid)
        assert s.shape == (10,)
        assert np.all((s >= 0.0) & (s <= 1.0))
