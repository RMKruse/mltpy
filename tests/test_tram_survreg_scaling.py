"""R parity for ``Survreg(scaling=X_s)`` — issue #76.

End-to-end coverage of the
``tram::Survreg(Surv(y, event) ~ x_d | x_s, data, scale = ~x_s)``
convenience surface across all three parametric families
(``"weibull"``, ``"lognormal"``, ``"loglogistic"``).  The kwarg must
thread through to the scaled-baseline likelihood (#71) and the
scaled-predict path (#72) without callers having to reach for
:class:`pymlt.MLT` directly.

Reference data lives in ``reference/scaling_survreg_<dist>_*`` and is
produced by ``reference/generate_reference.R``.

Sign conventions:

* ``tram::Survreg`` uses ``negative = TRUE`` (so ``h − X_d·β``).  pymlt
  parametrises ``h + X_d·β``; the β block compares against ``-β_R``.
* γ is sign-aligned across the two parameterisations
  (ADR 0002, Decision 5).

Notes
-----
``tram::Survreg`` always fits a strictly affine, two-parameter baseline
on ``log(t)`` (a ``polynomial_basis`` with intercept + ``log(t)``).
pymlt's ``Survreg`` uses ``LogBernsteinBasis`` on the log-time scale;
with ``order = 1`` the basis is also affine in ``log(t)`` and produces
an equivalent (linearly reparameterised) baseline at the MLE.  The raw
``theta_b`` block is therefore in different coordinates between the two
parameterisations and is not asserted element-wise; log-likelihood, β
and γ are parameterisation-invariant and *are* compared at the issue's
rtol=1e-6 budget.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pymlt.tram import Survreg
from pymlt.variables import CensoredData

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"

_DISTRIBUTIONS = ("weibull", "lognormal", "loglogistic")


def _files_for(dist: str) -> list[str]:
    return [
        f"scaling_survreg_{dist}_y.txt",
        f"scaling_survreg_{dist}_event.txt",
        f"scaling_survreg_{dist}_x_d.txt",
        f"scaling_survreg_{dist}_x_s.txt",
        f"scaling_survreg_{dist}_support.txt",
        f"scaling_survreg_{dist}_theta.txt",
        f"scaling_survreg_{dist}_loglik.txt",
    ]


def _load(name: str) -> np.ndarray:
    return np.loadtxt(REF_DIR / name)


def _maybe_skip(files: list[str]) -> None:
    if not all((REF_DIR / f).exists() for f in files):
        pytest.skip(
            "scaling_survreg_* reference files not yet generated — "
            "run Rscript reference/generate_reference.R"
        )


@pytest.fixture(scope="module", params=_DISTRIBUTIONS)
def survreg_scaling_ref(request: pytest.FixtureRequest) -> dict:
    dist = request.param
    _maybe_skip(_files_for(dist))
    y = _load(f"scaling_survreg_{dist}_y.txt")
    event = _load(f"scaling_survreg_{dist}_event.txt").astype(bool)
    x_d = _load(f"scaling_survreg_{dist}_x_d.txt").reshape(-1, 1)
    x_s = _load(f"scaling_survreg_{dist}_x_s.txt").reshape(-1, 1)
    support = tuple(_load(f"scaling_survreg_{dist}_support.txt"))
    theta_full = _load(f"scaling_survreg_{dist}_theta.txt")
    ll = float(_load(f"scaling_survreg_{dist}_loglik.txt"))
    q_d = x_d.shape[1]
    q_s = x_s.shape[1]
    # tram::Survreg always uses an order=1 affine baseline on log(t).
    p = len(theta_full) - q_d - q_s
    assert p == 2
    return {
        "distribution": dist,
        "y": y,
        "event": event,
        "x_d": x_d,
        "x_s": x_s,
        "support": (float(support[0]), float(support[1])),
        "p": p,
        "q_d": q_d,
        "q_s": q_s,
        "beta_r": theta_full[p : p + q_d],
        "gamma_r": theta_full[p + q_d :],
        "log_likelihood": ll,
    }


@pytest.fixture(scope="module")
def fitted_survreg(survreg_scaling_ref: dict) -> Survreg:
    model = Survreg(
        support=survreg_scaling_ref["support"],
        distribution=survreg_scaling_ref["distribution"],
        order=1,
        scaling=survreg_scaling_ref["x_s"],
    )
    cd = CensoredData.right_censored(
        survreg_scaling_ref["y"], ~survreg_scaling_ref["event"]
    )
    model.fit(cd, X=survreg_scaling_ref["x_d"])
    return model


def test_survreg_accepts_scaling_kwarg(survreg_scaling_ref: dict) -> None:
    """Survreg(scaling=X_s).fit(cd, X=X_d) runs without raising."""
    model = Survreg(
        support=survreg_scaling_ref["support"],
        distribution=survreg_scaling_ref["distribution"],
        order=1,
        scaling=survreg_scaling_ref["x_s"],
    )
    cd = CensoredData.right_censored(
        survreg_scaling_ref["y"], ~survreg_scaling_ref["event"]
    )
    model.fit(cd, X=survreg_scaling_ref["x_d"])
    assert model.is_fitted_
    assert model.theta_ is not None
    assert model.theta_.shape == (
        survreg_scaling_ref["p"]
        + survreg_scaling_ref["q_d"]
        + survreg_scaling_ref["q_s"],
    )


def test_survreg_beta_gamma_match_R(
    fitted_survreg: Survreg, survreg_scaling_ref: dict
) -> None:
    """Fitted β (sign-flipped) and γ match R at rtol=1e-5/atol=1e-7."""
    p = survreg_scaling_ref["p"]
    q_d = survreg_scaling_ref["q_d"]
    assert fitted_survreg.theta_ is not None
    beta = fitted_survreg.theta_[p : p + q_d]
    gamma = fitted_survreg.theta_[p + q_d :]

    # pymlt parametrises h + Xβ; tram::Survreg uses h − Xβ_R.  Sign flip on β.
    np.testing.assert_allclose(
        beta, -survreg_scaling_ref["beta_r"], rtol=1e-5, atol=1e-6
    )
    # γ is sign-aligned across the two parameterisations.
    np.testing.assert_allclose(
        gamma, survreg_scaling_ref["gamma_r"], rtol=1e-5, atol=1e-6
    )


def test_survreg_log_likelihood_matches_R(
    fitted_survreg: Survreg, survreg_scaling_ref: dict
) -> None:
    """log-lik at the fitted parameters matches R at rtol=1e-6."""
    assert fitted_survreg.result_ is not None
    np.testing.assert_allclose(
        fitted_survreg.result_.log_likelihood,
        survreg_scaling_ref["log_likelihood"],
        rtol=1e-6,
        atol=1e-7,
    )


def test_survreg_gamma_property(
    fitted_survreg: Survreg, survreg_scaling_ref: dict
) -> None:
    """``gamma_`` returns the γ block (sign-aligned with R)."""
    np.testing.assert_allclose(
        fitted_survreg.gamma_, survreg_scaling_ref["gamma_r"], rtol=1e-5, atol=1e-6
    )


def test_survreg_feature_names_scaling_default(fitted_survreg: Survreg) -> None:
    """``feature_names_scaling_`` falls back to ``X1, X2, ...`` for ndarray input."""
    assert fitted_survreg.feature_names_scaling_ == ["X1"]


def test_survreg_feature_names_scaling_from_dataframe(
    survreg_scaling_ref: dict,
) -> None:
    """A pandas DataFrame's column names propagate to ``feature_names_scaling_``."""
    pd = pytest.importorskip("pandas")
    x_s_df = pd.DataFrame(survreg_scaling_ref["x_s"], columns=["het_x"])
    model = Survreg(
        support=survreg_scaling_ref["support"],
        distribution=survreg_scaling_ref["distribution"],
        order=1,
        scaling=x_s_df,
    )
    cd = CensoredData.right_censored(
        survreg_scaling_ref["y"], ~survreg_scaling_ref["event"]
    )
    model.fit(cd, X=survreg_scaling_ref["x_d"])
    assert model.feature_names_scaling_ == ["het_x"]


def test_survreg_gamma_raises_without_scaling(survreg_scaling_ref: dict) -> None:
    """Accessing ``gamma_`` on a non-scaling fit raises ValueError."""
    model = Survreg(
        support=survreg_scaling_ref["support"],
        distribution=survreg_scaling_ref["distribution"],
        order=1,
    )
    cd = CensoredData.right_censored(
        survreg_scaling_ref["y"], ~survreg_scaling_ref["event"]
    )
    model.fit(cd, X=survreg_scaling_ref["x_d"])
    with pytest.raises(ValueError, match="scaling="):
        _ = model.gamma_
    with pytest.raises(ValueError, match="scaling="):
        _ = model.feature_names_scaling_


def test_survreg_survival_matches_predict(
    fitted_survreg: Survreg, survreg_scaling_ref: dict
) -> None:
    """``survival(..., X_scale=...)`` matches ``1 - predict(what='distribution')``."""
    a, b = survreg_scaling_ref["support"]
    x_d_new = np.zeros((2, survreg_scaling_ref["q_d"]))
    x_s_new = np.array([[-1.0], [1.0]])
    t_grid = np.linspace(a + 0.1 * (b - a), a + 0.9 * (b - a), 4)

    m = x_d_new.shape[0]
    k = t_grid.shape[0]
    y_rep = np.repeat(t_grid[:, None], m, axis=1).ravel()
    X_d_rep = np.tile(x_d_new, (k, 1))
    X_s_rep = np.tile(x_s_new, (k, 1))

    surv = fitted_survreg.survival(y_rep, X=X_d_rep, X_scale=X_s_rep)
    cdf = fitted_survreg.predict(
        y_rep, X_new=X_d_rep, X_scale_new=X_s_rep, what="distribution"
    )
    np.testing.assert_allclose(surv, 1.0 - cdf, rtol=1e-10, atol=1e-12)


def test_survreg_survival_requires_X_scale(
    fitted_survreg: Survreg, survreg_scaling_ref: dict
) -> None:
    """``survival()`` errors when ``X_scale`` is omitted on a scaling fit."""
    a, b = survreg_scaling_ref["support"]
    t_grid = np.linspace(a + 0.1 * (b - a), a + 0.9 * (b - a), 3)
    x_d_new = np.zeros((3, survreg_scaling_ref["q_d"]))
    with pytest.raises(ValueError, match="X_scale_new must be"):
        fitted_survreg.survival(t_grid, X=x_d_new)
