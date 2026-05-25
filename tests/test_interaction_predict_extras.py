"""R-parity and self-consistency tests for the non-CDF/PDF prediction surface
on interacting tensor-product CTMs — issue #67.

Companion to ``test_interaction_r_parity.py``: that file checks CDF, PDF, θ,
log-likelihood and monotonicity; this file covers the remaining ``predict``
targets (``quantile``, ``survivor``, ``hazard``), ``simulate``, and ``plot``
for ``InteractionBasis`` models.

R coefficient-order note: same fixture as #65 — ``x`` is continuous on
``[0, 1]``, ``y`` is the conditional response, and the fitted CTM uses a
``BernsteinBasis(order=2) ⊠ BernsteinBasis(order=2)`` design.  See
``test_interaction_r_parity.py`` for the coefficient-reshape convention.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from scipy.stats import kstest

from mltpy import (
    ConditionalTransformationModel,
    InteractionBasis,
    OptimizerConfig,
)
from mltpy.basis import BernsteinBasis

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"


def _load_extras_reference(label: str) -> dict[str, np.ndarray]:
    """Load all reference fixtures needed by the extras tests."""
    base = REF_DIR
    required = [
        base / "interaction_bs_bs_y_train.txt",
        base / "interaction_bs_bs_x_train.txt",
        base / "interaction_bs_bs_y_support.txt",
        base / "interaction_bs_bs_y_grid.txt",
        base / "interaction_bs_bs_x_grid.txt",
        base / "interaction_bs_bs_probs.txt",
        base / f"interaction_bs_bs_{label}_survivor.txt",
        base / f"interaction_bs_bs_{label}_hazard.txt",
        base / f"interaction_bs_bs_{label}_quantile.txt",
    ]
    if not all(p.exists() for p in required):
        pytest.skip(
            f"interaction_bs_bs_{label}_* extras fixtures not generated — run "
            "Rscript reference/generate_reference.R"
        )
    return {
        "y_train": np.loadtxt(required[0]),
        "x_train": np.loadtxt(required[1]),
        "support": tuple(np.loadtxt(required[2])),
        "y_grid": np.loadtxt(required[3]),
        "x_grid": np.loadtxt(required[4]),
        "probs": np.loadtxt(required[5]),
        "surv_R": np.loadtxt(required[6]),
        "haz_R": np.loadtxt(required[7]),
        # R writes a (len(probs), len(x_grid)) matrix column-major: for each
        # x in x_grid, all probs vary fastest.
        "quant_R": np.loadtxt(required[8]),
    }


def _fit_mltpy(
    label: str,
    data: dict[str, np.ndarray],
    solver: str = "auglag",
) -> ConditionalTransformationModel:
    """Fit Bernstein-y × Bernstein-x interaction CTM at ``(p, q) = (3, 3)``."""
    y_basis = BernsteinBasis(order=2, support=tuple(data["support"]))
    x_basis = BernsteinBasis(order=2, support=(0.0, 1.0))
    ib = InteractionBasis(y_basis=y_basis, x_basis=x_basis)
    base_distribution = "normal" if label == "normal" else "logistic"
    model = ConditionalTransformationModel(
        basis=ib,
        base_distribution=base_distribution,
        optimizer_config=OptimizerConfig(solver=solver, random_state=0),  # type: ignore[arg-type]
    )
    model.fit(data["y_train"], data["x_train"])
    return model


def _flat_yx(y_grid: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R's ``expand.grid(y=y_grid, x=x_grid)``: y varies fastest."""
    y_mesh, x_mesh = np.meshgrid(y_grid, x_grid, indexing="xy")
    return y_mesh.ravel(), x_mesh.ravel()


# ---------------------------------------------------------------------------
# 1. Survivor and hazard parity with R on the held-out (y, x) lattice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["normal", "logistic"])
def test_survivor_matches_R(label: str) -> None:
    data = _load_extras_reference(label)
    model = _fit_mltpy(label, data)
    y_flat, x_flat = _flat_yx(data["y_grid"], data["x_grid"])
    surv = model.predict(y_flat, X_new=x_flat[:, None], what="survivor")
    # Same tolerance as the CDF/PDF parity test in #65 — see
    # test_interaction_r_parity.py for the rationale (auglag KKT divergence).
    np.testing.assert_allclose(surv, data["surv_R"], rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize("label", ["normal", "logistic"])
def test_hazard_matches_R(label: str) -> None:
    data = _load_extras_reference(label)
    model = _fit_mltpy(label, data)
    y_flat, x_flat = _flat_yx(data["y_grid"], data["x_grid"])
    haz = model.predict(y_flat, X_new=x_flat[:, None], what="hazard")
    np.testing.assert_allclose(haz, data["haz_R"], rtol=2e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Quantile parity with R, row-by-row over x_grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["normal", "logistic"])
def test_quantile_matches_R(label: str) -> None:
    data = _load_extras_reference(label)
    model = _fit_mltpy(label, data)

    probs = data["probs"]
    x_grid = data["x_grid"]
    n_p = len(probs)
    n_x = len(x_grid)

    # Build per-row (prob, x) pairs.  predict() returns one quantile per row.
    probs_flat = np.tile(probs, n_x)
    x_flat = np.repeat(x_grid, n_p)

    q_mltpy = model.predict(probs_flat, X_new=x_flat[:, None], what="quantile")
    # rtol slightly looser than CDF parity because the quantile is the
    # inverse of h(·|x); a 2e-4 error in θ amplifies near the tails.
    np.testing.assert_allclose(q_mltpy, data["quant_R"], rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. Quantile self-consistency: F(Q(p|x)|x) == p
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["normal", "logistic"])
def test_quantile_inverts_cdf(label: str) -> None:
    """Quantile is the right inverse of the CDF at the row's x."""
    data = _load_extras_reference(label)
    model = _fit_mltpy(label, data)

    probs = np.array([0.05, 0.20, 0.50, 0.80, 0.95])
    x_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    n_p = len(probs)
    n_x = len(x_vals)
    probs_flat = np.tile(probs, n_x)
    x_flat = np.repeat(x_vals, n_p)

    q = model.predict(probs_flat, X_new=x_flat[:, None], what="quantile")
    cdf_at_q = model.predict(q, X_new=x_flat[:, None], what="distribution")
    np.testing.assert_allclose(cdf_at_q, probs_flat, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. simulate() draws are correctly distributed
# ---------------------------------------------------------------------------


# simulate() draws 10k uniforms; a handful land in the extreme tails where the
# quantile leaves the finite basis support and is clipped. Expected and benign
# here — the KS test below confirms the draws still match the analytical CDF.
@pytest.mark.filterwarnings(r"ignore:predict\(what='quantile'\)")
@pytest.mark.parametrize("label", ["normal", "logistic"])
def test_simulate_ks_at_fixed_x(label: str) -> None:
    """``simulate(n, X=x0·ones)`` empirical CDF matches the analytical CDF."""
    data = _load_extras_reference(label)
    model = _fit_mltpy(label, data)

    n = 10_000
    x_fixed = 0.4
    X = np.full((n, 1), x_fixed, dtype=float)
    draws = model.simulate(n, X=X, random_state=20260521)

    # Build the analytical CDF at x_fixed and run a one-sample KS test.
    def analytical_cdf(y: np.ndarray) -> np.ndarray:
        y_ravel = np.atleast_1d(y).astype(float)
        Xq = np.full((y_ravel.size, 1), x_fixed, dtype=float)
        return model.predict(y_ravel, X_new=Xq, what="distribution")

    stat, p_value = kstest(draws, analytical_cdf)
    assert p_value > 0.01, (
        f"KS test rejected: stat={stat:.4f}, p={p_value:.4f}; "
        "simulate() draws do not match the analytical CDF."
    )


# ---------------------------------------------------------------------------
# 5. plot() works on an interaction model with an X argument
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["normal"])
def test_plot_interaction_with_X(label: str) -> None:
    """Smoke test: plot() with X draws one CDF curve per row of X."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _load_extras_reference(label)
    model = _fit_mltpy(label, data)

    a, b = data["support"]
    y_curve = np.linspace(a + 1e-3, b - 1e-3, 50)
    X = np.array([[0.1], [0.5], [0.9]])

    fig, axes = plt.subplots(1, 2)
    out = model.plot(y_curve, X=X, ax=(axes[0], axes[1]))
    assert out is not None
    # One Line2D per row of X on each panel.
    assert len(axes[0].get_lines()) == X.shape[0]
    assert len(axes[1].get_lines()) == X.shape[0]
    plt.close(fig)


def test_plot_interaction_requires_X() -> None:
    """plot() on an interaction model raises a clear error when X is None."""
    data = _load_extras_reference("normal")
    model = _fit_mltpy("normal", data)
    a, b = data["support"]
    y_curve = np.linspace(a + 1e-3, b - 1e-3, 20)
    with pytest.raises(ValueError, match="X"):
        model.plot(y_curve)
