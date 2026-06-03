"""Scaled-likelihood vcov / sandwich SE / Wald tests — issue #77.

End-to-end coverage of the inference surface (Hessian, per-observation
scores, ``vcov``, ``sandwich_vcov``, ``standard_errors``, ``wald_test``)
for the heteroskedastic CTM path of ADR 0002.  The shift-only path
already has full inference coverage in ``test_vcov.py``; this file
adds the ``[theta_b | beta | gamma]`` block layout, including the
``H_θγ`` / ``H_βγ`` cross-blocks and the new ``γ`` columns of
``score_matrix``.

R parity fixtures live in ``reference/scaling_vcov_*`` and are produced
by ``reference/generate_reference.R`` (``tram::vcov(...)`` and
``sandwich::vcovHC(...)``).  See ADR 0002 Decision 5 for the sign
convention: mltpy's β block flips for BoxCox (``negative = TRUE``) but
γ is sign-aligned with R ``tram::*``'s scaling block on every model.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import norm

from mltpy import (
    MLT,
    CensoredData,
    CensoringType,
    hessian,
    negative_log_likelihood,
    score_matrix,
)
from mltpy.basis import BernsteinBasis
from mltpy.likelihood import _get_dist
from mltpy.tram import BoxCox, Colr, Coxph

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"


def _toy_scaled_problem(
    n: int = 60, p: int = 5, q_d: int = 2, q_s: int = 1, seed: int = 0
) -> dict:
    """Build a tiny exact-data scaled problem for self-consistency tests."""
    rng = np.random.default_rng(seed)
    y = np.abs(rng.normal(1.5, 0.5, n)) + 0.1
    X = rng.normal(0.0, 1.0, (n, q_d))
    X_s = rng.normal(0.0, 1.0, (n, q_s))
    theta = np.concatenate(
        [
            np.linspace(-1.2, 1.4, p),  # theta_b (monotone)
            np.array([0.25, -0.15][:q_d]),  # beta
            np.array([0.30] * q_s),  # gamma
        ]
    )
    basis = BernsteinBasis(
        order=p - 1, support=(float(y.min() - 0.1), float(y.max() + 0.1))
    )
    return {
        "y": y,
        "X": X,
        "X_s": X_s,
        "theta": theta,
        "basis": basis,
        "p": p,
        "q_d": q_d,
        "q_s": q_s,
    }


def _make_right_censored(n: int, rng: np.random.Generator) -> CensoredData:
    y = np.abs(rng.normal(1.5, 0.5, n)) + 0.1
    event = rng.random(n) > 0.3
    return CensoredData.right_censored(y, censored=~event)


def _make_left_censored(n: int, rng: np.random.Generator) -> CensoredData:
    y = np.abs(rng.normal(1.5, 0.5, n)) + 0.1
    event = rng.random(n) > 0.3
    upper = y
    lower = np.where(event, y, -np.inf)
    exact = np.where(event, y, np.nan)
    return CensoredData(lower=lower, upper=upper, exact=exact)


def _make_interval(n: int, rng: np.random.Generator) -> CensoredData:
    y = np.abs(rng.normal(1.5, 0.5, n)) + 0.1
    half = 0.3
    return CensoredData(lower=y - half, upper=y + half, exact=np.full(n, np.nan))


def _support_from_cd(cd: CensoredData, pad: float = 0.1) -> tuple[float, float]:
    parts = [
        cd.lower[np.isfinite(cd.lower)],
        cd.upper[np.isfinite(cd.upper)],
        cd.exact[np.isfinite(cd.exact)],
    ]
    vals = np.concatenate([p for p in parts if p.size])
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo
    return lo - pad * span, hi + pad * span


def test_score_matrix_scaled_exact_normal_returns_full_gamma_columns():
    """score_matrix on scaled fit: shape (n, p+q_d+q_s), col-sum == -grad(NLL)."""
    prob = _toy_scaled_problem()
    scores = score_matrix(
        prob["theta"],
        prob["basis"],
        prob["y"],
        prob["X"],
        CensoringType.NONE,
        base_distribution="normal",
        scaling=prob["X_s"],
    )
    assert scores.shape == (len(prob["y"]), prob["p"] + prob["q_d"] + prob["q_s"])

    _, grad_nll = negative_log_likelihood(
        prob["theta"],
        prob["basis"],
        prob["y"],
        prob["X"],
        CensoringType.NONE,
        gradient=True,
        base_distribution="normal",
        scaling=prob["X_s"],
    )
    np.testing.assert_allclose(scores.sum(axis=0), -grad_nll, atol=1e-10)


def test_hessian_scaled_exact_normal_matches_finite_differences():
    """Analytical Hessian on scaled fit matches FD of analytical gradient.

    Exercises every block of the (p+q_d+q_s)² information matrix — including
    the new H_θγ, H_βγ, and H_γγ cross-blocks — for the exact / normal path.
    """
    prob = _toy_scaled_problem()

    def grad_fn(t: np.ndarray) -> np.ndarray:
        _, g = negative_log_likelihood(
            t,
            prob["basis"],
            prob["y"],
            prob["X"],
            CensoringType.NONE,
            gradient=True,
            base_distribution="normal",
            scaling=prob["X_s"],
        )
        return g

    H_ana = hessian(
        prob["theta"],
        prob["basis"],
        prob["y"],
        prob["X"],
        CensoringType.NONE,
        base_distribution="normal",
        scaling=prob["X_s"],
    )
    H_fd = np.stack(
        [
            approx_fprime(prob["theta"], lambda t, i=i: grad_fn(t)[i], 1e-5)
            for i in range(prob["theta"].size)
        ]
    )
    H_fd = 0.5 * (H_fd + H_fd.T)

    k = prob["theta"].size
    assert H_ana.shape == (k, k)
    np.testing.assert_allclose(H_ana, H_ana.T, atol=1e-10)
    np.testing.assert_allclose(H_ana, H_fd, atol=5e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Self-consistency across every (censoring, base_distribution) combination
# ---------------------------------------------------------------------------


_DISTS = ("normal", "logistic", "min_extreme_value", "max_extreme_value")


@pytest.mark.parametrize("base_distribution", _DISTS)
@pytest.mark.parametrize(
    "censoring,builder",
    [
        (CensoringType.NONE, None),
        (CensoringType.RIGHT, _make_right_censored),
        (CensoringType.LEFT, _make_left_censored),
        (CensoringType.INTERVAL, _make_interval),
    ],
)
def test_scaled_score_sums_to_minus_nll_gradient(
    base_distribution: str, censoring: CensoringType, builder
) -> None:
    """score_matrix(scaling=...).sum(0) == -grad(NLL) on every dispatch path."""
    rng = np.random.default_rng(123)
    n = 50
    p, q_d, q_s = 5, 2, 1
    theta = np.concatenate([np.linspace(-1.2, 1.4, p), [0.25, -0.15], [0.30]])
    X = rng.normal(0.0, 1.0, (n, q_d))
    X_s = rng.normal(0.0, 1.0, (n, q_s))
    if builder is None:
        y: np.ndarray | CensoredData = np.abs(rng.normal(1.5, 0.5, n)) + 0.1
        support = (float(y.min() - 0.1), float(y.max() + 0.1))
    else:
        y = builder(n, rng)
        support = _support_from_cd(y)
    basis = BernsteinBasis(order=p - 1, support=support)

    _, grad_nll = negative_log_likelihood(
        theta,
        basis,
        y,
        X,
        censoring,
        gradient=True,
        base_distribution=base_distribution,
        scaling=X_s,
    )
    scores = score_matrix(
        theta,
        basis,
        y,
        X,
        censoring,
        base_distribution=base_distribution,
        scaling=X_s,
    )
    assert scores.shape == (n, p + q_d + q_s)
    np.testing.assert_allclose(scores.sum(axis=0), -grad_nll, atol=1e-10)


@pytest.mark.parametrize("base_distribution", _DISTS)
@pytest.mark.parametrize(
    "censoring,builder",
    [
        (CensoringType.NONE, None),
        (CensoringType.RIGHT, _make_right_censored),
        (CensoringType.LEFT, _make_left_censored),
        (CensoringType.INTERVAL, _make_interval),
    ],
)
def test_scaled_hessian_matches_finite_differences(
    base_distribution: str, censoring: CensoringType, builder
) -> None:
    """Analytical Hessian matches FD of analytical gradient across all paths."""
    rng = np.random.default_rng(7)
    n = 40
    p, q_d, q_s = 5, 2, 1
    theta = np.concatenate([np.linspace(-1.2, 1.4, p), [0.25, -0.15], [0.30]])
    X = rng.normal(0.0, 1.0, (n, q_d))
    X_s = rng.normal(0.0, 1.0, (n, q_s))
    if builder is None:
        y: np.ndarray | CensoredData = np.abs(rng.normal(1.5, 0.5, n)) + 0.1
        support = (float(y.min() - 0.1), float(y.max() + 0.1))
    else:
        y = builder(n, rng)
        support = _support_from_cd(y)
    basis = BernsteinBasis(order=p - 1, support=support)

    def grad_fn(t: np.ndarray) -> np.ndarray:
        _, g = negative_log_likelihood(
            t,
            basis,
            y,
            X,
            censoring,
            gradient=True,
            base_distribution=base_distribution,
            scaling=X_s,
        )
        return g

    H_ana = hessian(
        theta,
        basis,
        y,
        X,
        censoring,
        base_distribution=base_distribution,
        scaling=X_s,
    )
    H_fd = np.stack(
        [
            approx_fprime(theta, lambda t, i=i: grad_fn(t)[i], 1e-5)
            for i in range(theta.size)
        ]
    )
    H_fd = 0.5 * (H_fd + H_fd.T)

    np.testing.assert_allclose(H_ana, H_ana.T, atol=1e-10)
    np.testing.assert_allclose(H_ana, H_fd, atol=5e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Model-level wiring: scaled fits expose hessian_ / estfun() / vcov()
# ---------------------------------------------------------------------------


def test_scaled_fit_vcov_returns_full_block() -> None:
    """``MLT(..., scaling=X_s).fit(...).vcov()`` returns the full block matrix.

    Before #77 the scaled fit path left ``hessian_`` as ``None`` and
    :meth:`vcov` raised ``RuntimeError``.  After wiring the analytical
    Hessian into ``fit``, ``vcov()`` must return a positive-definite
    symmetric matrix of shape ``(p + q_d + q_s, p + q_d + q_s)``.
    """
    prob = _toy_scaled_problem(n=80, p=5, q_d=2, q_s=1, seed=42)
    model = MLT(
        order=prob["p"] - 1,
        support=prob["basis"].support,
        scaling=prob["X_s"],
    )
    model.fit(prob["y"], X=prob["X"])

    k = prob["p"] + prob["q_d"] + prob["q_s"]
    V = model.vcov()
    assert V.shape == (k, k)
    np.testing.assert_allclose(V, V.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(V)
    assert eigvals.min() > 0.0


def test_scaled_fit_standard_errors_returns_full_block() -> None:
    """``standard_errors()`` returns ``sqrt(diag(vcov()))`` of length p+q_d+q_s.

    The γ block's SEs are precisely the entries this slice wires up; they
    were unreachable before #77.
    """
    prob = _toy_scaled_problem(n=80, p=5, q_d=2, q_s=1, seed=42)
    model = MLT(
        order=prob["p"] - 1,
        support=prob["basis"].support,
        scaling=prob["X_s"],
    )
    model.fit(prob["y"], X=prob["X"])
    k = prob["p"] + prob["q_d"] + prob["q_s"]
    se = model.standard_errors()
    assert se.shape == (k,)
    assert np.all(se > 0.0)
    np.testing.assert_allclose(se, np.sqrt(np.diag(model.vcov())), atol=1e-12)


def test_scaled_fit_estfun_returns_full_block_and_sums_to_minus_grad() -> None:
    """``estfun()`` on a scaled fit is ``(n, p+q_d+q_s)`` with the score identity.

    The KKT identity is ``estfun.sum(0) == -grad(NLL)``; at an unconstrained
    MLE both sides are zero, but with active monotonicity constraints the
    common value is the (negative) Lagrange multiplier vector.  The shape
    and the identity are what we test here, not ``==0``.
    """
    prob = _toy_scaled_problem(n=80, p=5, q_d=2, q_s=1, seed=42)
    model = MLT(
        order=prob["p"] - 1,
        support=prob["basis"].support,
        scaling=prob["X_s"],
    )
    model.fit(prob["y"], X=prob["X"])
    n = prob["y"].size
    k = prob["p"] + prob["q_d"] + prob["q_s"]
    U = model.estfun()
    assert U.shape == (n, k)

    _, grad_nll = negative_log_likelihood(
        model.theta_,
        model.basis,
        prob["y"],
        prob["X"],
        CensoringType.NONE,
        gradient=True,
        base_distribution=model.base_distribution,
        scaling=prob["X_s"],
    )
    np.testing.assert_allclose(U.sum(axis=0), -grad_nll, atol=1e-10)


def test_scaled_fit_sandwich_vcov_returns_full_block() -> None:
    """``sandwich_vcov()`` on a scaled fit yields a symmetric ``(k, k)`` matrix.

    The sandwich estimator ``V = H⁻¹ M H⁻¹`` is well-defined as soon as both
    ``hessian_`` and ``_estfun_cache_`` are full-rank — both newly wired in #77.
    """
    prob = _toy_scaled_problem(n=80, p=5, q_d=2, q_s=1, seed=42)
    model = MLT(
        order=prob["p"] - 1,
        support=prob["basis"].support,
        scaling=prob["X_s"],
    )
    model.fit(prob["y"], X=prob["X"])
    k = prob["p"] + prob["q_d"] + prob["q_s"]
    V_HC0 = model.sandwich_vcov()
    assert V_HC0.shape == (k, k)
    np.testing.assert_allclose(V_HC0, V_HC0.T, atol=1e-10)
    # Sandwich variance need not be PD in general, but the diagonal must be
    # positive (otherwise ``sandwich_se()`` returns NaN).
    assert np.all(np.diag(V_HC0) > 0.0)


def test_scaled_fit_wald_test_on_gamma_contrast_returns_finite_pvalue() -> None:
    """A Wald test on the γ block runs and yields ``W ≥ 0``, ``p ∈ (0, 1]``.

    The contrast picks out the single γ coefficient: ``H0: γ = 0``.  Without
    the wiring slice the call would fail because ``vcov()`` raised.
    """
    prob = _toy_scaled_problem(n=80, p=5, q_d=2, q_s=1, seed=42)
    model = MLT(
        order=prob["p"] - 1,
        support=prob["basis"].support,
        scaling=prob["X_s"],
    )
    model.fit(prob["y"], X=prob["X"])
    k = prob["p"] + prob["q_d"] + prob["q_s"]
    # Contrast row: pick out γ_1.
    R = np.zeros((1, k))
    R[0, prob["p"] + prob["q_d"]] = 1.0
    result = model.wald_test(R)
    assert result.df == 1
    assert result.statistic >= 0.0
    assert 0.0 < result.p_value <= 1.0
    assert result.vcov_type == "information"


# ---------------------------------------------------------------------------
# R parity: BoxCox / Coxph / Colr scaled vcov, sandwich, wald
# ---------------------------------------------------------------------------


def _load_scaled_vcov_fixture(tag: str) -> dict | None:
    """Return self-contained BoxCox/Coxph/Colr scaled vcov reference.

    Each ``scaling_vcov_<tag>_*`` fixture set rebuilds the full fit, so
    these tests do not share data with the upstream tracer fits (which are
    pinned for #70/#71 parity at the boundary).  The dedicated fits are
    tuned (sample size + order) to give an *interior* MLE for BoxCox and
    Colr, which is required for element-wise R parity (see the docstring
    on ``test_scaled_vcov_matches_R``).  Coxph remains structurally
    constraint-binding on its baseline; that test asserts parity on the
    β / γ sub-block only.
    """
    has_event = tag == "coxph"
    needed = [
        f"scaling_vcov_{tag}_y.txt",
        f"scaling_vcov_{tag}_x_d.txt",
        f"scaling_vcov_{tag}_x_s.txt",
        f"scaling_vcov_{tag}_support.txt",
        f"scaling_vcov_{tag}_theta.txt",
        f"scaling_vcov_{tag}.txt",
        f"scaling_vcov_{tag}_HC0.txt",
        f"scaling_vcov_{tag}_dim.txt",
        f"scaling_vcov_{tag}_wald_gamma.txt",
    ]
    if has_event:
        needed.append(f"scaling_vcov_{tag}_event.txt")
    if any(not (REF_DIR / n).exists() for n in needed):
        return None

    dim = np.loadtxt(REF_DIR / f"scaling_vcov_{tag}_dim.txt").astype(int)
    p, q_d, q_s = int(dim[0]), int(dim[1]), int(dim[2])
    k = p + q_d + q_s
    V_info = np.loadtxt(REF_DIR / f"scaling_vcov_{tag}.txt").reshape(k, k)
    V_HC0 = np.loadtxt(REF_DIR / f"scaling_vcov_{tag}_HC0.txt").reshape(k, k)
    wald = np.loadtxt(REF_DIR / f"scaling_vcov_{tag}_wald_gamma.txt")
    support = tuple(np.loadtxt(REF_DIR / f"scaling_vcov_{tag}_support.txt"))
    y_raw = np.loadtxt(REF_DIR / f"scaling_vcov_{tag}_y.txt")
    out: dict = {
        "p": p,
        "q_d": q_d,
        "q_s": q_s,
        "support": (float(support[0]), float(support[1])),
        "x_d": np.loadtxt(REF_DIR / f"scaling_vcov_{tag}_x_d.txt").reshape(-1, 1),
        "x_s": np.loadtxt(REF_DIR / f"scaling_vcov_{tag}_x_s.txt").reshape(-1, 1),
        "theta_r": np.loadtxt(REF_DIR / f"scaling_vcov_{tag}_theta.txt"),
        "V_info_r": V_info,
        "V_HC0_r": V_HC0,
        "W_r": float(wald[0]),
        "wald_df_r": int(wald[1]),
        "wald_p_r": float(wald[2]),
    }
    if has_event:
        event = np.loadtxt(REF_DIR / f"scaling_vcov_{tag}_event.txt").astype(bool)
        out["y"] = CensoredData.right_censored(y_raw, censored=~event)
    else:
        out["y"] = y_raw
    return out


def _sign_matrix_for(tag: str, p: int, q_d: int, q_s: int) -> np.ndarray:
    """Diagonal signing matrix that aligns mltpy's vcov with R's.

    mltpy parametrises ``h + Xβ``; ``tram::BoxCox`` uses ``negative = TRUE``
    so ``β_R = -β_mltpy`` — both rows and columns indexed by β flip sign in
    the vcov.  ``tram::Coxph`` / ``Colr`` use ``negative = FALSE`` (no flip).
    γ is sign-aligned across all three classes (ADR 0002 Decision 5).
    """
    s = np.ones(p + q_d + q_s)
    if tag == "boxcox":
        s[p : p + q_d] = -1.0
    return np.diag(s)


def _theta_mltpy_at_R(tag: str, ref: dict) -> np.ndarray:
    """Convert R's coef vector to mltpy's sign convention.

    BoxCox flips β to absorb ``negative = TRUE``; Coxph / Colr keep β.  γ is
    sign-aligned across all three classes (ADR 0002 Decision 5).
    """
    s = np.ones_like(ref["theta_r"])
    if tag == "boxcox":
        s[ref["p"] : ref["p"] + ref["q_d"]] = -1.0
    return ref["theta_r"] * s


def _base_dist_for(tag: str) -> str:
    return {
        "boxcox": "normal",
        "coxph": "min_extreme_value",
        "colr": "logistic",
    }[tag]


def _censoring_for(tag: str) -> CensoringType:
    return {
        "boxcox": CensoringType.NONE,
        "coxph": CensoringType.RIGHT,
        "colr": CensoringType.NONE,
    }[tag]


@pytest.mark.parametrize("tag", ["boxcox", "colr"])
def test_scaled_vcov_bare_matches_R(tag: str) -> None:
    """``inv(hessian(θ_R, ...))`` matches R's ``vcov(as.mlt(fit))``.

    Function-vs-function check at R's reported θ (sign-flipped for BoxCox's
    β).  For BoxCox and Colr the dedicated fixtures land at *interior*
    MLEs (no active monotonicity constraints), so R's ``vcov.mlt`` reduces
    to bare ``solve(H)`` and mltpy's bare ``inv(H @ θ_R)`` matches the
    full ``(p+q_d+q_s)²`` block at rtol=1e-4 / atol=1e-6.

    The Coxph case lives in :func:`test_scaled_vcov_coxph_matches_R_via_auglag`
    because its baseline hazard is structurally constraint-binding and R's
    ``vcov.mlt`` applies an active-set penalty that bare ``inv(H)`` misses
    — the test there exercises mltpy's ``regularize='auglag'`` mode
    end-to-end instead.
    """
    ref = _load_scaled_vcov_fixture(tag)
    if ref is None:
        pytest.skip(f"scaling_vcov_{tag}_* reference fixtures not found")
    basis = BernsteinBasis(order=ref["p"] - 1, support=ref["support"])
    theta_mltpy = _theta_mltpy_at_R(tag, ref)
    H = hessian(
        theta_mltpy,
        basis,
        ref["y"],
        ref["x_d"],
        _censoring_for(tag),
        base_distribution=_base_dist_for(tag),
        scaling=ref["x_s"],
    )
    V_py = np.linalg.inv(H)
    S = _sign_matrix_for(tag, ref["p"], ref["q_d"], ref["q_s"])
    V_expected = S @ ref["V_info_r"] @ S
    np.testing.assert_allclose(V_py, V_expected, rtol=1e-4, atol=1e-6)


def test_scaled_vcov_coxph_matches_R_via_auglag() -> None:
    """``Coxph.vcov(regularize='auglag')`` matches R's ``vcov(as.mlt(fit))``.

    The scaled-baseline Coxph fit lands at the monotonicity boundary (two
    adjacent θ_b coefficients tie up).  At that boundary R's
    ``vcov(as.mlt(fit))`` applies an active-set penalty that bare
    ``inv(H)`` from mltpy misses — the function-vs-function check would
    fail at ``max_rel ≈ 37`` on the full block.  ``vcov(regularize='auglag')``
    pre-augments the Hessian along binding rows with the same
    ``ρ · Aᵀ_active A_active`` term R uses, recovering full-block parity at
    ``rtol ≈ 5e-3`` (driven by the small optimiser drift in θ, ``Δθ ≈
    1e-5``).

    Why ``'auglag'`` rather than the default ``'active'``: the latter is
    lazy and only fires on bare-inversion failure, so on the
    well-conditioned (cond≈243) Coxph Hessian here it returns bare
    ``inv(H)`` — the same value that mismatches R.  ``'auglag'`` is the
    opt-in unconditional mode introduced precisely for this case (see
    ``vcov`` docstring for the trade-off).  The BoxCox / Colr fixtures
    land interior and continue to match R via the bare path in
    :func:`test_scaled_vcov_bare_matches_R`.
    """
    ref = _load_scaled_vcov_fixture("coxph")
    if ref is None:
        pytest.skip("scaling_vcov_coxph_* reference fixtures not found")
    model = _fit_scaled("coxph", ref)
    V_py = model.vcov(regularize="auglag")
    S = _sign_matrix_for("coxph", ref["p"], ref["q_d"], ref["q_s"])
    V_expected = S @ ref["V_info_r"] @ S
    # Full block parity, including the θ_b and cross-block entries that
    # bare ``inv(H)`` was ~37× off on.  Tolerance absorbs the small
    # optimiser drift between mltpy's and R's auglag (``Δθ ≈ 1e-5``)
    # propagated through the inverse of an ill-conditioned matrix.
    np.testing.assert_allclose(V_py, V_expected, rtol=5e-3, atol=1e-4)


@pytest.mark.parametrize("tag", ["boxcox", "colr"])
def test_scaled_sandwich_matches_R(tag: str) -> None:
    """``V_HC0 = V_info U'U V_info`` (HC0 sandwich) matches R element-wise.

    Reuses R's vcov for the bread and mltpy's analytical score matrix for
    the meat, mirroring the R recipe: ``vcov(as.mlt(fit)) %*%
    crossprod(estfun(as.mlt(fit))) %*% vcov(as.mlt(fit))``.  Because the
    meat is computed from the same per-observation scores (signed for β
    under BoxCox) the comparison is essentially testing whether mltpy's
    ``score_matrix`` reproduces R's ``estfun.mlt`` row-by-row at the same θ.

    The Coxph case lives in
    :func:`test_scaled_sandwich_coxph_matches_R_via_auglag` for the same
    reason as the vcov test: R's bread carries an active-set penalty that
    bare ``inv(H)`` misses, and the leak compounds through both bread
    copies of the sandwich.
    """
    ref = _load_scaled_vcov_fixture(tag)
    if ref is None:
        pytest.skip(f"scaling_vcov_{tag}_* reference fixtures not found")
    basis = BernsteinBasis(order=ref["p"] - 1, support=ref["support"])
    theta_mltpy = _theta_mltpy_at_R(tag, ref)
    H = hessian(
        theta_mltpy,
        basis,
        ref["y"],
        ref["x_d"],
        _censoring_for(tag),
        base_distribution=_base_dist_for(tag),
        scaling=ref["x_s"],
    )
    V_py_info = np.linalg.inv(H)
    U = score_matrix(
        theta_mltpy,
        basis,
        ref["y"],
        ref["x_d"],
        _censoring_for(tag),
        base_distribution=_base_dist_for(tag),
        scaling=ref["x_s"],
    )
    V_py_HC0 = V_py_info @ (U.T @ U) @ V_py_info
    S = _sign_matrix_for(tag, ref["p"], ref["q_d"], ref["q_s"])
    V_expected = S @ ref["V_HC0_r"] @ S
    np.testing.assert_allclose(V_py_HC0, V_expected, rtol=1e-4, atol=1e-6)


def test_scaled_sandwich_coxph_matches_R_via_auglag() -> None:
    """``Coxph.sandwich_vcov(regularize='auglag')`` matches R's sandwich full-block.

    Same mechanism as :func:`test_scaled_vcov_coxph_matches_R_via_auglag`:
    the scaled Coxph fit binds two adjacent θ_b coefficients on the
    monotonicity boundary, so R's ``vcov(as.mlt(fit))`` — which forms the
    bread of the sandwich — augments along the active constraint.  Mltpy's
    bread must follow suit; otherwise the penalty leak compounds through
    both ``V_info`` copies of ``V = V_info · UᵀU · V_info``.

    End-to-end fit through the public ``sandwich_vcov`` API (not
    function-vs-function at θ_R), so the tolerance absorbs both the small
    optimiser drift in θ̂ (``Δθ ≈ 1e-5``) and the meat reweighting that
    comes from evaluating ``U`` at mltpy's θ̂ rather than R's.  Empirically
    those two error sources combine to ``max_rel ≈ 5e-5`` on the full block;
    we use the same ``rtol=1e-4, atol=1e-6`` as the BoxCox / Colr branch so
    Coxph is no longer the loose case.  The previous sub-block-only check at
    rtol=1e-1 is superseded entirely.
    """
    ref = _load_scaled_vcov_fixture("coxph")
    if ref is None:
        pytest.skip("scaling_vcov_coxph_* reference fixtures not found")
    model = _fit_scaled("coxph", ref)
    V_py_HC0 = model.sandwich_vcov(regularize="auglag")
    S = _sign_matrix_for("coxph", ref["p"], ref["q_d"], ref["q_s"])
    V_expected = S @ ref["V_HC0_r"] @ S
    np.testing.assert_allclose(V_py_HC0, V_expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("tag", ["boxcox", "colr"])
def test_scaled_wald_test_gamma_matches_R(tag: str) -> None:
    """Wald(H0: γ_1 = 0) on a scaled fit matches R's W statistic and p-value.

    The fit is re-run in mltpy (so this hits the public ``wald_test`` API
    end-to-end, not just the analytical Hessian).  The statistic and p-value
    target R's ``Rθ' (RVR')⁻¹ Rθ`` computed from ``vcov(as.mlt(fit))``.

    For BoxCox / Colr the vcov is interior and parity holds at rtol=1e-3 —
    mltpy re-fits rather than using R's reported θ, so the optimiser stops
    at a slightly different θ̂ and that residual propagates into Rθ and
    (RVR').  The Coxph case lives in
    :func:`test_scaled_wald_test_gamma_coxph_matches_R_via_auglag` because
    its active-constraint penalty has to be reproduced via
    ``regularize='auglag'`` for parity on the γ-row of (RVR').
    """
    ref = _load_scaled_vcov_fixture(tag)
    if ref is None:
        pytest.skip(f"scaling_vcov_{tag}_* reference fixtures not found")
    model = _fit_scaled(tag, ref)
    k = ref["p"] + ref["q_d"] + ref["q_s"]
    R = np.zeros((1, k))
    R[0, ref["p"] + ref["q_d"]] = 1.0
    result = model.wald_test(R)
    rtol = 1e-3
    atol = 1e-5
    np.testing.assert_allclose(result.statistic, ref["W_r"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(result.p_value, ref["wald_p_r"], rtol=rtol, atol=atol)
    assert result.df == ref["wald_df_r"] == 1


def test_scaled_wald_test_gamma_coxph_matches_R_via_auglag() -> None:
    """Wald(H0: γ_1 = 0) on scaled Coxph matches R via auglag-bread vcov.

    The Wald statistic is ``Rθ̂ᵀ (R V Rᵀ)⁻¹ Rθ̂`` with ``V`` the bread of
    the sandwich (or the information vcov, here).  On scaled Coxph the
    fitted θ̂ binds the baseline monotonicity boundary, so R's
    ``vcov(as.mlt(fit))`` augments along the active constraint.  Calling
    ``model.wald_test(R, regularize='auglag')`` threads that same
    augmentation into mltpy's V, recovering parity on the γ-row that bare
    ``inv(H)`` was off by ~37×.  Empirically W matches R at ``rel ≈ 8e-5``;
    we use the same ``rtol=1e-3, atol=1e-5`` as the BoxCox / Colr branch
    (driven by the optimiser-drift residual in θ̂ that propagates into Rθ
    and (RVR'), not by the active-constraint penalty).  The previous loose
    Coxph branch (``rtol=5e-2, atol=1e-2``) is superseded entirely.
    """
    ref = _load_scaled_vcov_fixture("coxph")
    if ref is None:
        pytest.skip("scaling_vcov_coxph_* reference fixtures not found")
    model = _fit_scaled("coxph", ref)
    k = ref["p"] + ref["q_d"] + ref["q_s"]
    R = np.zeros((1, k))
    R[0, ref["p"] + ref["q_d"]] = 1.0
    result = model.wald_test(R, regularize="auglag")
    rtol = 1e-3
    atol = 1e-5
    np.testing.assert_allclose(result.statistic, ref["W_r"], rtol=rtol, atol=atol)
    np.testing.assert_allclose(result.p_value, ref["wald_p_r"], rtol=rtol, atol=atol)
    assert result.df == ref["wald_df_r"] == 1


def _fit_scaled(tag: str, ref: dict):
    """Fit the appropriate mltpy model class on the loaded reference data."""
    if tag == "boxcox":
        model: object = MLT(
            order=ref["p"] - 1,
            support=ref["support"],
            scaling=ref["x_s"],
        )
        model.fit(ref["y"], X=ref["x_d"])
    elif tag == "coxph":
        model = Coxph(
            order=ref["p"] - 1,
            support=ref["support"],
            scaling=ref["x_s"],
        )
        model.fit(ref["y"], X=ref["x_d"])
    elif tag == "colr":
        model = Colr(
            order=ref["p"] - 1,
            support=ref["support"],
            scaling=ref["x_s"],
        )
        model.fit(ref["y"], X=ref["x_d"])
    else:  # pragma: no cover
        raise ValueError(f"unknown tag {tag!r}")
    return model


# ---------------------------------------------------------------------------
# intercept_score under scaling
# ---------------------------------------------------------------------------


def test_residuals_score_works_on_scaled_fit() -> None:
    """``residuals(type='score')`` on a scaled fit returns ``(n,)`` finite values.

    Before #77 the call would raise a shape-mismatch ``ValueError`` because
    the underlying ``intercept_score`` private helper did not know about γ
    and tried to multiply ``X_d`` (q_d cols) by ``theta[p:]`` (q_d + q_s
    entries).  After #77 the artificial intercept is well-defined on the
    *final* h (i.e. post-scaling), and the closed-form score formulas apply
    unchanged once h is evaluated at the scaled value.

    Semantics under scaling
    -----------------------
    Let ``h(y|x_d, x_s) = h_0(y) · exp(0.5·x_s·γ) + x_d·β + offset`` be the
    fitted CTM (ADR 0002).  The hypothetical intercept α is added *after*
    scaling: ``h̃(y|x) = h(y|x) + α``.  Therefore ``∂h̃/∂α = 1`` for every
    row, and ``∂ℓ_i/∂α = ψ(h_i)`` evaluated at the *scaled* h — same
    closed form as the unscaled case, just at a different point.  γ does
    not appear in the formula itself, but it shapes h_i and so changes
    the *value* of the residual.
    """
    prob = _toy_scaled_problem(n=60, p=5, q_d=2, q_s=1, seed=11)
    model = MLT(
        order=prob["p"] - 1,
        support=prob["basis"].support,
        scaling=prob["X_s"],
    )
    model.fit(prob["y"], X=prob["X"])
    r = model.residuals(type="score")
    assert r.shape == (prob["y"].size,)
    assert np.all(np.isfinite(r))


def test_scaled_intercept_score_matches_closed_form_exact_normal() -> None:
    """``intercept_score(scaling=...)`` matches ``-ψ(scaled_h_i)`` row-by-row.

    Closed form for exact / normal: ``∂ℓ_i/∂α = -h_i`` where ``α`` is the
    artificial intercept on h.  Under scaling ``h_i = h_0(y_i)·exp(0.5·x_s·γ)
    + x_d·β``, so the residual is ``-h_i`` evaluated at the scaled h.  This
    pins the wiring beyond shape-only.
    """
    from mltpy.likelihood import intercept_score

    prob = _toy_scaled_problem(n=40, p=5, q_d=2, q_s=1, seed=21)
    theta = prob["theta"]
    p, q_d = prob["p"], prob["q_d"]
    theta_b = theta[:p]
    beta = theta[p : p + q_d]
    gamma = theta[p + q_d :]
    B = prob["basis"].evaluate(prob["y"])
    f = np.exp(0.5 * prob["X_s"] @ gamma)
    h_scaled = (B @ theta_b) * f + prob["X"] @ beta
    # For standard normal ``ψ(h) = d log f(h)/dh = -h`` so
    # ``intercept_score = ψ(h) = -h`` evaluated at the scaled h.
    expected = -h_scaled

    got = intercept_score(
        theta,
        prob["basis"],
        prob["y"],
        prob["X"],
        CensoringType.NONE,
        base_distribution="normal",
        scaling=prob["X_s"],
    )
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_coxsnell_residuals_shift_and_scaling_match_cumhazard() -> None:
    """Cox-Snell residuals on a shift+scaling fit equal the model's own
    cumulative hazard ``-log S(y)``.

    For an exact observation the Cox-Snell residual is ``-log S(y|x)``,
    which ``predict(what='cumhazard', ...)`` returns via the
    scaling-correct h evaluation ``h = h_0(y)·exp(0.5·x_s·γ) + x_d·β``.
    Before the fix the residuals() Cox-Snell path split ``theta[p:]`` as if
    it were all shift coefficients, so with shift *and* scaling covariates
    it multiplied ``X_d`` (q_d cols) by ``[β | γ]`` (q_d + q_s entries) and
    raised a shape-mismatch ``ValueError``.
    """
    prob = _toy_scaled_problem(n=60, p=5, q_d=2, q_s=1, seed=11)
    model = MLT(
        order=prob["p"] - 1,
        support=prob["basis"].support,
        scaling=prob["X_s"],
    )
    model.fit(prob["y"], X=prob["X"])

    r_cs = model.residuals(type="cox-snell")
    oracle = model.predict(
        prob["y"], X_new=prob["X"], what="cumhazard", X_scale_new=prob["X_s"]
    )
    assert r_cs.shape == (prob["y"].size,)
    np.testing.assert_allclose(r_cs, oracle, rtol=1e-10, atol=1e-12)


def test_coxsnell_residuals_scaling_only_apply_scale_factor() -> None:
    """Cox-Snell residuals on a scaling-only fit (no shift covariates)
    include the ``exp(0.5·x_s·γ)`` factor.

    With no shift covariates the old code did not crash but silently
    omitted the scaling factor, evaluating ``h_0(y)`` instead of
    ``h_0(y)·exp(0.5·x_s·γ)``.  The cumhazard oracle, which applies the
    factor, would then disagree.
    """
    prob = _toy_scaled_problem(n=50, p=5, q_d=0, q_s=1, seed=7)
    model = MLT(
        order=prob["p"] - 1,
        support=prob["basis"].support,
        scaling=prob["X_s"],
    )
    model.fit(prob["y"])

    r_cs = model.residuals(type="cox-snell")
    oracle = model.predict(prob["y"], what="cumhazard", X_scale_new=prob["X_s"])
    np.testing.assert_allclose(r_cs, oracle, rtol=1e-10, atol=1e-12)
    # Guard against a degenerate γ≈0 fit that would make the omitted-factor
    # bug invisible: the scaling block must be meaningfully non-zero, so a
    # residual computed at h_0(y) alone would differ from r_cs.
    gamma = model.gamma_coef_
    assert gamma is not None and np.max(np.abs(gamma)) > 0.05


def _fit_scaled_model(
    n: int = 60, p: int = 5, q_d: int = 2, q_s: int = 1, seed: int = 3
) -> tuple[MLT, dict]:
    """Fit a small exact-data scaled MLT and return (model, problem)."""
    prob = _toy_scaled_problem(n=n, p=p, q_d=q_d, q_s=q_s, seed=seed)
    model = MLT(
        order=prob["p"] - 1,
        support=prob["basis"].support,
        scaling=prob["X_s"],
    )
    if q_d > 0:
        model.fit(prob["y"], X=prob["X"])
    else:
        model.fit(prob["y"])
    return model, prob


def _eta_of_theta(
    model: MLT,
    theta: np.ndarray,
    y: np.ndarray,
    x_d: np.ndarray | None,
    x_s: np.ndarray | None,
    what: str,
    dist,
) -> np.ndarray:
    """η(y; θ) on the scaled model, as a pure function of the full θ vector.

    Mirrors the linear-predictor definitions of :meth:`confband` so that an
    independent finite-difference Jacobian can validate the analytical one,
    including the γ block.
    """
    p = model.basis.order + 1
    q_s = 0 if model.scaling is None else model.scaling.shape[1]
    q_d = theta.size - p - q_s
    theta_b = theta[:p]
    beta = theta[p : p + q_d] if q_d > 0 else None
    gamma = theta[p + q_d :] if q_s > 0 else None

    B = model.basis.evaluate(y)
    D = model.basis.derivative(y, order=1)
    s = float(np.exp(0.5 * (x_s @ gamma))) if (gamma is not None) else 1.0
    h = (B @ theta_b) * s
    hp = (D @ theta_b) * s
    if beta is not None and x_d is not None:
        h = h + x_d @ beta
    if what in ("trafo", "distribution", "survivor"):
        return h
    if what == "density":
        return dist.logpdf(h) + np.log(hp)
    return dist.logpdf(h) + np.log(hp) - dist.logsf(h)  # hazard


@pytest.mark.parametrize(
    "what", ["trafo", "distribution", "survivor", "density", "hazard"]
)
def test_confband_scaling_matches_finite_difference_delta(what: str) -> None:
    """Scaling-aware ``confband`` equals an independent FD delta-method band.

    Finite-differences η (a pure function of the full ``[θ_b | β | γ]``
    vector) to form the delta-method Jacobian, applies the model's own
    ``vcov`` and the same back-transforms / ``z`` as :meth:`confband`, and
    asserts agreement.  This exercises the γ columns of the analytical
    Jacobian, which the old shift-only code omitted entirely.
    """
    model, prob = _fit_scaled_model()
    dist = _get_dist(model.base_distribution)
    x_d = prob["X"][0]
    x_s = prob["X_s"][0]
    # Interior grid keeps |h| well below the ±30 clip on every `what`.
    grid = np.linspace(prob["y"].min() + 0.1, prob["y"].max() - 0.1, 12)

    band = model.confband(
        grid, X=x_d[None, :], what=what, X_scale=x_s[None, :], level=0.95
    )

    theta = model.theta_
    V = model.vcov()
    z = norm.ppf(0.5 * (1.0 + 0.95))

    def eta_fn(t: np.ndarray) -> np.ndarray:
        return _eta_of_theta(model, t, grid, x_d, x_s, what, dist)

    eta0 = eta_fn(theta)
    # FD Jacobian J[i, j] = ∂η_i/∂θ_j.
    J = np.empty((grid.size, theta.size))
    step = 1e-6
    for j in range(theta.size):
        tp = theta.copy()
        tp[j] += step
        J[:, j] = (eta_fn(tp) - eta0) / step
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", J, V, J), 0.0))
    lo_eta, hi_eta = eta0 - z * se, eta0 + z * se

    if what == "trafo":
        est_ref, lo_ref, hi_ref = eta0, lo_eta, hi_eta
    elif what == "distribution":
        est_ref, lo_ref, hi_ref = dist.cdf(eta0), dist.cdf(lo_eta), dist.cdf(hi_eta)
    elif what == "survivor":
        est_ref, lo_ref, hi_ref = dist.sf(eta0), dist.sf(hi_eta), dist.sf(lo_eta)
    else:  # density / hazard
        est_ref, lo_ref, hi_ref = np.exp(eta0), np.exp(lo_eta), np.exp(hi_eta)

    np.testing.assert_allclose(band[:, 0], est_ref, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(band[:, 1], lo_ref, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(band[:, 2], hi_ref, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("what", ["distribution", "survivor", "density"])
def test_confband_scaling_estimate_matches_point_prediction(what: str) -> None:
    """The band's central column equals the scaling-aware point prediction.

    Guards specifically against the old bug of omitting the
    ``exp(0.5·x_s·γ)`` factor on ``h``: if the factor were dropped the
    centre curve would disagree with :meth:`predict`.
    """
    model, prob = _fit_scaled_model()
    x_d = prob["X"][0]
    x_s = prob["X_s"][0]
    grid = np.linspace(prob["y"].min() + 0.1, prob["y"].max() - 0.1, 15)

    band = model.confband(grid, X=x_d[None, :], what=what, X_scale=x_s[None, :])
    pred_what = {
        "distribution": "distribution",
        "survivor": "survivor",
        "density": "density",
    }[what]
    X_rep = np.repeat(x_d[None, :], grid.size, axis=0)
    Xs_rep = np.repeat(x_s[None, :], grid.size, axis=0)
    point = model.predict(grid, X_new=X_rep, what=pred_what, X_scale_new=Xs_rep)
    np.testing.assert_allclose(band[:, 0], point, rtol=1e-9, atol=1e-11)


def test_confband_scaling_factor_actually_shifts_band() -> None:
    """A non-zero γ profile changes the band — it is not scaling-blind.

    Two different ``X_scale`` profiles must yield different bands; if the γ
    block were ignored (the old bug) both calls would return the same curve.
    """
    model, prob = _fit_scaled_model()
    x_d = prob["X"][0]
    grid = np.linspace(prob["y"].min() + 0.1, prob["y"].max() - 0.1, 10)

    band_a = model.confband(
        grid, X=x_d[None, :], what="survivor", X_scale=np.array([[1.5]])
    )
    band_b = model.confband(
        grid, X=x_d[None, :], what="survivor", X_scale=np.array([[-1.5]])
    )
    # The fitted γ is meaningfully non-zero, so the two profiles diverge.
    assert model.gamma_coef_ is not None
    assert np.max(np.abs(model.gamma_coef_)) > 0.02
    assert not np.allclose(band_a, band_b)


def test_confband_scaling_only_no_shift_covariates() -> None:
    """Scaling-only fit (q_d == 0): X must be None, X_scale required."""
    model, prob = _fit_scaled_model(q_d=0, q_s=1, seed=11)
    x_s = prob["X_s"][0]
    grid = np.linspace(prob["y"].min() + 0.1, prob["y"].max() - 0.1, 8)

    band = model.confband(grid, what="distribution", X_scale=x_s[None, :])
    assert band.shape == (grid.size, 3)
    # Central column matches the scaling-only point prediction.
    Xs_rep = np.repeat(x_s[None, :], grid.size, axis=0)
    point = model.predict(grid, what="distribution", X_scale_new=Xs_rep)
    np.testing.assert_allclose(band[:, 0], point, rtol=1e-9, atol=1e-11)

    with pytest.raises(ValueError, match="without shift covariates"):
        model.confband(grid, X=np.array([[0.0, 0.0]]), X_scale=x_s[None, :])


def test_confband_scaling_argument_validation() -> None:
    """X_scale presence/shape is validated against the fitted scaling layout."""
    model, prob = _fit_scaled_model()
    x_d = prob["X"][0]
    x_s = prob["X_s"][0]
    grid = np.linspace(prob["y"].min() + 0.1, prob["y"].max() - 0.1, 6)

    # Missing X_scale on a scaling fit.
    with pytest.raises(ValueError, match="X_scale is required"):
        model.confband(grid, X=x_d[None, :], what="survivor")

    # Wrong X_scale width.
    with pytest.raises(ValueError, match="X_scale has shape"):
        model.confband(
            grid, X=x_d[None, :], what="survivor", X_scale=np.array([[0.1, 0.2]])
        )

    # X_scale supplied to a non-scaling model is rejected.
    plain = MLT(order=prob["p"] - 1, support=prob["basis"].support)
    plain.fit(prob["y"], X=prob["X"])
    with pytest.raises(ValueError, match="without scaling"):
        plain.confband(grid, X=x_d[None, :], what="survivor", X_scale=x_s[None, :])


# ---------------------------------------------------------------------------
# R parity: scaling-aware confband vs hand-built delta-method band from
# tram::BoxCox(y ~ x_d | x_s).  The reference is produced by
# reference/generate_reference.R (confband_scaling_* block) from the *same*
# interior seed-770 BoxCox fit as the scaling_vcov_boxcox vcov fixture, so
# mltpy's bare vcov matches R's and the band agrees end-to-end.  See
# docs/adr/0002-scaling-terms.md.
# ---------------------------------------------------------------------------

_CONFBAND_SCALING_WHAT = ["trafo", "distribution", "survivor", "density", "hazard"]


def _load_confband_scaling_ref() -> dict | None:
    """Load the R confband-scaling fixtures, or None if not materialised."""
    needed = [
        REF_DIR / "scaling_vcov_boxcox_y.txt",
        REF_DIR / "scaling_vcov_boxcox_x_d.txt",
        REF_DIR / "scaling_vcov_boxcox_x_s.txt",
        REF_DIR / "scaling_vcov_boxcox_support.txt",
        REF_DIR / "confband_scaling_y_grid.txt",
        REF_DIR / "confband_scaling_profile.txt",
        *[REF_DIR / f"confband_scaling_{w}.txt" for w in _CONFBAND_SCALING_WHAT],
    ]
    if not all(p.exists() for p in needed):
        return None
    y = np.loadtxt(REF_DIR / "scaling_vcov_boxcox_y.txt")
    x_d = np.loadtxt(REF_DIR / "scaling_vcov_boxcox_x_d.txt")
    x_s = np.loadtxt(REF_DIR / "scaling_vcov_boxcox_x_s.txt")
    support = tuple(np.loadtxt(REF_DIR / "scaling_vcov_boxcox_support.txt"))
    y_grid = np.loadtxt(REF_DIR / "confband_scaling_y_grid.txt")
    xd0, xs0 = np.loadtxt(REF_DIR / "confband_scaling_profile.txt")
    bands = {
        w: np.loadtxt(REF_DIR / f"confband_scaling_{w}.txt").reshape(y_grid.size, 3)
        for w in _CONFBAND_SCALING_WHAT
    }
    return {
        "y": y,
        "x_d": x_d,
        "x_s": x_s,
        "support": support,
        "y_grid": y_grid,
        "xd0": float(xd0),
        "xs0": float(xs0),
        "bands": bands,
    }


@pytest.mark.parametrize("what", _CONFBAND_SCALING_WHAT)
def test_confband_scaling_matches_R(what: str) -> None:
    """Scaling-aware ``confband`` matches R's delta-method band end-to-end.

    Uses the interior seed-770 ``BoxCox(y ~ x_d | x_s)`` fit (same data as
    the ``scaling_vcov_boxcox`` fixture), so mltpy's ``inv(H)`` vcov equals
    R's ``vcov(as.mlt(fit))`` and the comparison reflects the band formula
    and γ-Jacobian rather than an optimiser/active-set vcov gap.  The band is
    parameterisation-invariant, so no β sign flip is needed despite mltpy and
    R differing on β's sign internally.
    """
    ref = _load_confband_scaling_ref()
    if ref is None:
        pytest.skip("confband_scaling_* reference fixtures not materialised")

    model = BoxCox(order=4, support=ref["support"], scaling=ref["x_s"][:, None]).fit(
        ref["y"], X=ref["x_d"][:, None]
    )
    band = model.confband(
        ref["y_grid"],
        X=np.array([[ref["xd0"]]]),
        what=what,
        X_scale=np.array([[ref["xs0"]]]),
        level=0.95,
    )
    # Tolerance absorbs the small optimiser drift between mltpy's and R's
    # auglag (Δθ ≈ 1e-5) propagated through the non-linear band; the centre
    # column is tighter than the SE-driven endpoints.
    np.testing.assert_allclose(band, ref["bands"][what], rtol=5e-3, atol=2e-3)
