"""Behavioural tests for the basis-family structural contracts.

These exercise the public, ``@runtime_checkable`` protocols that pin the
duck-typed basis interface (previously enforced only by docstring and by
hand-maintained union annotations that had drifted from the runtime
``_SUPPORTED_X_BASIS_TYPES`` check).
"""

from __future__ import annotations

import pytest

from mltpy.basis import (
    BasisLike,
    BernsteinBasis,
    InteractionBasis,
    InterceptBasis,
    LegendreBasis,
    LogBasis,
    LogBernsteinBasis,
    OneHotBasis,
    OrdinalBasis,
    PolynomialBasis,
    XBasisLike,
)


def _full_bases() -> list[object]:
    """One instance of every basis exposing the full y-basis contract."""
    return [
        BernsteinBasis(order=3, support=(0.0, 1.0)),
        LogBernsteinBasis(order=3, support=(1.0, 5.0)),
        PolynomialBasis(order=2, support=(0.0, 1.0)),
        LegendreBasis(order=2, support=(0.0, 1.0)),
        LogBasis(support=(1.0, 5.0)),
        InterceptBasis(support=(0.0, 1.0)),
        OrdinalBasis(K=4),
    ]


class TestBasisLike:
    @pytest.mark.parametrize("basis", _full_bases())
    def test_full_bases_satisfy_basislike(self, basis: object) -> None:
        assert isinstance(basis, BasisLike)

    def test_onehot_is_not_basislike(self) -> None:
        # OneHotBasis is an x-only basis: no derivative / integrate / support.
        assert not isinstance(OneHotBasis(K=3), BasisLike)


class TestXBasisLike:
    @pytest.mark.parametrize(
        "basis",
        [
            BernsteinBasis(order=3, support=(0.0, 1.0)),
            OrdinalBasis(K=4),
            InterceptBasis(support=(0.0, 1.0)),
            OneHotBasis(K=3),
        ],
    )
    def test_supported_x_bases_satisfy_xbasislike(self, basis: object) -> None:
        assert isinstance(basis, XBasisLike)


class TestInteractionXBasisRejection:
    def test_rejection_message_lists_onehotbasis(self) -> None:
        # OneHotBasis is a valid x-basis; the rejection message must name it
        # as one of the allowed options (it previously omitted it).
        with pytest.raises(ValueError, match="OneHotBasis"):
            InteractionBasis(
                y_basis=BernsteinBasis(order=2, support=(0.0, 1.0)),
                x_basis=PolynomialBasis(order=2, support=(0.0, 1.0)),
            )

    def test_onehot_is_accepted_as_x_basis(self) -> None:
        ib = InteractionBasis(
            y_basis=BernsteinBasis(order=2, support=(0.0, 1.0)),
            x_basis=OneHotBasis(K=3),
        )
        assert ib.n_x_params == 3
