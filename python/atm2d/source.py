"""Backward-compat shim — the contents of this module moved to
:mod:`atm2d.sources` in REFACTOR_SCHEMA Phase 4a.

Importing from ``kintera.atm2d.source`` still resolves to the same
symbols; new code should import from ``kintera.atm2d.sources`` instead.
"""

from .sources import (  # noqa: F401
    IndexedBoundaryFluxSource,
    IndexedBoundaryVelocitySource,
    IndexedFirstOrderSource,
    IndexedMassActionSource,
    IndexedReversibleFirstOrderSource,
    LocalSourceLinearization,
    LocalSourceTerm,
    RateProvider,
    build_source_global_operator,
    build_source_linearization,
    fold_charge_balance_into_jacobian,
)

__all__ = [
    "LocalSourceLinearization",
    "LocalSourceTerm",
    "RateProvider",
    "IndexedFirstOrderSource",
    "IndexedMassActionSource",
    "IndexedReversibleFirstOrderSource",
    "IndexedBoundaryFluxSource",
    "IndexedBoundaryVelocitySource",
    "build_source_linearization",
    "build_source_global_operator",
    "fold_charge_balance_into_jacobian",
]
