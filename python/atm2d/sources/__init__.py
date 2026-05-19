"""atm2d source-term subpackage.

Public surface:

  - :mod:`atm2d.sources.protocol`        — ``LocalSourceTerm`` protocol,
    ``LocalSourceLinearization`` dataclass, ``RateProvider`` type alias.
  - :mod:`atm2d.sources.indexed`         — built-in mass-action / first-order
    / boundary-flux concrete source-term classes.
  - :mod:`atm2d.sources.combine`         — ``build_source_linearization``,
    ``build_source_global_operator``.
  - :mod:`atm2d.sources.charge_balance`  — ``fold_charge_balance_into_jacobian``
    (added in REFACTOR_SCHEMA Phase 1b).
"""

from .charge_balance import fold_charge_balance_into_jacobian
from .combine import build_source_global_operator, build_source_linearization
from .indexed import (
    IndexedBoundaryFluxSource,
    IndexedBoundaryVelocitySource,
    IndexedFirstOrderSource,
    IndexedMassActionSource,
    IndexedReversibleFirstOrderSource,
)
from .protocol import LocalSourceLinearization, LocalSourceTerm, RateProvider

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
