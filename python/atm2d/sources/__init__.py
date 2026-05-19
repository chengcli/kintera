"""atm2d source-term subpackage (incubator for the source.py split).

Phase 1 of the refactor adds new pure-function helpers here. The full
move of ``LocalSourceTerm`` / ``IndexedFirstOrderSource`` / etc. into
this package happens in Phase 4 (REFACTOR_SCHEMA.html §4, §7).
"""
from .charge_balance import fold_charge_balance_into_jacobian

__all__ = ["fold_charge_balance_into_jacobian"]
