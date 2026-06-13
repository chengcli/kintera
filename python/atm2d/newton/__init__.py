"""atm2d Newton-solver subpackage.

  - :mod:`atm2d.newton.result`           — ``NewtonResult``,
    ``per_species_relative_change``, postprocess Protocols.
  - :mod:`atm2d.newton.coupled`          — ``newton_implicit_step``
    (transport + chemistry in one global sparse system).
  - :mod:`atm2d.newton.chemistry_only`   — ``chemistry_only_newton_step``
    (cell-local chemistry only; transport handled by operator-split).
"""

from .chemistry_only import chemistry_only_newton_step
from .coupled import newton_implicit_step
from .operator_split import operator_split_advance, operator_split_step
from .result import (
    ConcentrationPostprocess,
    NewtonResult,
    SystemPostprocess,
    per_species_relative_change,
)

__all__ = [
    "NewtonResult",
    "SystemPostprocess",
    "ConcentrationPostprocess",
    "per_species_relative_change",
    "newton_implicit_step",
    "chemistry_only_newton_step",
    "operator_split_step",
    "operator_split_advance",
]
