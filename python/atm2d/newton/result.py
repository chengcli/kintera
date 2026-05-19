"""Shared types for the Newton family of solvers.

The KB Titan driver and tests refer to these as
``kintera.atm2d.NewtonResult`` / ``per_species_relative_change``; this
module is where they live after the Phase 4b split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from ..matrix import SparseSystemMatrix


class SystemPostprocess(Protocol):
    def __call__(
        self,
        system: SparseSystemMatrix,
        rhs: torch.Tensor,
    ) -> tuple[SparseSystemMatrix, torch.Tensor]: ...


class ConcentrationPostprocess(Protocol):
    def __call__(self, concentration: torch.Tensor) -> torch.Tensor: ...


@dataclass
class NewtonResult:
    concentration: torch.Tensor
    converged: bool
    iterations: int
    max_relative_change: float
    residual_history: list[float] = field(default_factory=list)


def per_species_relative_change(
    new_conc: torch.Tensor,
    old_conc: torch.Tensor,
    *,
    species_scale_floor: float = 1.0,
) -> float:
    """Return ``max | new − old | / max(|old|, floor)`` over all species.

    Same family of fractional-change check KINETICS-base uses in
    ``CONVRG``: a Newton iterate is considered converged when the
    per-species relative change drops below the tolerance. The floor
    prevents trace species near zero from dominating the test.
    """
    diff = (new_conc - old_conc).abs()
    scale = old_conc.abs().clamp(min=species_scale_floor)
    return diff.div(scale).amax().item()


__all__ = [
    "NewtonResult",
    "SystemPostprocess",
    "ConcentrationPostprocess",
    "per_species_relative_change",
]
