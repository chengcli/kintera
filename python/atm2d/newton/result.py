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
    layer_relative_floor: float = 0.0,
) -> float:
    """Return ``max | new − old | / scale`` over all species.

    Same family of fractional-change check KINETICS-base uses in
    ``CONVRG``: a Newton iterate is considered converged when the
    per-species relative change drops below the tolerance.

    ``scale`` is ``|old|`` floored from below, and the floor is what makes
    this usable. Without one, a species passing through zero produces an
    unbounded ratio and pins the maximum forever, so the solver reports
    non-convergence on a step its chemistry has already solved.

    Two floors are available and the larger applies:

    ``species_scale_floor``
        An absolute concentration. Adequate only when the column spans a
        narrow density range.
    ``layer_relative_floor``
        A fraction of the most abundant species *in the same layer*. Use
        this on columns spanning many density decades, where no single
        absolute floor can work: on an HD189733b column running from
        ~1e22 cm^-3 at 1000 bar to ~1e10 cm^-3 at the top, a floor loose
        enough to ignore numerical dust at depth silently swallows real
        chemistry at the top, and one tight enough at the top leaves the
        deep layers pinned by species at a 1e-16 mixing ratio. Measured
        there: an absolute 1e6 cm^-3 floor fails to converge at *every*
        step size down to 1e-8 s (100 iterations), while flooring at
        1e-14 of each layer's maximum converges in two.
    """
    diff = (new_conc - old_conc).abs()
    scale = old_conc.abs()
    if layer_relative_floor > 0.0:
        # Species is the last axis; take the per-layer maximum over it.
        layer_max = scale.amax(dim=-1, keepdim=True)
        scale = torch.maximum(scale, layer_relative_floor * layer_max)
    return diff.div(scale.clamp(min=species_scale_floor)).amax().item()


__all__ = [
    "NewtonResult",
    "SystemPostprocess",
    "ConcentrationPostprocess",
    "per_species_relative_change",
]
