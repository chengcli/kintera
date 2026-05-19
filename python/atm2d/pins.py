"""Generic boundary-pin API for ``AtmState2D``.

A *boundary pin* freezes selected ``(column, level, species)`` cells to
prescribed values. Pins are accumulated as a list of
:class:`BoundaryPinSpec` objects, then assembled into a single
``(mask, values)`` pair via :func:`build_pin_mask`. The result can be
applied either:

  * **post-Newton**, via :func:`apply_pin_mask_to_concentration`, to
    re-impose the constraint after each operator-split solve; or
  * **in-Newton**, via :func:`apply_pin_mask_as_dirichlet_rows`, to
    embed the constraint as Dirichlet rows of the implicit system.

Pins are how we encode planet/adapter-specific boundary semantics
(KB Titan cold-trap, mixing-ratio lower BC, escape-velocity upper BC)
into the otherwise-generic ``AtmState2D`` solver.

Charge balance — recomputing ``E = Σ(cations) − Σ(anions)`` per cell —
is exposed alongside the pin API because in practice it is always
applied next to the pins (KB's ``E`` is one of its NFIX species but
its converged profile matches the cation sum, so we recompute rather
than freeze it at the initial 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .atm_state2d import AtmState2D
from .matrix import SparseSystemMatrix, flatten_state_index


@dataclass(frozen=True)
class BoundaryPinSpec:
    """A single boundary pin.

    Pins selected ``(level, species)`` cells of the concentration array
    to prescribed values.

    Parameters
    ----------
    species_indices:
        Axis-2 species indices to pin. Empty list = no-op.
    level_indices:
        Axis-1 level indices to pin. ``None`` means pin every level.
    values:
        Tensor of pin values. Must broadcast to the slice
        ``concentration[:, level_indices, species_indices]``. The
        simplest pattern is to pass the full ``state.concentration``
        and let advanced indexing select the right cells inside
        :func:`build_pin_mask`.
    """

    species_indices: list[int]
    level_indices: list[int] | None
    values: torch.Tensor


def build_pin_mask(
    state: AtmState2D,
    specs: list[BoundaryPinSpec],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine ``specs`` into a single ``(mask, values)`` pair shaped
    like ``state.concentration``.

    Later specs override earlier ones at overlapping ``(column, level,
    species)`` cells — the iteration order in ``specs`` determines the
    final pinned values at any conflicting cell.
    """
    mask = torch.zeros_like(state.concentration, dtype=torch.bool)
    values = torch.zeros_like(state.concentration)
    for spec in specs:
        if not spec.species_indices:
            continue
        species_idx = torch.tensor(
            spec.species_indices, dtype=torch.long, device=mask.device
        )
        if spec.level_indices is None:
            mask[:, :, species_idx] = True
            # ``spec.values`` may be the full state.concentration array
            # or a slice. Slice the column-axis selection we just wrote.
            v = spec.values
            if v.shape == state.concentration.shape:
                values[:, :, species_idx] = v[:, :, species_idx]
            else:
                values[:, :, species_idx] = v
        else:
            level_idx = torch.tensor(
                spec.level_indices, dtype=torch.long, device=mask.device
            )
            # Use ix_-style outer-indexing into level × species.
            mesh_lvl, mesh_sp = torch.meshgrid(level_idx, species_idx, indexing="ij")
            mask[:, mesh_lvl, mesh_sp] = True
            v = spec.values
            if v.shape == state.concentration.shape:
                values[:, mesh_lvl, mesh_sp] = v[:, mesh_lvl, mesh_sp]
            else:
                values[:, mesh_lvl, mesh_sp] = v
    return mask, values


def apply_pin_mask_to_concentration(
    concentration: torch.Tensor,
    mask: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """In-place overwrite ``concentration`` cells where ``mask`` is
    true with the corresponding entries of ``values``. Returns the
    mutated tensor (for fluent chaining)."""
    concentration[mask] = values[mask]
    return concentration


def apply_pin_mask_as_dirichlet_rows(
    system: SparseSystemMatrix,
    rhs: torch.Tensor,
    mask: torch.Tensor,
    values: torch.Tensor,
) -> tuple[SparseSystemMatrix, torch.Tensor]:
    """Embed ``(mask, values)`` into ``system`` as Dirichlet rows.

    For every cell ``mask[c, l, s] == True``, the corresponding row of
    the flattened system is replaced by an identity row, and the rhs
    entry is set to ``values[c, l, s]``. Returns the modified system
    and the original ``rhs`` (the system's ``replace_rows`` carries
    the override mask internally so the caller does not need to mutate
    ``rhs`` here).
    """
    pinned = mask.nonzero(as_tuple=False)
    if pinned.numel() == 0:
        return system, rhs
    row_ids = flatten_state_index(
        pinned[:, 0],
        pinned[:, 1],
        pinned[:, 2],
        system.nlyr,
        system.nspecies,
    )
    unit_values = torch.ones(
        row_ids.numel(), dtype=system.dtype, device=system.device
    )
    return (
        system.replace_rows(
            row_ids,
            row_ids,
            unit_values,
            rhs_override_mask=mask.to(device=system.device),
            rhs_override_values=values.to(dtype=system.dtype, device=system.device),
        ),
        rhs,
    )


def recompute_charge_balance_e(
    concentration: torch.Tensor,
    *,
    cation_indices: list[int],
    anion_indices: list[int],
    e_index: int,
) -> torch.Tensor:
    """In-place overwrite ``concentration[..., e_index]`` with
    ``clamp(Σ(cations) − Σ(anions), min=0)``.

    Charge neutrality demands ``E + Σ(anions) = Σ(cations)``. KB's
    converged Titan output shows ``E ≈ Σ(cations)`` within a few
    percent, so we recompute ``E`` from the current cation/anion
    populations rather than freezing it at the initial-atm value of 0.

    Without this reset, dissociative-recombination reactions
    ``X⁺ + E → ...`` see ``rate = 0`` and cations accumulate by
    several OoM at the photoionization altitude.
    """
    if cation_indices:
        pos_sum = concentration[..., cation_indices].sum(dim=-1)
    else:
        pos_sum = torch.zeros_like(concentration[..., 0])
    if anion_indices:
        neg_sum = concentration[..., anion_indices].sum(dim=-1)
    else:
        neg_sum = torch.zeros_like(concentration[..., 0])
    concentration[..., e_index] = torch.clamp(pos_sum - neg_sum, min=0.0)
    return concentration


__all__ = [
    "BoundaryPinSpec",
    "build_pin_mask",
    "apply_pin_mask_to_concentration",
    "apply_pin_mask_as_dirichlet_rows",
    "recompute_charge_balance_e",
]
