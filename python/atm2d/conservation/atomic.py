"""Per-cell, per-element atomic-budget projection.

Background
----------
Backward-Euler chemistry Newton at very large dt (≥1e+8 s) can land on
non-physical fixed-points where trace species accumulate to 10–1000× the
local atomic budget. KB's `MARCH`/`CONVRG` clips this implicitly via its
acceptance rule; we apply an explicit per-cell projection between Newton
solves.

Strategy: per element, per cell:
  1. Sum current variable-species atoms of that element.
  2. If the sum exceeds ``budget``, identify "hog" species (those whose
     atom contribution is above the per-species mean).
  3. Subtract the excess from the hogs proportionally to how much they
     contribute, leaving non-hog (slow-chemistry) species untouched.

This preserves slow chemistry chains (which stay below the mean) while
clipping the runaway species that eat the atomic budget.

The function is pure — caller supplies ``atom_counts`` (per-species
integer counts for each element of interest), ``fixed_mask`` (per-species
bool flagging "do not touch"), and ``budget`` (per-cell allowed total).
"""

from __future__ import annotations

from typing import Mapping

import torch


def count_atoms(species_name: str, element: str) -> int:
    """Count occurrences of ``element`` in a species formula like ``"CH4"``,
    ``"C2H6"``, ``"NH4+"``, ``"(1)CH2"``, ``"c-C3H3+"``.

    Handles:
      * Parenthesized state labels — ``"(1)CH2"``, ``"(3)CH2"``, ``"N(2D)"`` —
        by skipping the parenthesized region entirely.
      * Trailing ``+`` / ``-`` / ``*`` charge / excited markers.
      * Two-letter element names by checking the next character for lowercase
        (so ``"Cl"`` does not match ``"C"``).

    Returns the integer count.
    """
    name = species_name.rstrip("+-*")
    cnt = 0
    i = 0
    while i < len(name):
        ch = name[i]
        if ch == "(":
            while i < len(name) and name[i] != ")":
                i += 1
            i += 1
            continue
        # Match element at this position (single-letter; reject 2-letter coincidences)
        if name[i:i + len(element)] == element:
            end = i + len(element)
            # Reject if followed by lowercase (e.g. "Cl" when looking for "C").
            if end < len(name) and name[end].islower():
                i += 1
                continue
            if end < len(name) and name[end].isdigit():
                # Match KB Titan diagnostic driver: read single trailing
                # digit only. Multi-digit subscripts (e.g. ``"C4H10"``)
                # are intentionally truncated to preserve bit-identical
                # behavior with the pre-refactor diagnostic — fixing
                # this is a separate physics PR after the seam settles.
                cnt += int(name[end])
                i = end + 1
            else:
                cnt += 1
                i = end
        else:
            i += 1
    return cnt


def project_atomic_budget(
    concentration: torch.Tensor,
    *,
    atom_counts: Mapping[str, torch.Tensor],
    fixed_mask: torch.Tensor,
    budget: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Project ``concentration`` onto a per-cell, per-element atomic budget.

    Parameters
    ----------
    concentration:
        Tensor shaped ``(ncol, nlyr, nspecies)``. Returned tensor has the same
        shape.
    atom_counts:
        Mapping ``element -> (nspecies,)`` integer-valued tensor giving how
        many atoms of that element each species contains.
    fixed_mask:
        ``(nspecies,)`` bool tensor flagging species that must not be
        modified (e.g. ``N2``, ``M``, ``E`` in KB Titan).
    budget:
        Mapping ``element -> (ncol, nlyr, 1)`` tensor giving the per-cell
        allowed total atoms-of-that-element. Typically computed once from
        initial conditions + a chemistry headroom.

    Returns
    -------
    Tensor with the same shape as ``concentration``, with hog-species
    contributions trimmed so that each element's per-cell atom sum lies
    within ``budget[element]``.
    """
    c = concentration
    for elem, counts in atom_counts.items():
        mask = (counts > 0) & (~fixed_mask)
        if not mask.any():
            continue
        counts_view = counts.view(1, 1, -1)
        # Per-cell current atoms vs budget.
        cur = (c * counts_view * mask).sum(dim=-1, keepdim=True)
        budget_elem = budget[elem]
        excess = (cur - budget_elem).clamp(min=0.0)
        if not (excess > 0).any():
            continue
        # Hog species: atom contribution > per-species mean.
        per_species_atoms = c * counts_view * mask
        n_species_with_elem = mask.sum().clamp(min=1.0).item()
        mean_atoms = per_species_atoms.sum(dim=-1, keepdim=True) / n_species_with_elem
        hog_mask = per_species_atoms > mean_atoms
        hog_atoms = torch.where(
            hog_mask, per_species_atoms, torch.zeros_like(per_species_atoms)
        )
        hog_total = hog_atoms.sum(dim=-1, keepdim=True).clamp(min=1.0)
        hog_fraction = hog_atoms / hog_total
        atoms_to_subtract = hog_fraction * excess
        counts_safe = counts_view.clamp(min=1.0)
        delta_c = atoms_to_subtract / counts_safe
        apply_mask = hog_mask & (excess > 0).expand_as(hog_mask)
        c = torch.where(apply_mask, (c - delta_c).clamp(min=0.0), c)
    return c
