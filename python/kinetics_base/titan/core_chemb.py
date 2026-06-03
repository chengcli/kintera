"""Titan UPDATE_CHEMB override layer for the core-engine pipeline.

KB's `UPDATE_CHEMB` replaces the `.pun` catalog rate constant for ~24 Titan
reactions with bespoke piecewise / variable-Fc / `rk2 - zk*density` formulas
(see :mod:`chemb_overrides`). These are genuinely *special* (not expressible as
core Arrhenius/falloff parameters), so under the `unify-titan-chem-onto-core`
refactor they live outside the core engine: the C++ translator
(`KineticsOptions.from_kinetics_base_pun`) builds the catalog rate for every
thermal reaction, and this thin Titan layer overrides the matched reactions'
rate constants on top of the core `Kinetics.forward` output.

Overrides are matched by canonical reaction *signature* (sorted reactant /
product names, including ``M``), so the layer is network-agnostic: it fires for
the right reactions regardless of the integer ids a given `.pun` network uses.
Provenance (which core reaction column each override replaces) is recorded.
"""
from __future__ import annotations

from typing import Any

import torch

from .chemb_overrides import (
    has_titan_chemb_override_by_signature,
    titan_chemb_rate_constant_by_signature,
)


def _reaction_sides(reaction: Any) -> tuple[list[str], list[str]]:
    """Expand a core `Reaction` into reactant/product name lists (coeff-expanded)."""
    reactants: list[str] = []
    products: list[str] = []
    for name, coeff in reaction.reactants().items():
        reactants.extend([name] * int(round(coeff)))
    for name, coeff in reaction.products().items():
        products.extend([name] * int(round(coeff)))
    return reactants, products


class ChembOverrideLayer:
    """Applies KB UPDATE_CHEMB rate-constant overrides to core forward output.

    Attributes:
        columns: core reaction-column indices that carry an override.
        provenance: per-override record {column, reactants, products, source}.
    """

    def __init__(
        self,
        columns: list[int],
        sides: list[tuple[list[str], list[str]]],
        provenance: list[dict],
    ) -> None:
        self.columns = columns
        self._sides = sides
        self.provenance = provenance

    def __len__(self) -> int:
        return len(self.columns)

    def override_rate_constants(
        self, temperature: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        """Override rate constants for the matched columns, shape (..., n_override)."""
        if not self.columns:
            return temperature.new_zeros(temperature.shape + (0,))
        cols = [
            titan_chemb_rate_constant_by_signature(reac, prod, temperature, density)
            for (reac, prod) in self._sides
        ]
        return torch.stack(cols, dim=-1)

    def apply(
        self,
        rate_constants: torch.Tensor,
        temperature: torch.Tensor,
        density: torch.Tensor,
    ) -> torch.Tensor:
        """Return `rate_constants` (..., nreaction) with overridden columns replaced.

        `temperature`/`density` broadcast to the leading dims of `rate_constants`.
        """
        if not self.columns:
            return rate_constants
        out = rate_constants.clone()
        overrides = self.override_rate_constants(temperature, density)
        for i, col in enumerate(self.columns):
            out[..., col] = overrides[..., i]
        return out


def build_chemb_override_layer(reactions: list[Any]) -> ChembOverrideLayer:
    """Scan the core reaction list (column order) for UPDATE_CHEMB overrides.

    `reactions` is `KineticsOptions.reactions()` — the same order the core
    `Kinetics.forward` rate columns follow.
    """
    columns: list[int] = []
    sides: list[tuple[list[str], list[str]]] = []
    provenance: list[dict] = []
    for col, reaction in enumerate(reactions):
        reactants, products = _reaction_sides(reaction)
        if has_titan_chemb_override_by_signature(reactants, products):
            columns.append(col)
            sides.append((reactants, products))
            provenance.append(
                {
                    "column": col,
                    "reactants": reactants,
                    "products": products,
                    "source": "UPDATE_CHEMB",
                }
            )
    return ChembOverrideLayer(columns, sides, provenance)
