from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...atm2d import AtmState2D


@dataclass
class KBTitanState:
    species: list[str]
    fixed_species: list[str]
    varying_species: list[str]
    conversion: dict[str, str]
    concentration: torch.Tensor
    density: torch.Tensor
    kzz: torch.Tensor
    state: AtmState2D

@dataclass
class KBTitanSpecialEntry:
    index: int
    kind: int
    target_id: int
    comment: str

@dataclass
class KBTitanSpecialIndex:
    """Lookup table for KINETICS-base's ISP special-code indices."""

    entries: list[KBTitanSpecialEntry]

    @property
    def by_index(self) -> dict[int, KBTitanSpecialEntry]:
        return {entry.index: entry for entry in self.entries}

    def target_id(self, index: int, *, kind: int | None = None) -> int | None:
        entry = self.by_index.get(index)
        if entry is None:
            return None
        if kind is not None and entry.kind != kind:
            return None
        return entry.target_id

    def target_ids(self, indices: set[int], *, kind: int | None = None) -> set[int]:
        targets: set[int] = set()
        for index in indices:
            target = self.target_id(index, kind=kind)
            if target is not None:
                targets.add(target)
        return targets

    def targets_for_kind(self, kind: int) -> set[int]:
        return {entry.target_id for entry in self.entries if entry.kind == kind}

@dataclass
class KBTitanSourceTerm:
    kind: str
    reaction_id: int | None
    reactants: list[str]
    products: list[str]
    parameters: dict[str, Any]

@dataclass
class KBTitanActiveNetwork:
    species_mapping: dict[int, int]
    reaction_mapping: dict[int, int]

    @property
    def active_reaction_ids(self) -> set[int]:
        return {
            reaction_id
            for reaction_id, operational_id in self.reaction_mapping.items()
            if operational_id != 0
        }

@dataclass
class KBTitanBoundaryEntry:
    lower_kind: int
    lower_value: float
    upper_kind: int
    upper_value: float
    species: str

