from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..atm2d import AtmState2D


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
class KBTitanSourceTerm:
    kind: str
    reaction_id: int | None
    reactants: list[str]
    products: list[str]
    parameters: dict[str, Any]

@dataclass
class KBTitanBoundaryEntry:
    lower_kind: int
    lower_value: float
    upper_kind: int
    upper_value: float
    species: str

