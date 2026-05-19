"""Linearized-source protocol shared by all atm2d source-term classes.

A *local source term* contributes a per-cell tendency
``dc/dt = S(c)`` and Jacobian ``dS/dc`` that can be assembled into the
implicit Newton system. Concrete implementations live in
:mod:`atm2d.sources.indexed` (built-in chemistry shapes) and in the
KB Titan adapter (:mod:`kintera.kinetics_base.titan.atm2d_sources`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from ..atm_state2d import AtmState2D


@dataclass
class LocalSourceLinearization:
    """Cell-local source tendency and Jacobian on an ``AtmState2D`` grid."""

    tendency: torch.Tensor
    jacobian: torch.Tensor


class LocalSourceTerm(Protocol):
    """Protocol for source terms that can be linearized cell-by-cell."""

    def linearize(self, state: AtmState2D) -> LocalSourceLinearization:
        """Return source tendency and d(source)/d(concentration)."""


RateProvider = Callable[[AtmState2D], torch.Tensor]


__all__ = ["LocalSourceLinearization", "LocalSourceTerm", "RateProvider"]
