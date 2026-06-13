"""Core-engine configuration object (transport form + chemistry solver).

Consolidates the scattered ``KINTERA_*`` environment-variable switches for the
core atm2d engine into a single dataclass. The library reads configuration
through :func:`get_core_config`; by default it loads fresh from the environment
on each access (preserving the historical env-at-call-time behaviour), or an
explicit :class:`CoreConfig` can be installed via :func:`set_core_config`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class CoreConfig:
    """Core transport + chemistry-solver configuration.

    Attributes
    ----------
    transport_form:
        Vertical transport discretization (``"c_diffusion"``,
        ``"mr_diffusion"``, ``"mr_exp"``, ``"mr_hybrid"``). ``None`` selects the
        density-aware default (mr when a density field is supplied, else c).
    chem_solver:
        Chemistry solver implementation (``"newton"``, ``"bdf"``, ``"lsoda"``,
        ``"radau"``).
    """

    transport_form: str | None = None
    chem_solver: str = "newton"

    @classmethod
    def from_env(cls) -> "CoreConfig":
        """Build a CoreConfig from ``KINTERA_*`` environment variables."""
        return cls(
            transport_form=os.environ.get("KINTERA_TRANSPORT_FORM"),
            chem_solver=os.environ.get("KINTERA_CHEM_SOLVER", "newton").lower(),
        )


_active: CoreConfig | None = None


def set_core_config(config: CoreConfig | None) -> None:
    """Install an explicit CoreConfig (``None`` restores env-driven loading)."""
    global _active
    _active = config


def get_core_config() -> CoreConfig:
    """Return the active CoreConfig, or one loaded fresh from the environment."""
    return _active if _active is not None else CoreConfig.from_env()
