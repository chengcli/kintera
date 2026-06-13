"""Titan-specific configuration (photo policy, chemb overrides, EI scales).

Consolidates the scattered ``KINTERA_TITAN_*`` / ``KINTERA_EI_*`` /
``KINTERA_DISABLE_CHEMB_OVERRIDES`` environment-variable switches for the Titan
layer into a single dataclass. The library reads configuration through
:func:`get_titan_config`; by default it loads fresh from the environment on each
access (preserving the historical env-at-call-time behaviour), or an explicit
:class:`TitanConfig` can be installed via :func:`set_titan_config`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Default electron-impact channel branching scales (Cravens-style); keyed by the
# historical env-var name so the loader and consumers agree.
_EI_DEFAULTS: dict[str, float] = {
    "KINTERA_EI_SCALE_N2_N2P": 4.15,
    "KINTERA_EI_SCALE_N2_NP": 117.0,
    "KINTERA_EI_SCALE_CH4_CH3P": 2.07,
    "KINTERA_EI_SCALE_CH4_CH2P": 2.61,
    "KINTERA_EI_SCALE_CH4_HP": 0.083,
    "KINTERA_EI_SCALE_OTHER_NP": 0.0035,
    "KINTERA_EI_SCALE_DEFAULT": 0.25,
}


@dataclass
class TitanConfig:
    """Titan photo policy + chemb-override + electron-impact configuration.

    Attributes
    ----------
    photo_allow_radicals:
        Activate photolysis for any single-reactant reaction with a
        cross-section file (KB's behaviour), not just the IPHOTO opacity list.
    photo_include_xscn:
        Include the Cheng ``_XSCN_`` (X-ray/EUV radical) cross-section files
        (excluded by default to match KB moses00).
    disable_chemb_overrides:
        Disable the KB ``UPDATE_CHEMB`` per-reaction rate overrides.
    ei_scale:
        Global electron-impact branching-ratio multiplier.
    ei_channel_scales:
        Per-channel electron-impact scales keyed by env-var name.
    """

    photo_allow_radicals: bool = False
    photo_include_xscn: bool = False
    disable_chemb_overrides: bool = False
    ei_scale: float = 1.0
    ei_channel_scales: dict[str, float] = field(
        default_factory=lambda: dict(_EI_DEFAULTS)
    )

    def ei_channel(self, name: str) -> float:
        """Return the per-channel EI scale for env-var ``name``."""
        return self.ei_channel_scales.get(name, _EI_DEFAULTS.get(name, 1.0))

    @classmethod
    def from_env(cls) -> "TitanConfig":
        """Build a TitanConfig from the Titan ``KINTERA_*`` environment vars."""
        scales = {
            name: float(os.environ.get(name, str(default)))
            for name, default in _EI_DEFAULTS.items()
        }
        return cls(
            photo_allow_radicals=bool(
                os.environ.get("KINTERA_TITAN_PHOTO_ALLOW_RADICALS")
            ),
            photo_include_xscn=bool(
                os.environ.get("KINTERA_TITAN_PHOTO_INCLUDE_XSCN")
            ),
            disable_chemb_overrides=bool(
                os.environ.get("KINTERA_DISABLE_CHEMB_OVERRIDES", "")
            ),
            ei_scale=float(os.environ.get("KINTERA_EI_SCALE", "1.0")),
            ei_channel_scales=scales,
        )


_active: TitanConfig | None = None


def set_titan_config(config: TitanConfig | None) -> None:
    """Install an explicit TitanConfig (``None`` restores env-driven loading)."""
    global _active
    _active = config


def get_titan_config() -> TitanConfig:
    """Return the active TitanConfig, or one loaded fresh from the environment."""
    return _active if _active is not None else TitanConfig.from_env()
