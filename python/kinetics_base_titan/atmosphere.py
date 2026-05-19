from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..atm2d import AtmState2D, SparseSystemMatrix
from ..atm2d.matrix import flatten_state_index
from ..kintera import parse_kinetics_base_atmosphere, parse_kinetics_base_pun
from .models import KBTitanState
from .parsing import _apply_cheng_cold_trap_boundaries, _apply_lower_mixing_ratio_boundaries


def kinetics_base_profile_tensor(profile: Any, species: list[str]) -> torch.Tensor:
    rows = [
        [profile.species_profiles[name][i] for name in species]
        for i in range(len(profile.altitude))
    ]
    return torch.tensor(rows)

def altitude_faces_from_kinetics_base_centers_km(altitude: list[float]) -> torch.Tensor:
    centers = torch.tensor(altitude) * 1.0e5
    faces = torch.empty(centers.numel() + 1)
    faces[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    faces[0] = centers[0] - (faces[1] - centers[0])
    faces[-1] = centers[-1] + (centers[-1] - faces[-2])
    if faces[0] < 0:
        faces[0] = 0.0
    return faces

def kinetics_base_concentration_from_profile(
    profile: Any,
    species: list[str],
    *,
    boundary_path: str | Path | None = None,
    pun_metadata: dict[str, Any] | None = None,
    fixed_species: set[str] | None = None,
) -> tuple[torch.Tensor, dict[str, str]]:
    concentration = kinetics_base_profile_tensor(profile, species)
    density = torch.tensor(profile.density).view(-1, 1)
    conversion: dict[str, str] = {}

    for j, name in enumerate(species):
        column = concentration[:, j]
        if _is_kinetics_base_concentration_profile(
            name, pun_metadata, fixed_species
        ):
            conversion[name] = _concentration_profile_reason(
                name, pun_metadata, fixed_species
            )
            continue
        if _is_kinetics_base_mixing_ratio_profile(profile, name):
            concentration[:, j] = column * density[:, 0]
            conversion[name] = "profile_marker_mixing_ratio_times_density"
            continue
        conversion[name] = "number_density"

    if boundary_path is not None:
        _apply_lower_mixing_ratio_boundaries(
            concentration, conversion, density[:, 0], species, boundary_path
        )

    # Apply KINETICS-base __CHENG cold trap: CH4 at and below level 24
    # (0-indexed 23) is pinned to the prepared atmosphere/boundary values.
    # This must come after boundary_path processing so the CH4 pin metadata
    # survives the surface boundary conversion.
    #
    # Pass the atm-file mixing-ratio profile so the cold trap can restore
    # CH4 below the trap. The bc_save lower BC for CH4 (mixing-ratio
    # 4e-4) is incorrect — KB uses the atm-file value (4e-3). Restoring
    # from the profile fixes the off-by-10× initial CH4 at lev 0–22.
    profile_values: dict[str, torch.Tensor] = {}
    for j, name in enumerate(species):
        if name.lower() == "ch4":
            # Use raw profile (mixing ratio) values, not the density-multiplied
            # tensor that's already been modified by the lower-BC pass.
            raw = kinetics_base_profile_tensor(profile, [name])[:, 0]
            profile_values[name] = raw
            break
    _apply_cheng_cold_trap_boundaries(
        concentration, conversion, density[:, 0], species, profile_values=profile_values
    )

    zero_density = density[:, 0] == 0
    if torch.any(zero_density):
        concentration[zero_density, :] = 0.0

    return concentration, conversion

def build_kinetics_base_titan_state(
    atmosphere: str | Path | Any,
    *,
    species: list[str] | None = None,
    fixed_species: list[str] | None = None,
    boundary_path: str | Path | None = None,
    pun_path: str | Path | None = None,
    pun_metadata: dict[str, Any] | None = None,
) -> KBTitanState:
    if isinstance(atmosphere, (str, Path)):
        atmosphere = parse_kinetics_base_atmosphere(str(atmosphere))

    selected_species = species or list(atmosphere.species_profiles.keys())
    fixed = fixed_species or [
        name
        for name in ["JDUST", "N2", "E", "PROD", "U", "RAYEAR", "SGA", "M"]
        if name in selected_species
    ]
    fixed_set = set(fixed)
    varying = [name for name in selected_species if name not in fixed_set]
    if pun_metadata is None and pun_path is not None:
        pun_metadata = kinetics_base_species_metadata_from_pun(pun_path)

    concentration_2d, conversion = kinetics_base_concentration_from_profile(
        atmosphere,
        selected_species,
        boundary_path=boundary_path,
        pun_metadata=pun_metadata,
        fixed_species=fixed_set,
    )
    concentration = concentration_2d.view(1, len(atmosphere.altitude), len(selected_species))
    state = AtmState2D(
        x1f=altitude_faces_from_kinetics_base_centers_km(atmosphere.altitude),
        x2f=torch.tensor([0.0, 1.0]),
        temperature=torch.tensor(atmosphere.temperature).view(1, -1),
        pressure=torch.tensor(atmosphere.pressure).view(1, -1),
        concentration=concentration,
    )

    return KBTitanState(
        species=selected_species,
        fixed_species=fixed,
        varying_species=varying,
        conversion=conversion,
        concentration=concentration,
        density=torch.tensor(atmosphere.density).view(1, -1),
        kzz=torch.tensor(atmosphere.eddy_diffusion).view(1, -1),
        state=state,
    )


def kinetics_base_titan_species_diffusion_scale(
    species: list[str],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return species-wise eddy diffusion scales for the Cheng Titan network."""
    return torch.ones(len(species), dtype=dtype or torch.get_default_dtype(), device=device)


def apply_kinetics_base_titan_boundary_pins(
    concentration: torch.Tensor,
    titan_state: KBTitanState,
) -> torch.Tensor:
    """Re-apply KINETICS-base Titan fixed/boundary constraints after a solve step.

    KINETICS-base's ``__CHENG`` branch fixes the CH4 cold-trap boundary row, but
    lower atmospheric levels still evolve over long integrations.

    Also enforces charge neutrality: ``E = Σ(cations) − Σ(anions)`` per cell.
    KB treats E as one of its NFIX species but its .pun output shows
    E ≈ Σ(cations) within a few percent, so KB must be recomputing E from
    charge balance each step (rather than freezing it at the initial-atm
    value of 0). Without this, kintera's dissociative-recombination
    reactions ``X+ + E → products`` see rate = 0 (since pinned E = 0) and
    cations accumulate by 4–10× at the photoionization altitude, then
    cascade through proton-transfer reactions to multi-OoM excess at lower
    altitudes — confirmed by diagnostic_tools/rate_diff.
    """
    mask, values = kinetics_base_titan_boundary_pin_mask(titan_state)
    mask = mask.to(device=concentration.device)
    values = values.to(dtype=concentration.dtype, device=concentration.device)
    concentration[mask] = values[mask]

    species = titan_state.species
    if "E" in species:
        e_idx = species.index("E")
        pos_indices = [j for j, n in enumerate(species) if n.endswith("+")]
        neg_indices = [
            j for j, n in enumerate(species)
            if n.endswith("-") and n != "E"
        ]
        if pos_indices:
            pos_sum = concentration[:, :, pos_indices].sum(dim=-1)
        else:
            pos_sum = torch.zeros_like(concentration[:, :, 0])
        if neg_indices:
            neg_sum = concentration[:, :, neg_indices].sum(dim=-1)
        else:
            neg_sum = torch.zeros_like(concentration[:, :, 0])
        concentration[:, :, e_idx] = torch.clamp(pos_sum - neg_sum, min=0.0)
    return concentration


def kinetics_base_titan_boundary_pin_mask(
    titan_state: KBTitanState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-value Titan cells matching KINETICS-base boundary semantics."""

    lower_boundary_conversions = {
        "lower_boundary_mixing_ratio_times_density",
        "lower_boundary_deposition_velocity_zero",
    }
    upper_boundary_conversions = {
        "upper_boundary_escape_velocity_zero",
    }
    cold_trap_conversions = {
        "kinetics_base_cheng_cold_trap_mixing_ratio",
    }
    mask = torch.zeros_like(titan_state.concentration, dtype=torch.bool)
    values = torch.zeros_like(titan_state.concentration)

    fixed_species = [
        titan_state.species.index(name)
        for name in titan_state.fixed_species
        if name in titan_state.species
    ]
    if fixed_species:
        mask[:, :, fixed_species] = True
        values[:, :, fixed_species] = titan_state.concentration[:, :, fixed_species]

    lower_boundary_species = [
        titan_state.species.index(name)
        for name, conversion in titan_state.conversion.items()
        if conversion in lower_boundary_conversions
    ]
    if lower_boundary_species:
        mask[:, 0, lower_boundary_species] = True
        values[:, 0, lower_boundary_species] = titan_state.concentration[
            :, 0, lower_boundary_species
        ]

    upper_boundary_species = [
        titan_state.species.index(name)
        for name, conversion in titan_state.conversion.items()
        if conversion in upper_boundary_conversions
    ]
    if upper_boundary_species:
        nonzero_levels = (titan_state.density[0] > 0).nonzero(as_tuple=True)[0]
        last_real_lyr = (
            int(nonzero_levels[-1])
            if nonzero_levels.numel() > 0
            else titan_state.state.nlyr - 1
        )
        mask[:, last_real_lyr, upper_boundary_species] = True
        values[:, last_real_lyr, upper_boundary_species] = 0.0

    cold_trap_species = [
        titan_state.species.index(name)
        for name, conversion in titan_state.conversion.items()
        if conversion in cold_trap_conversions
    ]
    if cold_trap_species:
        # Cheng cold trap pins CH4 to the atm-file mixing-ratio profile at
        # lev 0–23 (NBOT=24 in 1-indexed Fortran). Above the cold trap,
        # KB's converged result still tracks the atm-file CH4 profile
        # within 2–3% at lev 24–39: KB's vertical mixing of CH4 against
        # photolysis destruction reaches a quasi-static profile that's
        # essentially the initial atm. In our operator-split, chemistry
        # at dt=1e+9 destroys CH4 faster than transport can refill,
        # collapsing CH4 to ~zero at lev 28+. That cuts the dominant
        # CH2+/CH3+ loss channel (X+ + CH4 → products), forcing those
        # cations to recombine with E instead, which produces bare C
        # at 1e+6× KB rate — the upstream of the C+ runaway.
        # Pin CH4 across the full atmosphere to match KB's behavior.
        mask[:, :, cold_trap_species] = True
        values[:, :, cold_trap_species] = titan_state.concentration[
            :, :, cold_trap_species
        ]

    return mask, values


def apply_kinetics_base_titan_dirichlet_rows(
    system: SparseSystemMatrix,
    rhs: torch.Tensor,
    titan_state: KBTitanState,
) -> tuple[SparseSystemMatrix, torch.Tensor]:
    """Enforce Titan fixed/boundary cells as Dirichlet rows in an implicit system."""

    mask, values = kinetics_base_titan_boundary_pin_mask(titan_state)
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
    unit_values = torch.ones(row_ids.numel(), dtype=system.dtype, device=system.device)
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

def kinetics_base_species_metadata_from_pun(pun: str | Path | Any) -> dict[str, Any]:
    if isinstance(pun, (str, Path)):
        pun = parse_kinetics_base_pun(str(pun))
    return {species.name: species for species in pun.species}

def _is_kinetics_base_concentration_profile(
    name: str,
    pun_metadata: dict[str, Any] | None,
    fixed_species: set[str] | None,
) -> bool:
    metadata = None if pun_metadata is None else pun_metadata.get(name)
    if metadata is None:
        return name.endswith("*") or _is_kinetics_base_charged_species_name(name)
    return (
        name.endswith("*")
        or _has_kinetics_base_electron_composition(metadata)
        or not any(value != 0 for value in metadata.composition)
        or (
            fixed_species is not None
            and name in fixed_species
            and metadata.molecular_weight <= 0.0
        )
    )

def _is_kinetics_base_mixing_ratio_profile(profile: Any, name: str) -> bool:
    mixing_ratio_species = getattr(profile, "mixing_ratio_species_profiles", None)
    if mixing_ratio_species is None:
        return False
    return name in mixing_ratio_species

def _concentration_profile_reason(
    name: str,
    pun_metadata: dict[str, Any] | None,
    fixed_species: set[str] | None,
) -> str:
    metadata = None if pun_metadata is None else pun_metadata.get(name)
    if name.endswith("*"):
        return "pun_star_species_number_density"
    if metadata is not None and _has_kinetics_base_electron_composition(metadata):
        return "pun_electron_or_ion_number_density"
    if metadata is None and _is_kinetics_base_charged_species_name(name):
        return "charged_species_number_density"
    if metadata is not None and not any(value != 0 for value in metadata.composition):
        return "pun_empty_composition_number_density"
    if (
        metadata is not None
        and fixed_species is not None
        and name in fixed_species
        and metadata.molecular_weight <= 0.0
    ):
        return "fixed_pun_zero_molecular_weight_number_density"
    return "number_density"

def _has_kinetics_base_electron_composition(metadata: Any) -> bool:
    # KINETICS .pun files encode charge using the electron pseudo-element.
    # In the Titan network this is the final composition slot: E has +1,
    # cations have -1, and neutral species have 0.
    composition = getattr(metadata, "composition", [])
    return bool(composition) and composition[-1] != 0

def _is_kinetics_base_charged_species_name(name: str) -> bool:
    return name == "E" or name.endswith("+") or name.endswith("-")



