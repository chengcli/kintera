from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...atm2d import AtmState2D, SparseSystemMatrix
from ...atm2d.matrix import flatten_state_index
from ...kintera import parse_kinetics_base_atmosphere, parse_kinetics_base_pun
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


def _titan_pin_specs(titan_state: KBTitanState) -> list:
    """Build the list of :class:`BoundaryPinSpec` describing KB Titan's
    boundary semantics.

    Four pin sources:

    1. **Fixed species** (KB ``NFIX``: ``JDUST``, ``N2``, ``E``, ``PROD``,
       ``U``, ``RAYEAR``, ``SGA``, ``M``). All levels pinned to the initial
       concentration.
    2. **Lower-boundary mixing-ratio / deposition species** (KB ``type=5``
       and ``type=2`` lower BC). Lev 0 pinned.
    3. **Upper-boundary escape species** (KB ``type=2`` positive upper BC).
       Last real lev pinned to 0 (escape sink).
    4. **Cheng cold-trap species** (CH4 in the Cheng branch). All levels
       pinned to the atm-file profile (G19+G26).
    """
    from ...atm2d.pins import BoundaryPinSpec

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

    specs: list[BoundaryPinSpec] = []
    species = titan_state.species
    c0 = titan_state.concentration

    fixed_indices = [
        species.index(name)
        for name in titan_state.fixed_species
        if name in species
    ]
    if fixed_indices:
        specs.append(
            BoundaryPinSpec(
                species_indices=fixed_indices,
                level_indices=None,
                values=c0,
            )
        )

    lower_boundary_indices = [
        species.index(name)
        for name, conv in titan_state.conversion.items()
        if conv in lower_boundary_conversions
    ]
    if lower_boundary_indices:
        specs.append(
            BoundaryPinSpec(
                species_indices=lower_boundary_indices,
                level_indices=[0],
                values=c0,
            )
        )

    upper_boundary_indices = [
        species.index(name)
        for name, conv in titan_state.conversion.items()
        if conv in upper_boundary_conversions
    ]
    if upper_boundary_indices:
        nonzero_levels = (titan_state.density[0] > 0).nonzero(as_tuple=True)[0]
        last_real_lyr = (
            int(nonzero_levels[-1])
            if nonzero_levels.numel() > 0
            else titan_state.state.nlyr - 1
        )
        # Pin upper-boundary species at last_real_lyr to 0 (escape sink).
        zeros = torch.zeros_like(c0)
        specs.append(
            BoundaryPinSpec(
                species_indices=upper_boundary_indices,
                level_indices=[last_real_lyr],
                values=zeros,
            )
        )

    cold_trap_indices = [
        species.index(name)
        for name, conv in titan_state.conversion.items()
        if conv in cold_trap_conversions
    ]
    if cold_trap_indices:
        # G19+G26: Cheng cold trap pins CH4 to the atm-file profile across
        # all levels. KB's converged CH4 profile tracks the atm-file values
        # within 2-3% from lev 0 through lev 39; without this pin our
        # chemistry destroys CH4 above the cold trap and triggers the
        # bare-C cascade that produces C+ runaway. See GAP_STATUS §G19/G26.
        specs.append(
            BoundaryPinSpec(
                species_indices=cold_trap_indices,
                level_indices=None,
                values=c0,
            )
        )

    return specs


def _titan_charge_balance_indices(species: list[str]) -> "tuple[list[int], list[int], int] | None":
    """Return ``(cation_indices, anion_indices, e_index)`` if ``E`` is in
    ``species``; otherwise ``None``. Cations end with ``+``; anions end
    with ``-`` excluding ``E`` itself."""
    if "E" not in species:
        return None
    e_index = species.index("E")
    cation_indices = [j for j, n in enumerate(species) if n.endswith("+")]
    anion_indices = [
        j for j, n in enumerate(species)
        if n.endswith("-") and n != "E"
    ]
    return cation_indices, anion_indices, e_index


def apply_kinetics_base_titan_boundary_pins(
    concentration: torch.Tensor,
    titan_state: KBTitanState,
) -> torch.Tensor:
    """Re-apply KINETICS-base Titan fixed/boundary constraints after a solve step.

    Generic pin assembly + charge-balance reset, parameterised by KB
    Titan's boundary conventions. Internally builds a list of
    :class:`atm2d.pins.BoundaryPinSpec` and delegates to the generic
    ``apply_pin_mask_to_concentration`` + ``recompute_charge_balance_e``.

    KINETICS-base's ``__CHENG`` branch fixes the CH4 cold-trap boundary
    row, but lower atmospheric levels still evolve over long
    integrations. We also enforce charge neutrality: KB treats ``E`` as
    one of its NFIX species but its .pun output shows ``E ≈ Σ(cations)``
    within a few percent, so we recompute ``E`` from the cation sum.
    """
    from ...atm2d.pins import (
        apply_pin_mask_to_concentration,
        build_pin_mask,
        recompute_charge_balance_e,
    )

    specs = _titan_pin_specs(titan_state)
    mask, values = build_pin_mask(titan_state.state, specs)
    mask = mask.to(device=concentration.device)
    values = values.to(dtype=concentration.dtype, device=concentration.device)
    concentration = apply_pin_mask_to_concentration(concentration, mask, values)

    indices = _titan_charge_balance_indices(titan_state.species)
    if indices is not None:
        cation_indices, anion_indices, e_index = indices
        concentration = recompute_charge_balance_e(
            concentration,
            cation_indices=cation_indices,
            anion_indices=anion_indices,
            e_index=e_index,
        )
    return concentration


def kinetics_base_titan_boundary_pin_mask(
    titan_state: KBTitanState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-value Titan cells matching KINETICS-base boundary semantics.

    Thin wrapper that builds a Titan-specific list of
    :class:`atm2d.pins.BoundaryPinSpec` and assembles them via the
    generic :func:`atm2d.pins.build_pin_mask`.
    """
    from ...atm2d.pins import build_pin_mask

    specs = _titan_pin_specs(titan_state)
    return build_pin_mask(titan_state.state, specs)


def apply_kinetics_base_titan_dirichlet_rows(
    system: SparseSystemMatrix,
    rhs: torch.Tensor,
    titan_state: KBTitanState,
) -> tuple[SparseSystemMatrix, torch.Tensor]:
    """Enforce Titan fixed/boundary cells as Dirichlet rows in an implicit system.

    Thin wrapper around :func:`atm2d.pins.apply_pin_mask_as_dirichlet_rows`.
    """
    from ...atm2d.pins import apply_pin_mask_as_dirichlet_rows

    mask, values = kinetics_base_titan_boundary_pin_mask(titan_state)
    return apply_pin_mask_as_dirichlet_rows(system, rhs, mask, values)

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



