from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .atm2d import AtmState2D
from .kintera import parse_kinetics_base_atmosphere, parse_kinetics_base_pun


@dataclass
class KBTitanState:
    species: list[str]
    fixed_species: list[str]
    varying_species: list[str]
    conversion: dict[str, str]
    concentration: torch.Tensor
    kzz: torch.Tensor
    state: AtmState2D


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
        if torch.max(torch.abs(column)).item() <= 1.0:
            concentration[:, j] = column * density[:, 0]
            conversion[name] = "mixing_ratio_times_density"
        else:
            conversion[name] = "number_density"

    if boundary_path is not None:
        _apply_lower_mixing_ratio_boundaries(
            concentration, conversion, density[:, 0], species, boundary_path
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
        kzz=torch.tensor(atmosphere.eddy_diffusion).view(1, -1),
        state=state,
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
        return name.endswith("*")
    return (
        name.endswith("*")
        or not any(value != 0 for value in metadata.composition)
        or (
            fixed_species is not None
            and name in fixed_species
            and metadata.molecular_weight <= 0.0
        )
    )


def _concentration_profile_reason(
    name: str,
    pun_metadata: dict[str, Any] | None,
    fixed_species: set[str] | None,
) -> str:
    metadata = None if pun_metadata is None else pun_metadata.get(name)
    if name.endswith("*"):
        return "pun_star_species_number_density"
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


def _apply_lower_mixing_ratio_boundaries(
    concentration: torch.Tensor,
    conversion: dict[str, str],
    density: torch.Tensor,
    species: list[str],
    boundary_path: str | Path,
) -> None:
    species_index = {name: i for i, name in enumerate(species)}
    for kind, value, name in _read_kinetics_base_boundary_file(boundary_path):
        normalized = _normalize_kinetics_base_boundary_species(name)
        if kind != 5 or normalized not in species_index:
            continue
        j = species_index[normalized]
        concentration[0, j] = value * density[0]
        conversion[normalized] = "lower_boundary_mixing_ratio_times_density"


def _read_kinetics_base_boundary_file(
    boundary_path: str | Path,
) -> list[tuple[int, float, str]]:
    entries: list[tuple[int, float, str]] = []
    for line in Path(boundary_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            lower_kind = int(parts[0])
            lower_value = float(parts[1])
        except ValueError:
            continue
        entries.append((lower_kind, lower_value, parts[4]))
    return entries


def _normalize_kinetics_base_boundary_species(name: str) -> str:
    if len(name) > 1 and name[0].isdigit():
        return f"({name[0]}){name[1:]}"
    return name


__all__ = [
    "KBTitanState",
    "altitude_faces_from_kinetics_base_centers_km",
    "build_kinetics_base_titan_state",
    "kinetics_base_concentration_from_profile",
    "kinetics_base_profile_tensor",
    "kinetics_base_species_metadata_from_pun",
]
