from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .parsing import (
    _align_kinetics_base_flux_cross_section,
    _integrate_kinetics_base_photo_rate,
    _kinetics_base_reaction_key,
    _multiply_kinetics_base_cross_sections,
    _normalize_kinetics_base_photo_species,
    _parse_kinetics_base_catalog,
    _parse_kinetics_base_cross_section,
    _parse_kinetics_base_cross_section_on_flux,
    _parse_kinetics_base_flux,
)
from .radiation import (
    _kinetics_base_direct_actinic_flux,
    _kinetics_base_pyharp_actinic_flux,
)


def _kinetics_base_opacity_species_from_reaction_ids(
    reaction_ids: Any,
    reaction_by_id: dict[int, Any],
    species_by_id: dict[int, str],
) -> list[str] | None:
    if not isinstance(reaction_ids, list):
        return None
    opacity_species: list[str] = []
    seen: set[str] = set()
    for raw_id in reaction_ids:
        try:
            reaction_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        reaction = reaction_by_id.get(reaction_id)
        if reaction is None or len(reaction.reactant_ids) != 1:
            continue
        species = species_by_id.get(reaction.reactant_ids[0])
        if species is not None and species not in seen:
            seen.add(species)
            opacity_species.append(species)
    return opacity_species or None

def _is_active_kinetics_base_photo_branch(
    reactants: list[str],
    products: list[str],
    special_photo_parent_species: set[str],
    active_opacity_species: list[str] | None,
) -> bool:
    if not reactants:
        return False
    parent = reactants[0]
    if (
        active_opacity_species is not None
        and parent == "N2"
        and parent in active_opacity_species
    ):
        return sorted(products) == ["N", "N(2D)"]
    if parent == "CH4" and sorted(products) == ["(3)CH2", "H", "H"]:
        return False
    if not special_photo_parent_species or parent in special_photo_parent_species:
        return True
    if active_opacity_species is None or parent not in active_opacity_species:
        return False
    return True

def _kinetics_base_photo_rates(
    catalog_path: str | Path | None,
    cross_dir: str | Path | None,
    flux_path: str | Path | None,
    aerosol_extinction_path: str | Path | None = None,
    aerosol_albedo_path: str | Path | None = None,
    aerosol_asymmetry_path: str | Path | None = None,
    photolysis_optical_depth_scale: float = 1.0,
    solar_distance_au: float = 1.0,
    solar_mu0: float = 1.0,
    surface_albedo: float = 0.0,
    radiation_streams: int = 4,
    solar_latitude_deg: float | None = None,
    solar_declination_deg: float | None = None,
    solar_hour: float | None = None,
    planet_day_hours: float | None = None,
    diurnal_average: bool | None = None,
    diurnal_quadrature_points: int = 8,
    active_opacity_species: list[str] | None = None,
    radiation_active_nlyr: Any = None,
    kinetics_direct_radiation: bool = False,
) -> dict[str, dict[str, Any]]:
    if catalog_path is None or cross_dir is None or flux_path is None:
        return {}

    flux = _parse_kinetics_base_flux(flux_path)
    if not flux:
        return {}
    flux_scale = 1.0 / max(float(solar_distance_au), 1.0e-300) ** 2
    flux = [
        (wavelength, width, value * flux_scale)
        for wavelength, width, value in flux
    ]
    aerosol_extinction = _parse_kinetics_base_aerosol_extinction(
        aerosol_extinction_path, flux
    )
    aerosol_albedo = _parse_kinetics_base_aerosol_property(
        aerosol_albedo_path, flux, outside_value=0.0
    )
    aerosol_asymmetry = _parse_kinetics_base_aerosol_property(
        aerosol_asymmetry_path, flux, outside_value=0.0
    )

    rates: dict[str, dict[str, Any]] = {}
    cross_root = Path(cross_dir)
    catalog = _parse_kinetics_base_catalog(catalog_path)
    cross_cache = {
        filename: _parse_kinetics_base_cross_section_on_flux(cross_root / filename, flux)
        for _, filename in catalog
    }
    absorption_by_parent: dict[str, list[float]] = {}
    total_cross_section_by_species: dict[str, list[float]] = {}
    for equation, filename in catalog:
        datasets = cross_cache.get(filename, {})
        absorption = datasets.get(0)
        if not absorption:
            continue
        reactants, _ = _parse_kinetics_base_equation(equation)
        if reactants:
            parent_key = _kinetics_base_side_key(reactants)
            _, products = _parse_kinetics_base_equation(equation)
            if products == reactants:
                absorption_by_parent[parent_key] = absorption
            else:
                absorption_by_parent.setdefault(parent_key, absorption)
            if len(reactants) == 1:
                if products == reactants:
                    total_cross_section_by_species.setdefault(reactants[0], absorption)
                else:
                    total_cross_section_by_species.setdefault(
                        reactants[0], absorption
                    )
    if active_opacity_species is not None:
        active_opacity_set = set(active_opacity_species)
        total_cross_section_by_species = {
            species: values
            for species, values in total_cross_section_by_species.items()
            if species in active_opacity_set
        }

    for equation, filename in catalog:
        datasets = cross_cache.get(filename, {})
        if not datasets:
            continue
        reactants, products = _parse_kinetics_base_equation(equation)
        if not reactants or not products:
            continue
        cross = datasets.get(0)
        absorption = cross
        if cross is None and 2 in datasets:
            absorption = absorption_by_parent.get(_kinetics_base_side_key(reactants))
            if absorption is not None:
                cross = [
                    absorption_value * branch_value
                    for absorption_value, branch_value in zip(absorption, datasets[2])
                ]
        if not cross or not any(cross):
            continue
        rate = sum(
            cross_value * flux_value
            for cross_value, (_, _, flux_value) in zip(cross, flux)
        )
        if rate > 0.0:
            rates[_kinetics_base_reaction_key(reactants, products)] = {
                "rate": rate,
                "source": "catalog_flux",
                "wavelengths": [row[0] for row in flux],
                "flux": [row[2] for row in flux],
                "cross_section": cross,
                "absorption_cross_section": absorption or cross,
                "total_cross_section_by_species": total_cross_section_by_species,
                "aerosol_extinction": aerosol_extinction,
                "aerosol_albedo": aerosol_albedo,
                "aerosol_asymmetry": aerosol_asymmetry,
                "optical_depth_scale": photolysis_optical_depth_scale,
                "solar_mu0": solar_mu0,
                "surface_albedo": surface_albedo,
                "radiation_streams": radiation_streams,
                "solar_latitude_deg": solar_latitude_deg,
                "solar_declination_deg": solar_declination_deg,
                "solar_hour": solar_hour,
                "planet_day_hours": planet_day_hours,
                "diurnal_average": bool(diurnal_average),
                "diurnal_quadrature_points": diurnal_quadrature_points,
                "radiation_active_nlyr": radiation_active_nlyr,
                "kinetics_direct_radiation": kinetics_direct_radiation,
            }
    return rates

