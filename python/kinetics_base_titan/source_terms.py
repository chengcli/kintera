from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..kintera import parse_kinetics_base_pun
from .electron_impact import (
    _is_kinetics_base_electron_impact_reaction,
    _kinetics_base_electron_impact_profile,
    _kinetics_base_electron_impact_scale,
)
from .models import KBTitanSourceTerm
from .parsing import (
    _canonical_kinetics_base_species_name,
    _infer_kinetics_base_aerosol_path,
    _kinetics_base_reaction_key,
    _kinetics_base_vapor_coefficients_from_pun,
    _normalize_kinetics_base_boundary_species,
    _parse_kinetics_base_catalog,
    _parse_kinetics_base_equation,
    _parse_kinetics_base_run_radiation_inputs,
    parse_kinetics_base_boundary,
    parse_kinetics_base_special,
)
from .photochemistry import (
    _is_active_kinetics_base_photo_branch,
    _kinetics_base_opacity_species_from_reaction_ids,
    _kinetics_base_photo_rates,
)
from .physics import (
    _is_disabled_titan_special_placeholder,
    _pun_thermal_reaction_parameters,
    _reaction_has_zero_primary_rate,
)


def build_kinetics_base_titan_source_terms(
    pun: str | Path | Any,
    special_path: str | Path | None = None,
    boundary_path: str | Path | None = None,
    run_input_path: str | Path | None = None,
    photo_catalog_path: str | Path | None = None,
    cross_dir: str | Path | None = None,
    flux_path: str | Path | None = None,
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
) -> list[KBTitanSourceTerm]:
    vapor_coefficients: dict[str, tuple[float, float, float, float, float]] = {}
    if isinstance(pun, (str, Path)):
        vapor_coefficients = _kinetics_base_vapor_coefficients_from_pun(pun)
        pun = parse_kinetics_base_pun(str(pun))

    species_by_id = {species.id: species.name for species in pun.species}
    canonical_species = {name.upper(): name for name in species_by_id.values()}
    source_terms: list[KBTitanSourceTerm] = []
    special_reaction_ids: set[int] = set()
    special_photo_parent_species: set[str] = set()
    special_photo_clones = {
        101: (60, 1.0),
        102: (62, 1.0),
        103: (60, 1.0 / 300.0),
        104: (60, 1.0 / 300.0),
        105: (60, 1.0 / 300.0),
        106: (60, 1.0 / 200.0),
        107: (60, 1.0 / 200.0),
        119: (62, 1.0),
        120: (62, 1.0),
    }
    special_photo_multipliers = {
        # Cheng Titan modification in UPDATE_CHEMB:
        # C2H2 -> C2H + H is doubled relative to the catalog J value.
        10: 2.0,
    }
    if special_path is not None:
        special_entries = parse_kinetics_base_special(special_path)
        special_reaction_ids = {
            entry.target_id
            for entry in special_entries
            if entry.kind == 2
        }
        special_photo_reaction_ids = {
            entry.target_id for entry in special_entries if entry.kind == 3
        }
        for reaction in pun.reactions:
            if reaction.id not in special_photo_reaction_ids:
                continue
            reactants = [
                species_by_id[i] for i in reaction.reactant_ids if i in species_by_id
            ]
            if len(reactants) == 1:
                special_photo_parent_species.add(reactants[0])
    reaction_by_id = {reaction.id: reaction for reaction in pun.reactions}
    radiation_inputs = _parse_kinetics_base_run_radiation_inputs(run_input_path)
    active_photo_reaction_ids = _kinetics_base_active_photo_reaction_ids(
        radiation_inputs.get("active_photo_reaction_ids")
    )
    active_opacity_species = _kinetics_base_opacity_species_from_reaction_ids(
        radiation_inputs.get("active_opacity_reaction_ids"),
        reaction_by_id,
        species_by_id,
    )
    aerosol_extinction_enabled = bool(
        radiation_inputs.get("aerosol_extinction_enabled", aerosol_extinction_path is not None)
    )
    aerosol_scattering_enabled = bool(
        radiation_inputs.get("aerosol_scattering_enabled", aerosol_extinction_enabled)
    )
    photo_rates = _kinetics_base_photo_rates(
        photo_catalog_path,
        cross_dir,
        flux_path,
        aerosol_extinction_path if aerosol_extinction_enabled else None,
        (
            aerosol_albedo_path
            or _infer_kinetics_base_aerosol_path(
                aerosol_extinction_path, "extn", "albedo"
            )
        )
        if aerosol_scattering_enabled
        else None,
        (
            aerosol_asymmetry_path
            or _infer_kinetics_base_aerosol_path(
                aerosol_extinction_path, "extn", "asymm"
            )
        )
        if aerosol_scattering_enabled
        else None,
        photolysis_optical_depth_scale,
        float(radiation_inputs.get("solar_distance_au", solar_distance_au)),
        float(radiation_inputs.get("solar_mu0", solar_mu0)),
        surface_albedo,
        radiation_streams,
        solar_latitude_deg
        if solar_latitude_deg is not None
        else radiation_inputs.get("solar_latitude_deg"),
        solar_declination_deg
        if solar_declination_deg is not None
        else radiation_inputs.get("solar_declination_deg"),
        solar_hour if solar_hour is not None else radiation_inputs.get("solar_hour"),
        planet_day_hours
        if planet_day_hours is not None
        else radiation_inputs.get("planet_day_hours"),
        diurnal_average
        if diurnal_average is not None
        else radiation_inputs.get("diurnal_average"),
        diurnal_quadrature_points,
        active_opacity_species,
        radiation_inputs.get("radiation_active_nlyr"),
        bool(radiation_inputs.get("kinetics_direct_radiation", False)),
    )

    for reaction in pun.reactions:
        reactants = [species_by_id[i] for i in reaction.reactant_ids if i in species_by_id]
        products = [species_by_id[i] for i in reaction.product_ids if i in species_by_id]
        if len(reactants) == 2 and reactants[1] == "SGA" and len(products) == 1:
            source_terms.append(
                KBTitanSourceTerm(
                    kind="titan_condensation",
                    reaction_id=reaction.id,
                    reactants=reactants,
                    products=products,
                    parameters={"source": "pun_special" if reaction.id in special_reaction_ids else "pun"},
                )
            )
        elif (
            len(reactants) == 2
            and reactants[0].startswith("G")
            and reactants[1] == "U"
            and len(products) == 1
            and _reaction_has_zero_primary_rate(reaction)
        ):
            coeffs = vapor_coefficients.get(products[0])
            parameters: dict[str, float | str | int] = {"source": "pun_special"}
            if coeffs is not None:
                parameters.update(
                    {
                        "vapor_A": coeffs[0],
                        "vapor_B": coeffs[1],
                        "vapor_C": coeffs[2],
                        "vapor_Tmin_C": coeffs[3],
                        "vapor_Tmax_C": coeffs[4],
                    }
                )
            source_terms.append(
                KBTitanSourceTerm(
                    kind="titan_sublimation",
                    reaction_id=reaction.id,
                    reactants=reactants,
                    products=products,
                    parameters=parameters,
                )
            )
        elif _reaction_has_zero_primary_rate(reaction):
            photo_data = photo_rates.get(_kinetics_base_reaction_key(reactants, products))
            if photo_data is not None and reaction.id in special_photo_multipliers:
                scale = special_photo_multipliers[reaction.id]
                photo_data = dict(photo_data)
                photo_data["rate"] = float(photo_data.get("rate", 0.0)) * scale
                if isinstance(photo_data.get("cross_section"), list):
                    photo_data["cross_section"] = [
                        float(value) * scale for value in photo_data["cross_section"]
                    ]
                photo_data["source"] = "cheng_photo_multiplier"
                photo_data["scale"] = scale
            if photo_data is None and reaction.id in special_photo_clones:
                source_reaction_id, scale = special_photo_clones[reaction.id]
                source_reaction = reaction_by_id.get(source_reaction_id)
                if source_reaction is not None:
                    source_reactants = [
                        species_by_id[i]
                        for i in source_reaction.reactant_ids
                        if i in species_by_id
                    ]
                    source_products = [
                        species_by_id[i]
                        for i in source_reaction.product_ids
                        if i in species_by_id
                    ]
                    source_photo_data = photo_rates.get(
                        _kinetics_base_reaction_key(source_reactants, source_products)
                    )
                    if source_photo_data is not None:
                        photo_data = dict(source_photo_data)
                        photo_data["rate"] = float(photo_data.get("rate", 0.0)) * scale
                        if isinstance(photo_data.get("cross_section"), list):
                            photo_data["cross_section"] = [
                                float(value) * scale
                                for value in photo_data["cross_section"]
                            ]
                        photo_data["source"] = "cheng_photo_clone"
                        photo_data["source_reaction_id"] = source_reaction_id
                        photo_data["scale"] = scale
            if photo_data is not None:
                if _is_kinetics_base_electron_impact_reaction(products):
                    if (
                        active_photo_reaction_ids is not None
                        and reaction.id not in active_photo_reaction_ids
                    ):
                        continue
                    electron_parameters = dict(photo_data)
                    electron_parameters["attenuation"] = "none"
                    electron_scale = _kinetics_base_electron_impact_scale(
                        reactants, products
                    )
                    electron_parameters["rate"] = (
                        float(electron_parameters.get("rate", 0.0)) * electron_scale
                    )
                    electron_parameters["electron_impact_scale"] = electron_scale
                    profile = _kinetics_base_electron_impact_profile(reactants, products)
                    if profile is not None:
                        electron_parameters["rate_profile_multiplier"] = profile
                    else:
                        electron_parameters["min_altitude_km"] = 650.0
                    source_terms.append(
                        KBTitanSourceTerm(
                            kind="pun_electron_impact_reaction",
                            reaction_id=reaction.id,
                            reactants=reactants,
                            products=products,
                            parameters=electron_parameters,
                        )
                    )
                elif reactants == ["CH4"] and sorted(products) == ["(3)CH2", "H", "H"]:
                    source_terms.append(
                        KBTitanSourceTerm(
                            kind="pun_photo_rate_reaction",
                            reaction_id=reaction.id,
                            reactants=reactants,
                            products=["(3)CH2"],
                            parameters={
                                **photo_data,
                                "source": "cheng_product_only_photo_rate",
                                "suppress_reactant_loss": True,
                            },
                        )
                    )
                elif len(reactants) == 1 and _is_active_kinetics_base_photo_branch(
                    reactants,
                    products,
                    special_photo_parent_species,
                    active_opacity_species,
                ):
                    source_terms.append(
                        KBTitanSourceTerm(
                            kind="pun_photo_rate_reaction",
                            reaction_id=reaction.id,
                            reactants=reactants,
                            products=products,
                            parameters=photo_data,
                        )
                    )
                elif reaction.id in special_reaction_ids and not (
                    reactants == ["N2"] and sorted(products) == ["N", "N"]
                ):
                    source_terms.append(
                        KBTitanSourceTerm(
                            kind="pun_photo_rate_reaction",
                            reaction_id=reaction.id,
                            reactants=reactants,
                            products=products,
                            parameters={
                                **photo_data,
                                "source": "special_photo_rate",
                            },
                        )
                    )
            elif reaction.id in special_reaction_ids:
                if reaction.id == 642:
                    source_terms.append(
                        KBTitanSourceTerm(
                            kind="pun_thermal_reaction",
                            reaction_id=reaction.id,
                            reactants=reactants,
                            products=products,
                            parameters={
                                "source": "cheng_special_rate",
                                "formula": "c2h3_c2h5_branch",
                            },
                        )
                    )
                elif _is_disabled_titan_special_placeholder(reaction.id, reactants, products):
                    source_terms.append(
                        KBTitanSourceTerm(
                            kind="disabled_special_placeholder",
                            reaction_id=reaction.id,
                            reactants=reactants,
                            products=products,
                            parameters={"source": "recognized_noop_special"},
                        )
                    )
                elif reaction.rate_blocks and reaction.rate_blocks[0].A != 0.0:
                    source_terms.append(
                        KBTitanSourceTerm(
                            kind="pun_thermal_reaction",
                            reaction_id=reaction.id,
                            reactants=reactants,
                            products=products,
                            parameters={
                                **_pun_thermal_reaction_parameters(reaction),
                                "source": "special_thermal_rate",
                            },
                        )
                    )
                else:
                    source_terms.append(
                        KBTitanSourceTerm(
                            kind="unimplemented_special_reaction",
                            reaction_id=reaction.id,
                            reactants=reactants,
                            products=products,
                            parameters={"source": "special_zero_rate"},
                        )
                    )
        elif reaction.rate_blocks and reaction.rate_blocks[0].A != 0.0:
            source_terms.append(
                KBTitanSourceTerm(
                    kind="pun_thermal_reaction",
                    reaction_id=reaction.id,
                    reactants=reactants,
                    products=products,
                    parameters=_pun_thermal_reaction_parameters(reaction),
                )
            )

    if boundary_path is not None:
        for boundary in parse_kinetics_base_boundary(boundary_path):
            normalized = _canonical_kinetics_base_species_name(
                _normalize_kinetics_base_boundary_species(boundary.species),
                canonical_species,
            )
            if boundary.lower_kind == 2:
                source_terms.append(
                    KBTitanSourceTerm(
                        kind="lower_boundary_velocity",
                        reaction_id=None,
                        reactants=[normalized],
                        products=[normalized],
                        parameters={"value": boundary.lower_value},
                    )
                )
            elif boundary.lower_kind == 4:
                source_terms.append(
                    KBTitanSourceTerm(
                        kind="lower_boundary_flux",
                        reaction_id=None,
                        reactants=[],
                        products=[normalized],
                        parameters={"value": boundary.lower_value},
                    )
                )
            if boundary.upper_kind == 2:
                source_terms.append(
                    KBTitanSourceTerm(
                        kind="upper_boundary_velocity",
                        reaction_id=None,
                        reactants=[normalized],
                        products=[normalized],
                        parameters={"value": boundary.upper_value},
                    )
                )
            elif boundary.upper_kind == 4:
                source_terms.append(
                    KBTitanSourceTerm(
                        kind="upper_boundary_flux",
                        reaction_id=None,
                        reactants=[],
                        products=[normalized],
                        parameters={"value": boundary.upper_value},
                    )
                )

    return source_terms


def _kinetics_base_active_photo_reaction_ids(raw_ids: Any) -> set[int] | None:
    if not isinstance(raw_ids, list):
        return None
    ids: set[int] = set()
    for raw_id in raw_ids:
        try:
            ids.add(int(raw_id))
        except (TypeError, ValueError):
            continue
    return ids

