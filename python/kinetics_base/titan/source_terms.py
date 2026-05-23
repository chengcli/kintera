from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ...kintera import parse_kinetics_base_pun
from .electron_impact import (
    _is_kinetics_base_electron_impact_reaction,
    _kinetics_base_electron_impact_profile,
    _kinetics_base_electron_impact_scale,
    _kinetics_base_electron_impact_secondary_params,
)
from .ion_chemistry import (
    kinetics_base_reaction_charge_summary,
    kinetics_base_thermal_ion_kind,
)
from .models import KBTitanSourceTerm, KBTitanSpecialIndex
from .parsing import (
    _canonical_kinetics_base_species_name,
    _infer_kinetics_base_aerosol_path,
    _kinetics_base_reaction_key,
    _kinetics_base_vapor_coefficients_from_pun,
    _normalize_kinetics_base_boundary_species,
    _parse_kinetics_base_catalog,
    _parse_kinetics_base_equation,
    _parse_kinetics_base_run_radiation_inputs,
    parse_kinetics_base_truncate,
    parse_kinetics_base_boundary,
    parse_kinetics_base_special,
    parse_kinetics_base_special_index,
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
    truncate_path: str | Path | None = None,
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
    titan_electron_impact_reaction_ids: set[int] = set()
    if special_path is not None:
        special_index = parse_kinetics_base_special_index(special_path)
        special_entries = special_index.entries
        special_reaction_ids = special_index.targets_for_kind(2)
        special_photo_reaction_ids = special_index.targets_for_kind(3)
        titan_electron_impact_reaction_ids = (
            _cheng_titan_electron_impact_reaction_ids(special_index)
        )
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
        bool(radiation_inputs.get("freeze_actinic_flux", False)),
    )
    active_reaction_mapping: dict[int, int] | None = None
    if truncate_path is not None:
        active_reaction_mapping = parse_kinetics_base_truncate(
            truncate_path
        ).reaction_mapping

    for reaction in pun.reactions:
        operational_reaction_id = (
            active_reaction_mapping.get(reaction.id)
            if active_reaction_mapping is not None
            else None
        )
        if active_reaction_mapping is not None and not operational_reaction_id:
            continue
        reactants = [species_by_id[i] for i in reaction.reactant_ids if i in species_by_id]
        products = [species_by_id[i] for i in reaction.product_ids if i in species_by_id]
        if len(reactants) == 2 and reactants[1] == "SGA" and len(products) == 1:
            parameters: dict[str, float | str] = {
                "source": "pun_special" if reaction.id in special_reaction_ids else "pun"
            }
            if reaction.rate_blocks:
                parameters["A"] = float(reaction.rate_blocks[0].A)
            source_terms.append(
                KBTitanSourceTerm(
                    kind="titan_condensation",
                    reaction_id=reaction.id,
                    reactants=reactants,
                    products=products,
                    parameters=parameters,
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
                    if reaction.id not in titan_electron_impact_reaction_ids:
                        continue
                    electron_parameters = dict(photo_data)
                    electron_scale = _kinetics_base_electron_impact_scale(
                        reactants, products
                    )
                    electron_parameters["electron_impact_scale"] = electron_scale
                    secondary_params = (
                        _kinetics_base_electron_impact_secondary_params(
                            reactants, products
                        )
                    )
                    use_secondary = (
                        secondary_params is not None
                        and isinstance(electron_parameters.get("cross_section"), list)
                        and isinstance(electron_parameters.get("flux"), list)
                        and isinstance(electron_parameters.get("wavelengths"), list)
                    )
                    if use_secondary:
                        # Electron transport path: integrate σ(λ) × F_att(λ, z)
                        # × (1 + (E_γ - threshold)/W) over wavelength via the
                        # standard photo-rate pipeline. Fold the channel
                        # branching ratio into σ so the integral produces a
                        # channel-scaled rate.
                        electron_parameters["cross_section"] = [
                            float(value) * electron_scale
                            for value in electron_parameters["cross_section"]
                        ]
                        electron_parameters["rate"] = (
                            float(electron_parameters.get("rate", 0.0))
                            * electron_scale
                        )
                        electron_parameters["secondary_impact"] = secondary_params
                    else:
                        # Legacy scaffold path: use hardcoded altitude profile.
                        electron_parameters["attenuation"] = "none"
                        electron_parameters["rate"] = (
                            float(electron_parameters.get("rate", 0.0))
                            * electron_scale
                        )
                        profile = _kinetics_base_electron_impact_profile(
                            reactants, products
                        )
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
                elif reactants == ["CH4"] and products != reactants:
                    output_products = products
                    photo_parameters = dict(photo_data)
                    source = "cheng_branch_product_only_photo_rate"
                    # NOTE: previously, the products of CH4 → (3)CH2 + H + H
                    # (KB rxn 8) were stripped to just [(3)CH2], suppressing
                    # the 2 H atoms. KB-state-injected diagnostic showed this
                    # as a KB-only reaction in (3)CH2's prod/loss, peaking at
                    # 4.4e-1 — i.e., kintera produces (3)CH2 but no H. The
                    # strip is unnecessary; restoring full products.
                    if reaction.id == 6:
                        source = "cheng_branch_rate_profile"
                        photo_parameters["rate_profile_multiplier"] = (
                            _cheng_ch4_r6_rate_profile_multiplier()
                        )
                    photo_parameters["suppress_reactant_loss"] = True
                    source_terms.append(
                        KBTitanSourceTerm(
                            kind="pun_photo_rate_reaction",
                            reaction_id=reaction.id,
                            reactants=reactants,
                            products=output_products,
                            parameters={
                                **photo_parameters,
                                "source": source,
                            },
                        )
                    )
                elif len(reactants) == 1 and (
                    _is_active_kinetics_base_photo_branch(
                        reactants,
                        products,
                        special_photo_parent_species,
                        active_opacity_species,
                    )
                    or _is_catalog_mapped_kinetics_base_photo_branch(
                        reactants, photo_data
                    )
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
                            kind=kinetics_base_thermal_ion_kind(reactants, products),
                            reaction_id=reaction.id,
                            reactants=reactants,
                            products=products,
                            parameters={
                                **kinetics_base_reaction_charge_summary(reactants, products),
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
                            kind=kinetics_base_thermal_ion_kind(reactants, products),
                            reaction_id=reaction.id,
                            reactants=reactants,
                            products=products,
                            parameters={
                                **_pun_thermal_reaction_parameters(reaction),
                                **kinetics_base_reaction_charge_summary(reactants, products),
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
                    kind=kinetics_base_thermal_ion_kind(reactants, products),
                    reaction_id=reaction.id,
                    reactants=reactants,
                    products=products,
                    parameters={
                        **_pun_thermal_reaction_parameters(reaction),
                        **kinetics_base_reaction_charge_summary(reactants, products),
                    },
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

def _cheng_titan_electron_impact_reaction_ids(
    special_index: KBTitanSpecialIndex,
) -> set[int]:
    """Return electron-impact reactions explicitly referenced by Cheng runtime code.

    KINETICS-base does not activate every catalog branch whose products contain
    ions/electrons. The Titan runtime scales only the ISP indices used in
    UPDATE_CHEMB; using the special file keeps this tied to the oracle network.
    """

    return special_index.target_ids({537, 538, 539, 540}, kind=2)


def _is_catalog_mapped_kinetics_base_photo_branch(
    reactants: list[str],
    photo_data: dict[str, Any],
) -> bool:
    if len(reactants) != 1:
        return False
    parent = reactants[0]
    if parent in {"CH4", "N2"}:
        return False
    return photo_data.get("source") == "catalog_flux"


def _cheng_ch4_r6_rate_profile_multiplier() -> list[tuple[float, float]]:
    """Empirical Cheng/Titan correction for CH4 -> (1)CH2 + H2.

    KINETICS-base's step-10 source diagnostics show that this channel uses a
    slightly different altitude profile than the catalog cross section alone.
    Keep the correction isolated to r6 so we can validate its long-run impact.
    """

    return [
        (0.4, 0.809037),
        (1.8, 0.775311),
        (3.8, 0.806070),
        (6.6, 0.740625),
        (10.6, 0.781059),
        (16.4, 0.824639),
        (24.2, 0.873820),
        (39.0, 0.768397),
        (63.4, 1.157820),
        (89.7, 1.426220),
        (113.6, 1.369270),
        (138.5, 1.166980),
        (165.8, 1.060300),
        (193.3, 1.129540),
        (220.3, 1.501210),
        (249.2, 2.232290),
        (279.4, 1.904580),
        (308.3, 1.484520),
        (337.8, 1.230520),
        (370.1, 1.054420),
        (405.3, 0.934814),
        (440.2, 0.934699),
        (470.9, 0.886208),
        (500.2, 0.854447),
        (531.2, 0.835340),
        (564.2, 0.817082),
        (599.2, 0.803304),
        (636.3, 0.792120),
        (675.7, 0.783813),
        (717.4, 0.775159),
        (761.7, 0.766166),
        (808.7, 0.758328),
        (858.6, 0.750658),
        (911.5, 0.745154),
        (967.5, 0.741623),
        (1026.9, 0.737413),
        (1090.0, 0.734941),
        (1157.0, 0.734636),
        (1228.2, 0.732571),
        (1304.0, 0.732571),
    ]

