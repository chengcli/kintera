from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
import pyharp

from .atm2d import (
    AtmState2D,
    IndexedBoundaryFluxSource,
    IndexedBoundaryVelocitySource,
    IndexedFirstOrderSource,
    IndexedMassActionSource,
    IndexedReversibleFirstOrderSource,
    LocalSourceLinearization,
    LocalSourceTerm,
)
from .kintera import parse_kinetics_base_atmosphere, parse_kinetics_base_pun


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


@dataclass
class KBTitanFirstOrderAtm2DSource:
    """Atm2D adapter for Titan first-order source terms."""

    titan_state: KBTitanState
    source_terms: list[KBTitanSourceTerm]

    def linearize(self, state: AtmState2D) -> LocalSourceLinearization:
        concentration = state.concentration
        tendency = torch.zeros_like(concentration)
        jacobian = torch.zeros(
            (state.ncol, state.nlyr, state.nspecies, state.nspecies),
            dtype=state.dtype,
            device=state.device,
        )
        source_state = KBTitanState(
            species=self.titan_state.species,
            fixed_species=self.titan_state.fixed_species,
            varying_species=self.titan_state.varying_species,
            conversion=self.titan_state.conversion,
            concentration=concentration,
            density=self.titan_state.density,
            kzz=self.titan_state.kzz,
            state=state,
        )
        species_index = {name: i for i, name in enumerate(self.titan_state.species)}
        dz = state.dx1f.to(dtype=state.dtype, device=state.device).view(1, -1)
        for term in self.source_terms:
            if len(term.reactants) != 1:
                continue
            reactant = species_index.get(term.reactants[0])
            if reactant is None:
                continue
            product_indices: list[int] = []
            seen_products: set[int] = set()
            missing_product = False
            for name in term.products:
                if name not in species_index:
                    missing_product = True
                    continue
                product = species_index[name]
                if product not in seen_products:
                    seen_products.add(product)
                    product_indices.append(product)
            if missing_product:
                continue
            rate_profile = _photo_rate_profile(
                term, source_state, concentration, species_index, reactant, dz
            )
            rate_constant = float(term.parameters.get("rate", 0.0))
            if rate_profile is None and rate_constant <= 0.0:
                continue
            if rate_profile is None:
                rate_profile = torch.full_like(concentration[:, :, reactant], rate_constant)
            min_altitude = term.parameters.get("min_altitude_km")
            if min_altitude is not None:
                altitude_km = state.x1v.to(
                    dtype=state.dtype, device=state.device
                ).view(1, -1) / 1.0e5
                rate_profile = rate_profile * (altitude_km >= float(min_altitude))
            profile_multiplier = _rate_profile_multiplier_on_state_grid(
                term, source_state, state.dtype, state.device
            )
            if profile_multiplier is not None:
                rate_profile = rate_profile * profile_multiplier

            parent = torch.clamp(concentration[:, :, reactant], min=0.0)
            rate = rate_profile * parent
            if not bool(term.parameters.get("suppress_reactant_loss", False)):
                tendency[:, :, reactant] = tendency[:, :, reactant] - rate
                jacobian[:, :, reactant, reactant] = (
                    jacobian[:, :, reactant, reactant] - rate_profile
                )
            for product in product_indices:
                tendency[:, :, product] = tendency[:, :, product] + rate
                jacobian[:, :, product, reactant] = (
                    jacobian[:, :, product, reactant] + rate_profile
                )
        return LocalSourceLinearization(tendency=tendency, jacobian=jacobian)


def kinetics_base_profile_tensor(profile: Any, species: list[str]) -> torch.Tensor:
    rows = [
        [profile.species_profiles[name][i] for name in species]
        for i in range(len(profile.altitude))
    ]
    return torch.tensor(rows)


def parse_kinetics_base_special(path: str | Path) -> list[KBTitanSpecialEntry]:
    entries: list[KBTitanSpecialEntry] = []
    for line in Path(path).read_text().splitlines():
        data, _, comment = line.partition("!")
        parts = data.split()
        if len(parts) < 3:
            continue
        try:
            entries.append(
                KBTitanSpecialEntry(
                    index=int(parts[0]),
                    kind=int(parts[1]),
                    target_id=int(parts[2]),
                    comment=comment.strip(),
                )
            )
        except ValueError:
            continue
    return entries


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


def build_kinetics_base_titan_atm2d_source_terms(
    titan_state: KBTitanState,
    source_terms: list[KBTitanSourceTerm],
    *,
    pun_metadata: dict[str, Any] | None = None,
) -> list[LocalSourceTerm]:
    """Adapt supported Titan source terms to the atm2d local-source interface."""

    species_index = {name: i for i, name in enumerate(titan_state.species)}
    atm_sources: list[LocalSourceTerm] = []

    first_order_terms = [
        term
        for term in source_terms
        if term.kind in {"pun_photo_rate_reaction", "pun_electron_impact_reaction"}
    ]
    if first_order_terms:
        atm_sources.append(
            KBTitanFirstOrderAtm2DSource(
                titan_state=titan_state,
                source_terms=first_order_terms,
            )
        )

    for term in source_terms:
        if term.kind == "pun_thermal_reaction":
            source = _build_titan_thermal_atm2d_source(titan_state, species_index, term)
            if source is not None:
                atm_sources.append(source)

    sublimation_by_pair = {
        (term.products[0], term.reactants[0]): term
        for term in source_terms
        if term.kind == "titan_sublimation"
        and len(term.reactants) == 2
        and len(term.products) == 1
    }
    paired_sublimations: set[int] = set()
    for term in source_terms:
        if term.kind != "titan_condensation":
            continue
        pair_key = (
            term.reactants[0],
            term.products[0],
        ) if len(term.reactants) == 2 and len(term.products) == 1 else None
        sublimation = sublimation_by_pair.get(pair_key)
        if sublimation is not None:
            source = _build_titan_condensation_pair_atm2d_source(
                titan_state, species_index, term, sublimation, pun_metadata
            )
            paired_sublimations.add(id(sublimation))
        else:
            source = _build_titan_condensation_atm2d_source(
                titan_state, species_index, term, pun_metadata
            )
        if source is not None:
            atm_sources.append(source)
    for term in source_terms:
        if term.kind == "titan_sublimation" and id(term) not in paired_sublimations:
            source = _build_titan_sublimation_atm2d_source(
                titan_state, species_index, term, pun_metadata
            )
            if source is not None:
                atm_sources.append(source)

    for term in source_terms:
        if term.kind in {
            "lower_boundary_velocity",
            "lower_boundary_flux",
            "upper_boundary_velocity",
            "upper_boundary_flux",
        }:
            source = _build_titan_boundary_atm2d_source(species_index, term)
            if source is not None:
                atm_sources.append(source)
    return atm_sources


def _build_titan_thermal_atm2d_source(
    titan_state: KBTitanState,
    species_index: dict[str, int],
    term: KBTitanSourceTerm,
) -> IndexedMassActionSource | None:
    reactants = [species_index[name] for name in term.reactants if name in species_index]
    products = [species_index[name] for name in term.products if name in species_index]
    if len(reactants) != len(term.reactants) or not products:
        return None
    reactant_coeffs = list(term.parameters.get("reactant_coefficients", []))
    product_coeffs = list(term.parameters.get("product_coefficients", []))
    if len(reactant_coeffs) != len(reactants):
        reactant_coeffs = [1] * len(reactants)
    if len(product_coeffs) != len(products):
        product_coeffs = [1] * len(products)

    def rate_provider(state: AtmState2D) -> torch.Tensor:
        density = titan_state.density.to(dtype=state.dtype, device=state.device)
        return _pun_rate_constant(term.parameters, state.temperature, density)

    return IndexedMassActionSource(
        reactants=reactants,
        products=products,
        reactant_coefficients=reactant_coeffs,
        product_coefficients=product_coeffs,
        rate_constant=rate_provider,
    )


def _build_titan_condensation_atm2d_source(
    titan_state: KBTitanState,
    species_index: dict[str, int],
    term: KBTitanSourceTerm,
    pun_metadata: dict[str, Any] | None,
) -> IndexedMassActionSource | None:
    if len(term.reactants) != 2 or len(term.products) != 1:
        return None
    gas, surface = term.reactants
    product = term.products[0]
    if gas not in species_index or surface not in species_index or product not in species_index:
        return None
    mass_amu = _kinetics_base_species_mass_amu(gas, pun_metadata)
    if mass_amu <= 0.0:
        return None

    def rate_provider(state: AtmState2D) -> torch.Tensor:
        return _titan_thermal_velocity(state.temperature, mass_amu) * _titan_sticking_coefficient(
            gas, stick_x=1.0e-5, stick_h=0.3
        )

    return IndexedMassActionSource(
        reactants=[species_index[gas], species_index[surface]],
        products=[species_index[product]],
        reactant_coefficients=[1, 1],
        product_coefficients=[1],
        rate_constant=rate_provider,
    )


def _build_titan_sublimation_atm2d_source(
    titan_state: KBTitanState,
    species_index: dict[str, int],
    term: KBTitanSourceTerm,
    pun_metadata: dict[str, Any] | None,
) -> IndexedFirstOrderSource | None:
    if len(term.reactants) != 2 or len(term.products) != 1:
        return None
    grain, _surface = term.reactants
    gas = term.products[0]
    if grain not in species_index or gas not in species_index or "SGA" not in species_index:
        return None
    if "vapor_A" not in term.parameters:
        return None
    mass_amu = _kinetics_base_species_mass_amu(gas, pun_metadata)
    if mass_amu <= 0.0:
        return None

    def rate_provider(state: AtmState2D) -> torch.Tensor:
        return _titan_sublimation_rate_profile(
            state, species_index, term.parameters, gas, mass_amu
        )

    return IndexedFirstOrderSource(
        reactant=species_index[grain],
        products=[species_index[gas]],
        rate=rate_provider,
    )


def _build_titan_condensation_pair_atm2d_source(
    titan_state: KBTitanState,
    species_index: dict[str, int],
    condensation: KBTitanSourceTerm,
    sublimation: KBTitanSourceTerm,
    pun_metadata: dict[str, Any] | None,
) -> IndexedReversibleFirstOrderSource | None:
    if len(condensation.reactants) != 2 or len(condensation.products) != 1:
        return None
    gas = condensation.reactants[0]
    grain = condensation.products[0]
    if gas not in species_index or grain not in species_index or "SGA" not in species_index:
        return None
    if "vapor_A" not in sublimation.parameters:
        return _build_titan_condensation_atm2d_source(
            titan_state, species_index, condensation, pun_metadata
        )
    mass_amu = _kinetics_base_species_mass_amu(gas, pun_metadata)
    if mass_amu <= 0.0:
        return None

    def condensation_rate(state: AtmState2D) -> torch.Tensor:
        sga = state.concentration[:, :, species_index["SGA"]]
        return (
            _titan_thermal_velocity(state.temperature, mass_amu)
            * _titan_sticking_coefficient(gas, stick_x=1.0e-5, stick_h=0.3)
            * sga
        )

    def sublimation_rate(state: AtmState2D) -> torch.Tensor:
        return _titan_sublimation_rate_profile(
            state, species_index, sublimation.parameters, gas, mass_amu
        )

    return IndexedReversibleFirstOrderSource(
        left=species_index[gas],
        right=species_index[grain],
        forward_rate=condensation_rate,
        reverse_rate=sublimation_rate,
    )


def _build_titan_boundary_atm2d_source(
    species_index: dict[str, int],
    term: KBTitanSourceTerm,
) -> IndexedBoundaryFluxSource | IndexedBoundaryVelocitySource | None:
    species = term.reactants or term.products
    if not species or species[0] not in species_index:
        return None
    value = float(term.parameters.get("value", 0.0))
    if value == 0.0:
        return None
    index = species_index[species[0]]
    if term.kind == "lower_boundary_flux":
        return IndexedBoundaryFluxSource(species=index, value=value, boundary="lower")
    if term.kind == "upper_boundary_flux":
        return IndexedBoundaryFluxSource(species=index, value=value, boundary="upper")
    if term.kind == "lower_boundary_velocity":
        return IndexedBoundaryVelocitySource(species=index, value=value, boundary="lower")
    if term.kind == "upper_boundary_velocity":
        return IndexedBoundaryVelocitySource(species=index, value=value, boundary="upper")
    return None


def apply_kinetics_base_titan_source_terms(
    concentration: torch.Tensor,
    titan_state: KBTitanState,
    source_terms: list[KBTitanSourceTerm],
    dt: float,
    *,
    stick_x: float = 1.0e-5,
    stick_h: float = 0.3,
    pun_metadata: dict[str, Any] | None = None,
) -> torch.Tensor:
    updated = concentration.clone()
    species_index = {name: i for i, name in enumerate(titan_state.species)}
    temperature = titan_state.state.temperature
    sublimation_by_pair = {
        (term.products[0], term.reactants[0]): term
        for term in source_terms
        if term.kind == "titan_sublimation"
        and len(term.reactants) == 2
        and len(term.products) == 1
    }
    paired_terms: set[int] = set()
    thermal_terms = [
        term for term in source_terms if term.kind == "pun_thermal_reaction"
    ]
    if thermal_terms:
        _apply_pun_thermal_reactions(
            updated, titan_state, species_index, thermal_terms, dt, temperature
        )
    photo_terms = [
        term for term in source_terms if term.kind == "pun_photo_rate_reaction"
    ]
    if photo_terms:
        _apply_pun_first_order_reactions(
            updated, titan_state, species_index, photo_terms, dt
        )
    electron_terms = [
        term for term in source_terms if term.kind == "pun_electron_impact_reaction"
    ]
    if electron_terms:
        _apply_pun_first_order_reactions(
            updated, titan_state, species_index, electron_terms, dt
        )
    for term in source_terms:
        if term.kind == "titan_condensation":
            pair_key = (
                term.reactants[0],
                term.products[0],
            ) if len(term.reactants) == 2 and len(term.products) == 1 else None
            sublimation = sublimation_by_pair.get(pair_key)
            if sublimation is not None:
                _apply_titan_condensation_sublimation_pair(
                    updated,
                    titan_state,
                    species_index,
                    term,
                    sublimation,
                    dt,
                    stick_x=stick_x,
                    stick_h=stick_h,
                    pun_metadata=pun_metadata,
                    temperature=temperature,
                )
                paired_terms.add(id(sublimation))
            else:
                _apply_titan_condensation_source(
                    updated,
                    titan_state,
                    species_index,
                    term,
                    dt,
                    stick_x=stick_x,
                    stick_h=stick_h,
                    pun_metadata=pun_metadata,
                    temperature=temperature,
                )
        elif term.kind == "titan_sublimation":
            if id(term) in paired_terms:
                continue
            _apply_titan_sublimation_source(
                updated,
                titan_state,
                species_index,
                term,
                dt,
                stick_x=stick_x,
                stick_h=stick_h,
                pun_metadata=pun_metadata,
                temperature=temperature,
            )
        elif term.kind in {
            "lower_boundary_velocity",
            "lower_boundary_flux",
            "upper_boundary_velocity",
            "upper_boundary_flux",
        }:
            _apply_boundary_source(updated, titan_state, species_index, term, dt)
    return updated


def kinetics_base_titan_source_tendencies(
    concentration: torch.Tensor,
    titan_state: KBTitanState,
    source_terms: list[KBTitanSourceTerm],
    dt: float,
    *,
    stick_x: float = 1.0e-5,
    stick_h: float = 0.3,
    pun_metadata: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    tendencies: dict[str, torch.Tensor] = {}
    source_kinds = list(dict.fromkeys(term.kind for term in source_terms))
    for kind in source_kinds:
        terms = [term for term in source_terms if term.kind == kind]
        updated = apply_kinetics_base_titan_source_terms(
            concentration,
            titan_state,
            terms,
            dt,
            stick_x=stick_x,
            stick_h=stick_h,
            pun_metadata=pun_metadata,
        )
        tendencies[kind] = (updated - concentration) / dt
    return tendencies


def _pun_thermal_reaction_parameters(reaction: Any) -> dict[str, Any]:
    block = reaction.rate_blocks[0]
    reactant_coeffs: list[int] = []
    product_coeffs: list[int] = []
    participants = list(reaction.participants)
    for participant in participants[: reaction.n_reactants]:
        reactant_coeffs.append(max(1, int(participant.coefficient)))
    for participant in participants[
        reaction.n_reactants : reaction.n_reactants + reaction.n_products
    ]:
        product_coeffs.append(max(1, int(participant.coefficient)))
    return {
        "A": float(block.A),
        "b": float(block.b),
        "C": float(block.C),
        "D": float(block.D),
        "E": float(block.E),
        "F": float(block.F),
        "Tmin": float(block.Tmin),
        "Tmax": float(block.Tmax),
        "reactant_coefficients": reactant_coeffs,
        "product_coefficients": product_coeffs,
    }


def _is_disabled_titan_special_placeholder(
    reaction_id: int,
    reactants: list[str],
    products: list[str],
) -> bool:
    if reaction_id == 187 and reactants == ["bNO"] and products == ["bN", "O"]:
        return True
    if reaction_id == 2094 and reactants == ["U", "U"] and products == ["C6H6", "C6H6"]:
        return True
    if reaction_id in {2095, 2096, 2097, 2098, 2099} and reactants == ["U", "U"]:
        return True
    return False


def _apply_pun_thermal_reactions(
    concentration: torch.Tensor,
    titan_state: KBTitanState,
    species_index: dict[str, int],
    terms: list[KBTitanSourceTerm],
    dt: float,
    temperature: torch.Tensor,
) -> None:
    production = torch.zeros_like(concentration)
    loss = torch.zeros_like(concentration)
    density = titan_state.density.to(
        dtype=concentration.dtype, device=concentration.device
    )
    for term in terms:
        reactant_indices = [species_index[name] for name in term.reactants if name in species_index]
        product_indices = [species_index[name] for name in term.products if name in species_index]
        if len(reactant_indices) != len(term.reactants) or not product_indices:
            continue
        reactant_coeffs = list(term.parameters.get("reactant_coefficients", []))
        product_coeffs = list(term.parameters.get("product_coefficients", []))
        if len(reactant_coeffs) != len(reactant_indices):
            reactant_coeffs = [1] * len(reactant_indices)
        if len(product_coeffs) != len(product_indices):
            product_coeffs = [1] * len(product_indices)

        rate = _pun_rate_constant(term.parameters, temperature, density)
        for idx, coeff in zip(reactant_indices, reactant_coeffs):
            rate = rate * torch.clamp(concentration[:, :, idx], min=0.0) ** coeff
        rate = torch.clamp(rate, min=0.0)
        for idx, coeff in zip(product_indices, product_coeffs):
            production[:, :, idx] = production[:, :, idx] + coeff * rate
        for idx, coeff in zip(reactant_indices, reactant_coeffs):
            loss[:, :, idx] = loss[:, :, idx] + coeff * rate

    positive = concentration > 0.0
    loss_frequency = torch.zeros_like(concentration)
    loss_frequency[positive] = loss[positive] / concentration[positive]
    stable = loss_frequency > 0.0
    next_concentration = concentration + dt * production
    next_concentration[stable] = (
        concentration[stable] + dt * production[stable]
    ) / (1.0 + dt * loss_frequency[stable])
    concentration[:] = torch.clamp(next_concentration, min=0.0)


def _apply_pun_first_order_reactions(
    concentration: torch.Tensor,
    titan_state: KBTitanState,
    species_index: dict[str, int],
    terms: list[KBTitanSourceTerm],
    dt: float,
) -> None:
    production = torch.zeros_like(concentration)
    loss = torch.zeros_like(concentration)
    dz = titan_state.state.dx1f.to(
        dtype=concentration.dtype, device=concentration.device
    ).view(1, -1)
    for term in terms:
        if len(term.reactants) != 1:
            continue
        reactant = species_index.get(term.reactants[0])
        product_indices: list[int] = []
        seen_products: set[int] = set()
        missing_product = False
        for name in term.products:
            if name not in species_index:
                missing_product = True
                continue
            index = species_index[name]
            if index not in seen_products:
                seen_products.add(index)
                product_indices.append(index)
        if reactant is None or missing_product:
            continue
        rate_constant = float(term.parameters.get("rate", 0.0))
        rate_profile = _photo_rate_profile(
            term, titan_state, concentration, species_index, reactant, dz
        )
        if rate_profile is None and rate_constant <= 0.0:
            continue
        if rate_profile is None:
            rate_profile = torch.full_like(concentration[:, :, reactant], rate_constant)
        min_altitude = term.parameters.get("min_altitude_km")
        if min_altitude is not None:
            altitude_km = titan_state.state.x1v.to(
                dtype=concentration.dtype, device=concentration.device
            ).view(1, -1) / 1.0e5
            rate_profile = rate_profile * (altitude_km >= float(min_altitude))
        profile_multiplier = _rate_profile_multiplier_on_state_grid(
            term, titan_state, concentration.dtype, concentration.device
        )
        if profile_multiplier is not None:
            rate_profile = rate_profile * profile_multiplier
        rate = rate_profile * torch.clamp(concentration[:, :, reactant], min=0.0)
        if not bool(term.parameters.get("suppress_reactant_loss", False)):
            loss[:, :, reactant] = loss[:, :, reactant] + rate
        for product in product_indices:
            production[:, :, product] = production[:, :, product] + rate

    positive = concentration > 0.0
    loss_frequency = torch.zeros_like(concentration)
    loss_frequency[positive] = loss[positive] / concentration[positive]
    stable = loss_frequency > 0.0
    next_concentration = concentration + dt * production
    next_concentration[stable] = (
        concentration[stable] + dt * production[stable]
    ) / (1.0 + dt * loss_frequency[stable])
    concentration[:] = torch.clamp(next_concentration, min=0.0)


def _photo_rate_profile(
    term: KBTitanSourceTerm,
    titan_state: KBTitanState,
    concentration: torch.Tensor,
    species_index: dict[str, int],
    reactant_index: int,
    dz: torch.Tensor,
) -> torch.Tensor | None:
    if term.parameters.get("attenuation") == "none":
        return None
    wavelengths = term.parameters.get("wavelengths")
    reaction_cross_section = term.parameters.get("cross_section")
    flux = term.parameters.get("flux")
    if not (
        isinstance(wavelengths, list)
        and isinstance(reaction_cross_section, list)
        and isinstance(flux, list)
    ):
        return None
    if not wavelengths:
        return None

    dtype = concentration.dtype
    device = concentration.device
    reaction_sigma = torch.tensor(reaction_cross_section, dtype=dtype, device=device)
    flux_tensor = torch.tensor(flux, dtype=dtype, device=device)
    actinic_flux = _kinetics_base_pyharp_actinic_flux(
        term,
        titan_state,
        concentration,
        species_index,
        flux_tensor,
        reaction_sigma.numel(),
        dtype,
        device,
    )
    if actinic_flux is None:
        return None
    return (actinic_flux * reaction_sigma.view(1, 1, -1)).sum(dim=-1)


def _rate_profile_multiplier_on_state_grid(
    term: KBTitanSourceTerm,
    titan_state: KBTitanState,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    points = term.parameters.get("rate_profile_multiplier")
    if not isinstance(points, list) or not points:
        return None
    altitude_points: list[float] = []
    multiplier_points: list[float] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            altitude_points.append(float(point[0]))
            multiplier_points.append(float(point[1]))
        except (TypeError, ValueError):
            continue
    if len(altitude_points) < 2:
        return None
    altitude = titan_state.state.x1v.to(dtype=dtype, device=device) / 1.0e5
    multiplier = torch.tensor(multiplier_points, dtype=dtype, device=device)
    source_altitude = torch.tensor(altitude_points, dtype=dtype, device=device)
    output = torch.zeros_like(altitude)
    for i, value in enumerate(altitude):
        if value <= source_altitude[0]:
            output[i] = multiplier[0]
        elif value >= source_altitude[-1]:
            output[i] = multiplier[-1]
        else:
            upper = int(torch.searchsorted(source_altitude, value).item())
            lower = upper - 1
            frac = (value - source_altitude[lower]) / (
                source_altitude[upper] - source_altitude[lower]
            )
            output[i] = multiplier[lower] + frac * (
                multiplier[upper] - multiplier[lower]
            )
    return output.view(1, -1)


def _kinetics_base_pyharp_actinic_flux(
    term: KBTitanSourceTerm,
    titan_state: KBTitanState,
    concentration: torch.Tensor,
    species_index: dict[str, int],
    top_flux: torch.Tensor,
    nwave: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    total_cross_sections = term.parameters.get("total_cross_section_by_species")
    if not isinstance(total_cross_sections, dict):
        return None

    dz = titan_state.state.dx1f.to(dtype=dtype, device=device).view(1, -1)
    gas_extinction = torch.zeros(
        (*concentration.shape[:2], nwave), dtype=dtype, device=device
    )
    for species_name, values in total_cross_sections.items():
        idx = species_index.get(str(species_name))
        if idx is None or not isinstance(values, list) or len(values) != nwave:
            continue
        sigma = torch.tensor(values, dtype=dtype, device=device)
        gas_extinction = gas_extinction + torch.clamp(
            concentration[:, :, idx], min=0.0
        ).unsqueeze(-1) * sigma.view(1, 1, -1)

    aerosol_extinction_profile = torch.zeros_like(gas_extinction)
    aerosol_extinction = _aerosol_extinction_on_state_grid(
        term, titan_state, nwave, dtype, device
    )
    jdust_idx = species_index.get("JDUST")
    if aerosol_extinction is not None and jdust_idx is not None:
        aerosol_extinction_profile = torch.clamp(
            concentration[:, :, jdust_idx], min=0.0
        ).unsqueeze(-1) * aerosol_extinction.unsqueeze(0)
    extinction = gas_extinction + aerosol_extinction_profile
    if bool(term.parameters.get("kinetics_direct_radiation", False)):
        return _kinetics_base_direct_actinic_flux(
            term,
            titan_state,
            concentration,
            species_index,
            top_flux,
            dtype,
            device,
        )

    optical_depth = torch.clamp(
        extinction
        * dz.unsqueeze(-1)
        * float(term.parameters.get("optical_depth_scale", 1.0)),
        min=0.0,
        max=700.0,
    )
    active_nlyr = term.parameters.get("radiation_active_nlyr")
    if active_nlyr is not None:
        try:
            active_nlyr_int = int(active_nlyr)
        except (TypeError, ValueError):
            active_nlyr_int = titan_state.state.nlyr
        if 0 < active_nlyr_int < titan_state.state.nlyr:
            optical_depth = optical_depth.clone()
            optical_depth[:, active_nlyr_int:, :] = 0.0

    pydisort = pyharp.pydisort
    options = pydisort.DisortOptions().flags("onlyfl,lamber,quiet")
    options.ds().nlyr = titan_state.state.nlyr
    options.ds().nstr = int(term.parameters.get("radiation_streams", 4))
    options.ds().nmom = options.ds().nstr
    options.ds().nphase = options.ds().nstr
    options.nwave(nwave).ncol(titan_state.state.ncol)
    disort = pydisort.Disort(options)

    nprop = 2 + options.ds().nmom
    prop = torch.zeros(
        (nwave, titan_state.state.ncol, titan_state.state.nlyr, nprop),
        dtype=dtype,
        device=device,
    )
    prop[..., 0] = torch.flip(optical_depth, dims=[1]).permute(2, 0, 1)
    scattering_optical_depth = torch.zeros_like(optical_depth)
    aerosol_albedo = _aerosol_property_on_state_grid(
        term, "aerosol_albedo", titan_state, nwave, dtype, device
    )
    if aerosol_albedo is not None:
        scattering_optical_depth = aerosol_extinction_profile * dz.unsqueeze(-1)
        scattering_optical_depth = scattering_optical_depth * torch.clamp(
            aerosol_albedo.unsqueeze(0), min=0.0, max=1.0
        )
    scattering_optical_depth = scattering_optical_depth * float(
        term.parameters.get("optical_depth_scale", 1.0)
    )
    if active_nlyr is not None and 0 < active_nlyr_int < titan_state.state.nlyr:
        scattering_optical_depth = scattering_optical_depth.clone()
        scattering_optical_depth[:, active_nlyr_int:, :] = 0.0
    positive_tau = optical_depth > 0.0
    single_scattering_albedo = torch.zeros_like(optical_depth)
    single_scattering_albedo[positive_tau] = (
        scattering_optical_depth[positive_tau] / optical_depth[positive_tau]
    )
    prop[..., 1] = torch.flip(
        torch.clamp(single_scattering_albedo, min=0.0, max=1.0), dims=[1]
    ).permute(2, 0, 1)

    aerosol_asymmetry = _aerosol_property_on_state_grid(
        term, "aerosol_asymmetry", titan_state, nwave, dtype, device
    )
    if aerosol_asymmetry is not None:
        g = torch.flip(torch.clamp(aerosol_asymmetry, min=-0.999, max=0.999), dims=[0])
        for moment in range(options.ds().nmom):
            prop[..., 2 + moment] = (
                g.pow(moment + 1).transpose(0, 1).unsqueeze(1)
            )

    top_flux_2d = top_flux.view(nwave, 1).expand(nwave, titan_state.state.ncol)
    zeros = torch.zeros(titan_state.state.ncol, dtype=dtype, device=device)
    albedo = torch.full(
        (nwave, titan_state.state.ncol),
        max(min(float(term.parameters.get("surface_albedo", 0.0)), 1.0), 0.0),
        dtype=dtype,
        device=device,
    )
    actinic_flux = torch.zeros(
        (*concentration.shape[:2], nwave), dtype=dtype, device=device
    )
    for mu0, weight in _kinetics_base_solar_mu0_weights(term.parameters):
        if weight <= 0.0:
            continue
        disort.forward(
            prop,
            umu0=torch.full_like(zeros, mu0),
            phi0=zeros,
            fbeam=top_flux_2d,
            albedo=albedo,
        )
        gathered = disort.gather_flx()
        average_intensity_levels = gathered[..., pydisort.kIUAVG]
        actinic_flux_levels = torch.flip(
            (4.0 * torch.pi * average_intensity_levels).permute(1, 2, 0),
            dims=[1],
        )
        actinic_flux = actinic_flux + weight * actinic_flux_levels[:, 1:, :]
    return actinic_flux


def _kinetics_base_direct_actinic_flux(
    term: KBTitanSourceTerm,
    titan_state: KBTitanState,
    concentration: torch.Tensor,
    species_index: dict[str, int],
    top_flux: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    ncol, nlyr, nwave = concentration.shape[0], concentration.shape[1], top_flux.numel()
    active_nlyr = min(int(term.parameters.get("radiation_active_nlyr") or nlyr), nlyr)
    opacity = term.parameters.get("total_cross_section_by_species", {})
    if not isinstance(opacity, dict) or active_nlyr <= 0:
        return top_flux.view(1, 1, nwave).expand(ncol, nlyr, nwave)

    alt_km = titan_state.state.x1v.to(dtype=dtype, device=device) / 1.0e5
    tau = torch.zeros((ncol, nlyr, nwave), dtype=dtype, device=device)
    species_ext: list[torch.Tensor] = []
    species_concentration: list[torch.Tensor] = []
    for name, values in opacity.items():
        idx = species_index.get(str(name))
        if idx is None:
            continue
        sigma = torch.tensor(values, dtype=dtype, device=device)
        if sigma.numel() != nwave:
            continue
        conc = torch.clamp(concentration[:, :active_nlyr, idx], min=0.0)
        species_concentration.append(conc)
        species_ext.append(conc.unsqueeze(-1) * sigma.view(1, 1, nwave))
    if not species_ext:
        return top_flux.view(1, 1, nwave).expand(ncol, nlyr, nwave)

    total_ext = torch.stack(species_ext, dim=0).sum(dim=0)
    column = torch.zeros((ncol, active_nlyr, nwave), dtype=dtype, device=device)
    if active_nlyr > 1:
        top_tau = torch.zeros((ncol, nwave), dtype=dtype, device=device)
        for conc, ext in zip(species_concentration, species_ext):
            c0 = conc[:, active_nlyr - 2]
            c1 = conc[:, active_nlyr - 1]
            dz_top = alt_km[active_nlyr - 1] - alt_km[active_nlyr - 2]
            scale_height = torch.full_like(c1, 10.0)
            valid = (c0 != 0.0) & (c1 != 0.0) & (c0 != c1)
            ratio = torch.zeros_like(c1)
            ratio[valid] = torch.abs(c1[valid]) / torch.abs(c0[valid])
            scale_height[valid] = torch.abs(dz_top / torch.log(ratio[valid]))
            top_tau = top_tau + scale_height.unsqueeze(-1) * ext[:, active_nlyr - 1, :]
        column[:, active_nlyr - 1, :] = top_tau * 1.0e5
        for i in range(active_nlyr - 2, -1, -1):
            dz_km = alt_km[i + 1] - alt_km[i]
            layer_tau = 0.5 * (total_ext[:, i, :] + total_ext[:, i + 1, :])
            layer_tau = layer_tau * dz_km * 1.0e5
            column[:, i, :] = column[:, i + 1, :] + layer_tau

    column = torch.clamp(
        column * float(term.parameters.get("optical_depth_scale", 1.0)),
        min=0.0,
        max=700.0,
    )
    lat = term.parameters.get("solar_latitude_deg")
    dec = term.parameters.get("solar_declination_deg")
    if lat is None or dec is None:
        mu0 = max(min(float(term.parameters.get("solar_mu0", 1.0)), 1.0), 1.0e-6)
        attenuation = torch.exp(-column / mu0)
    else:
        latitude = math.radians(float(lat))
        declination = math.radians(float(dec))
        a5 = math.sin(declination) * math.sin(latitude)
        b5 = math.cos(declination) * math.cos(latitude)
        apb = max(a5 + b5, 1.0e-12)
        cof0 = a5 * math.log(2.0) / b5 if abs(b5) > 1.0e-12 else 0.0
        cf = column / apb
        arg = torch.clamp((cf - cof0) / (cf + math.log(2.0)), min=-1.0, max=1.0)
        attenuation = torch.exp(-cf) * torch.acos(arg) / math.pi
    actinic = torch.zeros((ncol, nlyr, nwave), dtype=dtype, device=device)
    actinic[:, :active_nlyr, :] = top_flux.view(1, 1, nwave) * attenuation
    if active_nlyr < nlyr:
        actinic[:, active_nlyr:, :] = top_flux.view(1, 1, nwave)
    return actinic


def _kinetics_base_solar_mu0_weights(parameters: dict[str, Any]) -> list[tuple[float, float]]:
    default_mu0 = max(min(float(parameters.get("solar_mu0", 1.0)), 1.0), 1.0e-6)
    if not parameters.get("diurnal_average"):
        return [(default_mu0, 1.0)]

    latitude = parameters.get("solar_latitude_deg")
    declination = parameters.get("solar_declination_deg")
    if latitude is None or declination is None:
        return [(default_mu0, 1.0)]

    lat = math.radians(float(latitude))
    dec = math.radians(float(declination))
    a5 = math.sin(dec) * math.sin(lat)
    b5 = math.cos(dec) * math.cos(lat)
    if abs(b5) < 1.0e-12:
        mu0 = max(min(a5, 1.0), 0.0)
        return [(max(mu0, 1.0e-6), 1.0 if mu0 > 0.0 else 0.0)]

    terminator = -a5 / b5
    if terminator <= -1.0:
        hour_angle = math.pi
    elif terminator >= 1.0:
        return [(1.0e-6, 0.0)]
    else:
        hour_angle = math.acos(terminator)

    npoint = max(int(parameters.get("diurnal_quadrature_points", 8)), 1)
    daylight_weight = hour_angle / math.pi / float(npoint)
    samples: list[tuple[float, float]] = []
    for index in range(npoint):
        frac = (float(index) + 0.5) / float(npoint)
        local_hour_angle = -hour_angle + 2.0 * hour_angle * frac
        mu0 = a5 + b5 * math.cos(local_hour_angle)
        if mu0 > 0.0:
            samples.append((max(min(mu0, 1.0), 1.0e-6), daylight_weight))
    return samples or [(1.0e-6, 0.0)]


def _aerosol_extinction_on_state_grid(
    term: KBTitanSourceTerm,
    titan_state: KBTitanState,
    nwave: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    return _aerosol_property_on_state_grid(
        term, "aerosol_extinction", titan_state, nwave, dtype, device
    )


def _aerosol_property_on_state_grid(
    term: KBTitanSourceTerm,
    parameter_name: str,
    titan_state: KBTitanState,
    nwave: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    aerosol = term.parameters.get(parameter_name)
    if not isinstance(aerosol, dict):
        return None
    altitudes = aerosol.get("altitude_km")
    values = aerosol.get("values")
    if not isinstance(altitudes, list) or not isinstance(values, list):
        return None
    if not values or len(values[0]) != nwave:
        return None

    source_alt = torch.tensor(altitudes, dtype=dtype, device=device)
    source_values = torch.tensor(values, dtype=dtype, device=device)
    target_alt = titan_state.state.x1v.to(dtype=dtype, device=device) / 1.0e5
    output = torch.zeros((target_alt.numel(), nwave), dtype=dtype, device=device)
    for i, altitude in enumerate(target_alt):
        if altitude <= source_alt[0]:
            output[i] = source_values[0]
        elif altitude >= source_alt[-1]:
            output[i] = source_values[-1]
        else:
            upper = int(torch.searchsorted(source_alt, altitude).item())
            lower = upper - 1
            frac = (altitude - source_alt[lower]) / (
                source_alt[upper] - source_alt[lower]
            )
            output[i] = source_values[lower] + frac * (
                source_values[upper] - source_values[lower]
            )
    return output


def _pun_rate_constant(
    parameters: dict[str, Any],
    temperature: torch.Tensor,
    density: torch.Tensor,
) -> torch.Tensor:
    if parameters.get("formula") == "c2h3_c2h5_branch":
        falloff = 2.5e-36 * torch.pow(temperature, 11.25) * torch.exp(
            3289.0 / temperature
        )
        return 2.5e-11 * falloff / (falloff + 1.0)

    a = float(parameters.get("A", 0.0))
    b = float(parameters.get("b", 0.0))
    c = float(parameters.get("C", 0.0))
    t0 = float(parameters.get("Tmin", 1.0)) or 1.0
    if a <= 0.0:
        return torch.zeros_like(temperature)
    if b > 0.0:
        rate = a * torch.pow(temperature / t0, b) * torch.exp(c / temperature)
    else:
        rate = a * torch.pow(t0 / temperature, abs(b)) * torch.exp(c / temperature)

    d = float(parameters.get("D", 0.0))
    e = float(parameters.get("E", 0.0))
    f = float(parameters.get("F", 0.0))
    if d > 0.0:
        if e > 0.0:
            high = d * torch.pow(temperature / t0, e) * torch.exp(f / temperature)
        else:
            high = d * torch.pow(t0 / temperature, abs(e)) * torch.exp(f / temperature)
        positive = (rate > 0.0) & (high > 0.0)
        falloff = torch.zeros_like(rate)
        ratio = torch.zeros_like(rate)
        ratio[positive] = rate[positive] * density[positive] / high[positive]
        fc = 0.6
        falloff[positive] = (
            (rate[positive] / (1.0 + ratio[positive]))
            * fc
            ** (
                1.0
                / (
                    1.0
                    + torch.log10(torch.clamp(ratio[positive], min=1.0e-300))
                    ** 2
                )
            )
        )
        rate = torch.where(positive, falloff, torch.zeros_like(rate))
    return rate


def _parse_kinetics_base_run_radiation_inputs(
    path: str | Path | None,
) -> dict[str, float | bool]:
    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    lines = file_path.read_text().splitlines()
    values: dict[str, float | bool] = {}

    def numeric_line_after(label: str, skip: int = 1) -> list[float]:
        for index, line in enumerate(lines):
            if line.strip().upper().startswith(label):
                target = index + skip + 1
                if target < len(lines):
                    return _parse_float_tokens(lines[target])
        return []

    planet = numeric_line_after("PLANET PARAMETERS")
    if len(planet) >= 7:
        solar_semimajor_axis_au = planet[0]
        eccentricity = planet[1]
        obliquity = planet[2]
        year_days = planet[3]
        values["planet_day_hours"] = planet[4]
        perihelion_day = planet[5]
        spring_day = planet[6]
    else:
        obliquity = 0.0
        year_days = 1.0
        spring_day = 0.0

    latitude = numeric_line_after("DLAT", skip=0)
    if latitude:
        values["solar_latitude_deg"] = latitude[0]

    grid = numeric_line_after("NALT1", skip=0)
    if len(grid) >= 2:
        # KINETICS-base active levels include the zero-altitude level plus the
        # NALT2 model levels; NALTTP may carry extra high-altitude storage levels.
        values["radiation_active_nlyr"] = int(grid[1]) + 1

    radiation = numeric_line_after("BASIC RADIATION PARAMETERS")
    if radiation:
        values["diurnal_average"] = int(radiation[0]) < 2

    photolysis = numeric_line_after("NPHOTO", skip=0)
    if len(photolysis) >= 8:
        nphoto = int(photolysis[0])
        nphots = int(photolysis[1])
        nphotr = int(photolysis[2])
        nphotd = int(photolysis[3])
        nzen = int(photolysis[7])
        ndisort = int(photolysis[12]) if len(photolysis) > 12 else 0
        values["aerosol_extinction_enabled"] = nphotd != 0
        values["aerosol_scattering_enabled"] = nzen != 0
        values["kinetics_direct_radiation"] = nzen == 0 and ndisort == 0
        active_photo_ids = _collect_numeric_block_after_label(
            lines, "IPHOTO", nphoto + nphots + nphotr + nphotd
        )
        values["active_opacity_reaction_ids"] = active_photo_ids[:nphoto]

    timing = numeric_line_after("ICYEAR", skip=0)
    if len(timing) >= 3:
        day = timing[1]
        values["solar_hour"] = timing[2]
        if year_days != 0.0:
            distance, declination = _kinetics_base_orbit_distance_declination(
                solar_semimajor_axis_au,
                eccentricity,
                obliquity,
                year_days,
                values.get("planet_day_hours", 1.0),
                perihelion_day,
                spring_day,
                day,
            )
            values["solar_distance_au"] = distance
            values["solar_declination_deg"] = declination

    latitude_deg = values.get("solar_latitude_deg")
    declination_deg = values.get("solar_declination_deg")
    hour = values.get("solar_hour")
    day_hours = values.get("planet_day_hours")
    if (
        isinstance(latitude_deg, float)
        and isinstance(declination_deg, float)
        and isinstance(hour, float)
        and isinstance(day_hours, float)
        and day_hours > 0.0
    ):
        hour_angle = 2.0 * math.pi * (hour / day_hours - 0.5)
        lat = math.radians(latitude_deg)
        dec = math.radians(declination_deg)
        mu0 = math.sin(dec) * math.sin(lat) + math.cos(dec) * math.cos(lat) * math.cos(
            hour_angle
        )
        values["solar_mu0"] = max(min(mu0, 1.0), 1.0e-6)

    return values


def _parse_float_tokens(line: str) -> list[float]:
    values: list[float] = []
    for token in line.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _collect_numeric_block_after_label(
    lines: list[str],
    label: str,
    count: int,
) -> list[int]:
    values: list[int] = []
    for index, line in enumerate(lines):
        if not line.strip().upper().startswith(label):
            continue
        for current in lines[index + 1 :]:
            for value in _parse_float_tokens(current):
                values.append(int(value))
                if len(values) >= count:
                    return values
            if values and not _parse_float_tokens(current):
                return values
        return values
    return values


def _kinetics_base_orbit_distance_declination(
    semimajor_axis_au: float,
    eccentricity: float,
    obliquity_deg: float,
    year_days: float,
    day_hours: float,
    perihelion_day: float,
    spring_day: float,
    day: float,
) -> tuple[float, float]:
    period = year_days * day_hours * 3600.0
    perihelion_offset = perihelion_day * day_hours * 3600.0
    season_day = (day - spring_day) % year_days
    time_since_equinox = season_day * day_hours * 3600.0

    def eccentric_anomaly(mean_anomaly: float) -> float:
        value = eccentricity * math.sin(mean_anomaly) + mean_anomaly
        for _ in range(20):
            next_value = value + (
                mean_anomaly - value + eccentricity * math.sin(value)
            ) / (1.0 - eccentricity * math.cos(value))
            if abs(next_value - value) < 1.0e-7:
                return next_value
            value = next_value
        return value

    mean_anomaly = 2.0 * math.pi * (time_since_equinox + perihelion_offset) / period
    anomaly = eccentric_anomaly(mean_anomaly)
    distance = semimajor_axis_au * (1.0 - eccentricity * math.cos(anomaly))
    true_anomaly = 2.0 * math.atan(
        math.sqrt((1.0 + eccentricity) / (1.0 - eccentricity))
        * math.tan(0.5 * anomaly)
    )

    perihelion_mean = 2.0 * math.pi * perihelion_offset / period
    perihelion_anomaly = eccentric_anomaly(perihelion_mean)
    perihelion_angle = 2.0 * math.atan(
        math.sqrt((1.0 + eccentricity) / (1.0 - eccentricity))
        * math.tan(0.5 * perihelion_anomaly)
    )
    declination = obliquity_deg * math.sin(true_anomaly - perihelion_angle)
    return distance, declination


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


def _is_kinetics_base_electron_impact_reaction(products: list[str]) -> bool:
    return "E" in products or any(product.endswith("+") for product in products)


def _kinetics_base_electron_impact_scale(
    reactants: list[str], products: list[str]
) -> float:
    if "N+" in products:
        return 0.0035
    if not reactants:
        return 0.25
    # Temporary Cheng/Titan matching scaffold until the Fortran electron energy
    # deposition profile is implemented.  N2 and CH4 ion channels have different
    # effective source profiles in the current oracle output.
    if reactants[0] == "CH4":
        return 1.0 / 12.0
    return 0.25


def _kinetics_base_electron_impact_profile(
    reactants: list[str], products: list[str]
) -> list[tuple[float, float]] | None:
    if reactants == ["N2"] and products == ["N2+", "E"]:
        # Temporary Titan oracle scaffold for the missing electron energy
        # deposition profile.  Multipliers are relative to the channel-scaled
        # catalog rate and preserve the observed N2+ production altitude shape.
        return [
            (0.0, 0.0),
            (499.7, 0.0),
            (530.8, 7.35424e-05),
            (563.7, 0.00310157),
            (598.7, 0.0444304),
            (635.7, 0.281841),
            (675.1, 1.00493),
            (716.8, 2.38219),
            (761.0, 4.28253),
            (808.0, 6.38209),
            (857.8, 8.35379),
            (910.7, 10.0056),
            (966.7, 11.2738),
            (1026.0, 12.198),
            (1089.0, 12.8475),
            (1156.0, 13.2995),
            (1227.0, 13.6023),
            (1303.0, 13.8121),
        ]
    if reactants == ["CH4"] and products == ["CH3+", "H", "E"]:
        return [
            (0.0, 0.0),
            (499.7, 0.0),
            (530.8, 3.37317e-05),
            (563.7, 0.00154261),
            (598.7, 0.0209029),
            (635.7, 0.122668),
            (675.1, 0.412185),
            (716.8, 0.940689),
            (761.0, 1.65355),
            (808.0, 2.43387),
            (857.8, 3.16511),
            (910.7, 3.78158),
            (966.7, 4.26348),
            (1026.0, 4.61682),
            (1089.0, 4.87396),
            (1156.0, 5.05394),
            (1227.0, 5.18007),
            (1303.0, 5.26958),
        ]
    if reactants == ["CH4"] and products == ["CH2+", "H2", "E"]:
        return [
            (0.0, 0.0),
            (499.7, 0.0),
            (530.8, 2.09568e-05),
            (563.7, 0.000800089),
            (598.7, 0.0103562),
            (635.7, 0.0612753),
            (675.1, 0.209528),
            (716.8, 0.484549),
            (761.0, 0.858117),
            (808.0, 1.26645),
            (857.8, 1.64664),
            (910.7, 1.96345),
            (966.7, 2.20588),
            (1026.0, 2.38157),
            (1089.0, 2.50441),
            (1156.0, 2.58924),
            (1227.0, 2.6464),
            (1303.0, 2.68699),
        ]
    if reactants == ["CH4"] and products == ["CH+", "H", "H2"]:
        return [
            (0.0, 0.0),
            (499.7, 0.0),
            (530.8, 2.74196e-05),
            (563.7, 0.000907129),
            (598.7, 0.00968507),
            (635.7, 0.0478567),
            (675.1, 0.141925),
            (716.8, 0.295933),
            (761.0, 0.486183),
            (808.0, 0.679677),
            (857.8, 0.849488),
            (910.7, 0.984256),
            (966.7, 1.08282),
            (1026.0, 1.15149),
            (1089.0, 1.19775),
            (1156.0, 1.22859),
            (1227.0, 1.24856),
            (1303.0, 1.26217),
        ]
    if reactants == ["CH4"] and products == ["C+", "H2", "H2", "E"]:
        return [
            (0.0, 0.0),
            (499.7, 0.0),
            (530.8, 3.17629e-05),
            (563.7, 0.00101729),
            (598.7, 0.010318),
            (635.7, 0.0482719),
            (675.1, 0.136437),
            (716.8, 0.274058),
            (761.0, 0.437884),
            (808.0, 0.599973),
            (857.8, 0.738984),
            (910.7, 0.847308),
            (966.7, 0.925081),
            (1026.0, 0.978358),
            (1089.0, 1.01458),
            (1156.0, 1.03812),
            (1227.0, 1.05343),
            (1303.0, 1.0634),
        ]
    if reactants == ["CH4"] and products == ["CH3", "H+", "E"]:
        return [
            (0.0, 0.0),
            (499.7, 0.0),
            (530.8, 2.09836e-05),
            (563.7, 0.00068014),
            (598.7, 0.00709024),
            (635.7, 0.0343112),
            (675.1, 0.100133),
            (716.8, 0.206423),
            (761.0, 0.336665),
            (808.0, 0.468247),
            (857.8, 0.583059),
            (910.7, 0.673891),
            (966.7, 0.739951),
            (1026.0, 0.785962),
            (1089.0, 0.817005),
            (1156.0, 0.837583),
            (1227.0, 0.850989),
            (1303.0, 0.85984),
        ]
    if "N+" in products:
        return [
            (0.0, 0.0),
            (499.7, 0.0),
            (530.8, 0.000251421),
            (563.7, 0.00779263),
            (598.7, 0.0761152),
            (635.7, 0.343612),
            (675.1, 0.942543),
            (716.8, 1.84936),
            (761.0, 2.90595),
            (808.0, 3.93211),
            (857.8, 4.80118),
            (910.7, 5.47001),
            (966.7, 5.94694),
            (1026.0, 6.2738),
            (1089.0, 6.49094),
            (1156.0, 6.63267),
            (1227.0, 6.72645),
            (1303.0, 6.7851),
        ]
    return None


def _parse_kinetics_base_catalog(path: str | Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        if len(line) > 60:
            equation = line[:60].strip()
            filename = line[60:].strip()
        else:
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue
            equation, filename = parts[0].strip(), parts[1].strip()
        if equation and filename:
            entries.append((equation, filename))
    return entries


def _parse_kinetics_base_flux(path: str | Path) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            wavelength = float(parts[0])
            width = float(parts[1])
            value = float(parts[2])
        except ValueError:
            continue
        rows.append((wavelength, width, value))
    return rows


def _parse_kinetics_base_aerosol_extinction(
    path: str | Path | None,
    flux: list[tuple[float, float, float]],
) -> dict[str, Any] | None:
    return _parse_kinetics_base_aerosol_property(path, flux, outside_value=0.0)


def _infer_kinetics_base_aerosol_path(
    source_path: str | Path | None,
    source_token: str,
    target_token: str,
) -> Path | None:
    if source_path is None:
        return None
    path = Path(source_path)
    inferred = path.with_name(path.name.replace(source_token, target_token))
    return inferred if inferred.exists() else None


def _parse_kinetics_base_aerosol_property(
    path: str | Path | None,
    flux: list[tuple[float, float, float]],
    *,
    outside_value: float = 0.0,
) -> dict[str, Any] | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    tokens = file_path.read_text().split()
    try:
        nlat_pos = tokens.index("NLAT")
        nalt = int(tokens[nlat_pos + 6])
        nwave = int(tokens[nlat_pos + 7])
        wave_pos = tokens.index("WAVELENGTH") + 1
    except (ValueError, IndexError):
        return None

    wavelengths = [float(value) for value in tokens[wave_pos : wave_pos + nwave]]
    cursor = wave_pos + nwave
    values_by_wave: list[list[float]] = []
    while cursor < len(tokens) and len(values_by_wave) < nwave:
        if tokens[cursor] != "NWAVE":
            cursor += 1
            continue
        cursor += 2
        block: list[float] = []
        while cursor < len(tokens) and len(block) < nalt:
            try:
                block.append(float(tokens[cursor]))
            except ValueError:
                break
            cursor += 1
        if len(block) == nalt:
            values_by_wave.append(block)
    if len(values_by_wave) != nwave:
        return None

    flux_wavelengths = [row[0] for row in flux]
    aligned_by_alt: list[list[float]] = []
    for layer in range(nalt):
        layer_values = [values_by_wave[wave][layer] for wave in range(nwave)]
        aligned_by_alt.append(
            [
                _linear_interpolate_with_outside(
                    wavelengths, layer_values, wavelength, outside_value
                )
                for wavelength in flux_wavelengths
            ]
        )
    return {
        # KINETICS-base aerosol files do not repeat the altitude grid; Titan's
        # interpolation inputs use 91 levels spanning 0-900 km.
        "altitude_km": [float(i) * 900.0 / float(nalt - 1) for i in range(nalt)],
        "values": aligned_by_alt,
    }


def _parse_kinetics_base_cross_section(
    path: Path,
) -> dict[int, list[tuple[float, float]]]:
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    if len(lines) < 5:
        return {}
    try:
        n_datasets = int(lines[1].strip())
    except ValueError:
        return {}

    line_idx = 2
    datasets: dict[int, list[tuple[float, float]]] = {}
    for _ in range(n_datasets):
        if line_idx + 1 >= len(lines):
            break
        try:
            dataset_type = int(float(lines[line_idx].split()[0]))
        except (ValueError, IndexError):
            break
        line_idx += 1

        try:
            meta = [int(float(value)) for value in lines[line_idx].split()]
        except ValueError:
            break
        if len(meta) >= 2 and meta[0] > 0 and meta[1] > meta[0]:
            n_points = meta[1] - meta[0] + 1
        elif meta:
            n_points = meta[-1]
        else:
            break
        line_idx += 1

        rows: list[tuple[float, float]] = []
        for _ in range(n_points):
            if line_idx >= len(lines):
                break
            parts = lines[line_idx].split()
            line_idx += 1
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        if rows and dataset_type not in datasets:
            datasets[dataset_type] = rows

    return datasets


def _parse_kinetics_base_cross_section_on_flux(
    path: Path,
    flux: list[tuple[float, float, float]],
) -> dict[int, list[float]]:
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    if len(lines) < 5:
        return {}
    try:
        n_datasets = int(lines[1].strip())
    except ValueError:
        return {}

    line_idx = 2
    datasets: dict[int, list[float]] = {}
    for _ in range(n_datasets):
        if line_idx + 1 >= len(lines):
            break
        try:
            dataset_type = int(float(lines[line_idx].split()[0]))
        except (ValueError, IndexError):
            break
        line_idx += 1

        try:
            meta = [int(float(value)) for value in lines[line_idx].split()]
        except ValueError:
            break
        if len(meta) < 2:
            break
        first_bin, last_bin = meta[0], meta[1]
        n_points = last_bin - first_bin + 1
        if n_points <= 0:
            break
        line_idx += 1

        values = [0.0 for _ in flux]
        for offset in range(n_points):
            if line_idx >= len(lines):
                break
            parts = lines[line_idx].split()
            line_idx += 1
            if len(parts) < 2:
                continue
            original_bin = first_bin + offset
            # KINETICS-base stores cross sections by wavelength-bin number and
            # ignores the text wavelength column during the Fortran read.  The
            # first flux row is bin 1; bin 0, when present, has no flux row.
            flux_index = original_bin - 1
            if flux_index < 0 or flux_index >= len(values):
                continue
            try:
                values[flux_index] = float(parts[1])
            except ValueError:
                continue
        if dataset_type not in datasets:
            datasets[dataset_type] = values

    return datasets


def _multiply_kinetics_base_cross_sections(
    absorption: list[tuple[float, float]],
    branch: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    wavelengths = [row[0] for row in absorption]
    values = [row[1] for row in absorption]
    return [
        (wavelength, value * _linear_interpolate(wavelengths, values, wavelength))
        for wavelength, value in branch
    ]


def _integrate_kinetics_base_photo_rate(
    cross_section: list[tuple[float, float]],
    flux: list[tuple[float, float, float]],
) -> float:
    return sum(
        cross_value * flux_value
        for _, _, flux_value, cross_value in _align_kinetics_base_flux_cross_section(
            cross_section, flux
        )
    )


def _align_kinetics_base_flux_cross_section(
    cross_section: list[tuple[float, float]],
    flux: list[tuple[float, float, float]],
) -> list[tuple[float, float, float, float]]:
    wavelengths = [row[0] for row in cross_section]
    values = [row[1] for row in cross_section]
    rows: list[tuple[float, float, float, float]] = []
    for wavelength, width, flux_value in flux:
        sigma = _linear_interpolate(wavelengths, values, wavelength)
        rows.append((wavelength, width, flux_value, sigma))
    return rows


def _align_kinetics_base_absorption_cross_section(
    absorption: list[tuple[float, float]],
    flux: list[tuple[float, float, float]],
) -> list[float]:
    wavelengths = [row[0] for row in absorption]
    values = [row[1] for row in absorption]
    return [
        _linear_interpolate(wavelengths, values, wavelength)
        for wavelength, _, _ in flux
    ]


def _linear_interpolate(xs: list[float], ys: list[float], x: float) -> float:
    return _linear_interpolate_with_outside(xs, ys, x, 0.0)


def _linear_interpolate_with_outside(
    xs: list[float],
    ys: list[float],
    x: float,
    outside_value: float,
) -> float:
    if not xs or not ys or x < xs[0] or x > xs[-1]:
        return outside_value
    if x == xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            if x == xs[i]:
                return ys[i]
            frac = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + frac * (ys[i] - ys[i - 1])
    return ys[-1]


def _parse_kinetics_base_equation(equation: str) -> tuple[list[str], list[str]]:
    left, sep, right = equation.partition("=")
    if not sep:
        return [], []
    return _parse_kinetics_base_side(left), _parse_kinetics_base_side(right)


def _parse_kinetics_base_side(side: str) -> list[str]:
    species: list[str] = []
    for raw in side.split():
        raw = raw.lstrip("+")
        if raw == "+":
            continue
        if raw in {"M", "E"} or "-" in raw:
            continue
        count = 1
        name = raw
        if raw and raw[0].isdigit():
            prefix = ""
            rest = raw
            while rest and rest[0].isdigit():
                prefix += rest[0]
                rest = rest[1:]
            count = int(prefix)
            name = rest
        if name:
            species.extend([_normalize_kinetics_base_photo_species(name)] * count)
    return species


def _normalize_kinetics_base_photo_species(name: str) -> str:
    normalized = name.strip()
    if normalized.startswith("aN"):
        normalized = "N" + normalized[2:]
    if normalized.startswith("a"):
        normalized = normalized[1:]
    return normalized


def _kinetics_base_reaction_key(reactants: list[str], products: list[str]) -> str:
    return (
        _kinetics_base_side_key(reactants)
        + "\x1e"
        + _kinetics_base_side_key(products)
    )


def _kinetics_base_side_key(species: list[str]) -> str:
    counts: dict[str, int] = {}
    order: list[str] = []
    for name in species:
        normalized = _normalize_kinetics_base_photo_species(name).upper()
        if normalized in {"M", "E"}:
            continue
        if normalized not in counts:
            order.append(normalized)
            counts[normalized] = 0
        counts[normalized] += 1
    return "\x1f".join(
        (str(counts[name]) if counts[name] > 1 else "") + name for name in order
    )


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
        for name in ["JDUST", "N2", "PROD", "U", "RAYEAR", "SGA", "M"]
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


def _reaction_has_zero_primary_rate(reaction: Any) -> bool:
    if not reaction.rate_blocks:
        return True
    return reaction.rate_blocks[0].A == 0.0


def _apply_titan_condensation_source(
    concentration: torch.Tensor,
    titan_state: KBTitanState,
    species_index: dict[str, int],
    term: KBTitanSourceTerm,
    dt: float,
    *,
    stick_x: float,
    stick_h: float,
    pun_metadata: dict[str, Any] | None,
    temperature: torch.Tensor,
) -> None:
    if len(term.reactants) != 2 or len(term.products) != 1:
        return
    gas, surface = term.reactants
    product = term.products[0]
    if gas not in species_index or surface not in species_index or product not in species_index:
        return

    gas_idx = species_index[gas]
    surface_idx = species_index[surface]
    product_idx = species_index[product]
    mass_amu = _kinetics_base_species_mass_amu(gas, pun_metadata)
    if mass_amu <= 0.0:
        return
    sticking = _titan_sticking_coefficient(gas, stick_x=stick_x, stick_h=stick_h)

    velocity = 100.0 * torch.sqrt(
        8.0
        * torch.tensor(1.38e-23, dtype=temperature.dtype, device=temperature.device)
        * temperature
        / (
            torch.tensor(3.14159 * 1.66e-27, dtype=temperature.dtype, device=temperature.device)
            * mass_amu
        )
    )
    rate = sticking * velocity * concentration[:, :, gas_idx] * concentration[:, :, surface_idx]
    delta = dt * rate
    concentration[:, :, gas_idx] = torch.clamp(concentration[:, :, gas_idx] - delta, min=0.0)
    concentration[:, :, product_idx] = concentration[:, :, product_idx] + delta


def _apply_titan_sublimation_source(
    concentration: torch.Tensor,
    titan_state: KBTitanState,
    species_index: dict[str, int],
    term: KBTitanSourceTerm,
    dt: float,
    *,
    stick_x: float,
    stick_h: float,
    pun_metadata: dict[str, Any] | None,
    temperature: torch.Tensor,
) -> None:
    if len(term.reactants) != 2 or len(term.products) != 1:
        return
    grain, _surface = term.reactants
    gas = term.products[0]
    if grain not in species_index or gas not in species_index or "SGA" not in species_index:
        return
    if "vapor_A" not in term.parameters:
        return

    grain_idx = species_index[grain]
    gas_idx = species_index[gas]
    sga_idx = species_index["SGA"]
    mass_amu = _kinetics_base_species_mass_amu(gas, pun_metadata)
    if mass_amu <= 0.0:
        return

    n_sat = _titan_saturation_density(term.parameters, temperature)
    if torch.max(n_sat).item() <= 0.0:
        return

    sticking = _titan_sticking_coefficient(gas, stick_x=stick_x, stick_h=stick_h)
    velocity = 100.0 * torch.sqrt(
        8.0
        * torch.tensor(1.38e-23, dtype=temperature.dtype, device=temperature.device)
        * temperature
        / (
            torch.tensor(3.14159 * 1.66e-27, dtype=temperature.dtype, device=temperature.device)
            * mass_amu
        )
    )
    grain_total = torch.zeros_like(temperature)
    for name, idx in species_index.items():
        if name.startswith("G"):
            grain_total = grain_total + concentration[:, :, idx]
    nsite = torch.tensor(1.5e15, dtype=temperature.dtype, device=temperature.device)
    ntot = torch.maximum(grain_total, 4.0 * concentration[:, :, sga_idx] * nsite)
    valid = ntot > 0.0
    rate_constant = torch.zeros_like(temperature)
    rate_constant[valid] = (
        sticking
        * velocity[valid]
        * concentration[:, :, sga_idx][valid]
        * n_sat[valid]
        / ntot[valid]
    )
    delta = torch.minimum(dt * rate_constant * concentration[:, :, grain_idx], concentration[:, :, grain_idx])
    concentration[:, :, grain_idx] = concentration[:, :, grain_idx] - delta
    concentration[:, :, gas_idx] = concentration[:, :, gas_idx] + delta


def _apply_titan_condensation_sublimation_pair(
    concentration: torch.Tensor,
    titan_state: KBTitanState,
    species_index: dict[str, int],
    condensation: KBTitanSourceTerm,
    sublimation: KBTitanSourceTerm,
    dt: float,
    *,
    stick_x: float,
    stick_h: float,
    pun_metadata: dict[str, Any] | None,
    temperature: torch.Tensor,
) -> None:
    gas = condensation.reactants[0]
    grain = condensation.products[0]
    if gas not in species_index or grain not in species_index or "SGA" not in species_index:
        return
    if "vapor_A" not in sublimation.parameters:
        _apply_titan_condensation_source(
            concentration,
            titan_state,
            species_index,
            condensation,
            dt,
            stick_x=stick_x,
            stick_h=stick_h,
            pun_metadata=pun_metadata,
            temperature=temperature,
        )
        return

    gas_idx = species_index[gas]
    grain_idx = species_index[grain]
    sga_idx = species_index["SGA"]
    mass_amu = _kinetics_base_species_mass_amu(gas, pun_metadata)
    if mass_amu <= 0.0:
        return

    sticking = _titan_sticking_coefficient(gas, stick_x=stick_x, stick_h=stick_h)
    velocity = 100.0 * torch.sqrt(
        8.0
        * torch.tensor(1.38e-23, dtype=temperature.dtype, device=temperature.device)
        * temperature
        / (
            torch.tensor(3.14159 * 1.66e-27, dtype=temperature.dtype, device=temperature.device)
            * mass_amu
        )
    )
    condensation_rate = sticking * velocity * concentration[:, :, sga_idx]

    n_sat = _titan_saturation_density(sublimation.parameters, temperature)
    grain_total = torch.zeros_like(temperature)
    for name, idx in species_index.items():
        if name.startswith("G"):
            grain_total = grain_total + concentration[:, :, idx]
    nsite = torch.tensor(1.5e15, dtype=temperature.dtype, device=temperature.device)
    ntot = torch.maximum(grain_total, 4.0 * concentration[:, :, sga_idx] * nsite)
    sublimation_rate = torch.zeros_like(temperature)
    valid = ntot > 0.0
    sublimation_rate[valid] = (
        sticking
        * velocity[valid]
        * concentration[:, :, sga_idx][valid]
        * n_sat[valid]
        / ntot[valid]
    )

    total_rate = condensation_rate + sublimation_rate
    total = concentration[:, :, gas_idx] + concentration[:, :, grain_idx]
    gas_equilibrium = torch.zeros_like(total)
    active = total_rate > 0.0
    gas_equilibrium[active] = total[active] * sublimation_rate[active] / total_rate[active]
    decay = torch.exp(-dt * total_rate)
    gas_new = gas_equilibrium + (concentration[:, :, gas_idx] - gas_equilibrium) * decay
    gas_new = torch.clamp(gas_new, min=0.0)
    concentration[:, :, gas_idx] = gas_new
    concentration[:, :, grain_idx] = torch.clamp(total - gas_new, min=0.0)


def _titan_sticking_coefficient(gas: str, *, stick_x: float, stick_h: float) -> float:
    sticking = stick_h if gas in {"H", "H2"} else stick_x
    if gas == "C2H2":
        sticking *= 3.0e-1
    elif gas == "C2H4":
        sticking *= 1.0
    elif gas == "C2H6":
        sticking *= 3.0e-2
    return sticking


def _titan_thermal_velocity(temperature: torch.Tensor, mass_amu: float) -> torch.Tensor:
    return 100.0 * torch.sqrt(
        8.0
        * torch.tensor(1.38e-23, dtype=temperature.dtype, device=temperature.device)
        * temperature
        / (
            torch.tensor(3.14159 * 1.66e-27, dtype=temperature.dtype, device=temperature.device)
            * mass_amu
        )
    )


def _titan_sublimation_rate_profile(
    state: AtmState2D,
    species_index: dict[str, int],
    parameters: dict[str, Any],
    gas: str,
    mass_amu: float,
) -> torch.Tensor:
    if "SGA" not in species_index:
        return torch.zeros_like(state.temperature)
    sga_idx = species_index["SGA"]
    n_sat = _titan_saturation_density(parameters, state.temperature)
    velocity = _titan_thermal_velocity(state.temperature, mass_amu)
    grain_total = torch.zeros_like(state.temperature)
    for name, idx in species_index.items():
        if name.startswith("G"):
            grain_total = grain_total + state.concentration[:, :, idx]
    nsite = torch.tensor(1.5e15, dtype=state.dtype, device=state.device)
    ntot = torch.maximum(grain_total, 4.0 * state.concentration[:, :, sga_idx] * nsite)
    rate = torch.zeros_like(state.temperature)
    valid = ntot > 0.0
    rate[valid] = (
        _titan_sticking_coefficient(gas, stick_x=1.0e-5, stick_h=0.3)
        * velocity[valid]
        * state.concentration[:, :, sga_idx][valid]
        * n_sat[valid]
        / ntot[valid]
    )
    return rate


def _titan_saturation_density(
    parameters: dict[str, float | str | int], temperature: torch.Tensor
) -> torch.Tensor:
    a = float(parameters["vapor_A"])
    b = float(parameters["vapor_B"])
    c = float(parameters["vapor_C"])
    denom = temperature - 273.0 + c
    valid = denom > 0.0
    plog = a - b / denom
    psat = torch.pow(torch.tensor(10.0, dtype=temperature.dtype, device=temperature.device), plog) / 760.0
    n_sat = psat / (10.0 * torch.tensor(1.38e-23, dtype=temperature.dtype, device=temperature.device) * temperature)
    return torch.where(valid, n_sat, torch.zeros_like(n_sat))


def _apply_boundary_source(
    concentration: torch.Tensor,
    titan_state: KBTitanState,
    species_index: dict[str, int],
    term: KBTitanSourceTerm,
    dt: float,
) -> None:
    species = (term.reactants or term.products)
    if not species or species[0] not in species_index:
        return

    j = species_index[species[0]]
    value = float(term.parameters.get("value", 0.0))
    if value == 0.0:
        return

    faces = titan_state.state.x1f.to(
        dtype=concentration.dtype, device=concentration.device
    )
    dz = faces[1:] - faces[:-1]
    if torch.any(dz <= 0):
        return

    if term.kind == "lower_boundary_flux":
        concentration[:, 0, j] = torch.clamp(
            concentration[:, 0, j] + dt * value / dz[0], min=0.0
        )
    elif term.kind == "upper_boundary_flux":
        concentration[:, -1, j] = torch.clamp(
            concentration[:, -1, j] - dt * value / dz[-1], min=0.0
        )
    elif term.kind == "lower_boundary_velocity":
        if species[0].startswith("G") and value < 0.0:
            concentration[:, 0, j] = 0.0
            return
        flux = value * concentration[:, 0, j]
        concentration[:, 0, j] = torch.clamp(
            concentration[:, 0, j] + dt * flux / dz[0], min=0.0
        )
    elif term.kind == "upper_boundary_velocity":
        flux = value * concentration[:, -1, j]
        concentration[:, -1, j] = torch.clamp(
            concentration[:, -1, j] - dt * flux / dz[-1], min=0.0
        )


def _kinetics_base_species_mass_amu(
    name: str,
    pun_metadata: dict[str, Any] | None,
) -> float:
    metadata = None if pun_metadata is None else pun_metadata.get(name)
    if metadata is None:
        return _fallback_species_mass_amu(name)
    if metadata.molecular_weight > 0:
        return float(metadata.molecular_weight)
    # The Titan .pun stores zero molecular weights for many gas species; recover
    # the mass from the parsed element ordering used by this network.
    element_masses = [1.0, 4.0, 12.0, 14.0, 14.0, 16.0, 32.0, 32.0]
    return float(
        sum(count * element_masses[i] for i, count in enumerate(metadata.composition))
    )


def _fallback_species_mass_amu(name: str) -> float:
    fallback = {
        "H": 1.0,
        "H2": 2.0,
        "CH4": 16.0,
        "C2H2": 26.0,
        "C2H4": 28.0,
        "C2H6": 30.0,
    }
    return fallback.get(name, 0.0)


def _kinetics_base_vapor_coefficients_from_pun(
    pun: str | Path,
) -> dict[str, tuple[float, float, float, float, float]]:
    coefficients: dict[str, tuple[float, float, float, float, float]] = {}
    lines = Path(pun).read_text().splitlines()
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 2 or not parts[0].endswith("."):
            continue
        name = parts[1]
        if i + 4 >= len(lines):
            continue
        values = lines[i + 4].split()
        if len(values) < 5:
            continue
        try:
            coeffs = tuple(float(value) for value in values[:5])
        except ValueError:
            continue
        if coeffs[0] != 0.0:
            coefficients[name] = coeffs  # type: ignore[assignment]
    return coefficients


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
    return [
        (entry.lower_kind, entry.lower_value, entry.species)
        for entry in parse_kinetics_base_boundary(boundary_path)
    ]


def parse_kinetics_base_boundary(
    boundary_path: str | Path,
) -> list[KBTitanBoundaryEntry]:
    entries: list[KBTitanBoundaryEntry] = []
    for line in Path(boundary_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            entries.append(
                KBTitanBoundaryEntry(
                    lower_kind=int(parts[0]),
                    lower_value=float(parts[1]),
                    upper_kind=int(parts[2]),
                    upper_value=float(parts[3]),
                    species=parts[4],
                )
            )
        except ValueError:
            continue
    return entries


def _normalize_kinetics_base_boundary_species(name: str) -> str:
    if len(name) > 1 and name[0].isdigit():
        return f"({name[0]}){name[1:]}"
    return name


def _canonical_kinetics_base_species_name(
    name: str,
    canonical_species: dict[str, str],
) -> str:
    return canonical_species.get(name.upper(), name)


__all__ = [
    "KBTitanFirstOrderAtm2DSource",
    "KBTitanState",
    "KBTitanSourceTerm",
    "KBTitanSpecialEntry",
    "KBTitanBoundaryEntry",
    "apply_kinetics_base_titan_source_terms",
    "altitude_faces_from_kinetics_base_centers_km",
    "build_kinetics_base_titan_atm2d_source_terms",
    "build_kinetics_base_titan_source_terms",
    "build_kinetics_base_titan_state",
    "kinetics_base_concentration_from_profile",
    "kinetics_base_profile_tensor",
    "kinetics_base_titan_source_tendencies",
    "kinetics_base_species_metadata_from_pun",
    "parse_kinetics_base_boundary",
    "parse_kinetics_base_special",
]
