from __future__ import annotations

from dataclasses import dataclass

import torch

from ..atm2d import (
    AtmState2D,
    IndexedBoundaryFluxSource,
    IndexedBoundaryVelocitySource,
    IndexedFirstOrderSource,
    IndexedMassActionSource,
    IndexedReversibleFirstOrderSource,
    LocalSourceLinearization,
    LocalSourceTerm,
)
from .models import KBTitanSourceTerm, KBTitanState
from .physics import (
    _kinetics_base_species_mass_amu,
    _pun_rate_constant,
    _titan_sticking_coefficient,
    _titan_sublimation_rate_profile,
    _titan_thermal_velocity,
)
from .source_integration import (
    _photo_rate_profile,
    _rate_profile_multiplier_on_state_grid,
)


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

