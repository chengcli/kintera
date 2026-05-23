from __future__ import annotations

from dataclasses import dataclass

import torch

from ...atm2d import (
    AtmState2D,
    IndexedBoundaryFluxSource,
    IndexedBoundaryVelocitySource,
    IndexedFirstOrderSource,
    IndexedMassActionSource,
    IndexedReversibleFirstOrderSource,
    LocalSourceLinearization,
    LocalSourceTerm,
    SparseSystemMatrix,
)
from ...atm2d.matrix import flatten_state_index
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
            concentration=self.titan_state.concentration,
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
            product_coefficients: dict[int, int] = {}
            missing_product = False
            for name in term.products:
                if name not in species_index:
                    missing_product = True
                    continue
                product = species_index[name]
                product_coefficients[product] = product_coefficients.get(product, 0) + 1
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
            for product, coefficient in product_coefficients.items():
                tendency[:, :, product] = tendency[:, :, product] + coefficient * rate
                jacobian[:, :, product, reactant] = (
                    jacobian[:, :, product, reactant] + coefficient * rate_profile
                )
        return LocalSourceLinearization(tendency=tendency, jacobian=jacobian)


@dataclass
class KBTitanProductOnlyCondensationSource:
    """Titan grain loading source that does not remove the gas parent locally."""

    titan_state: KBTitanState
    term: KBTitanSourceTerm
    pun_metadata: dict[str, Any] | None = None

    def linearize(self, state: AtmState2D) -> LocalSourceLinearization:
        species_index = {name: i for i, name in enumerate(self.titan_state.species)}
        if len(self.term.reactants) != 2 or len(self.term.products) != 1:
            return _zero_linearization(state)
        gas, surface = self.term.reactants
        grain = self.term.products[0]
        if gas not in species_index or surface not in species_index or grain not in species_index:
            return _zero_linearization(state)
        mass_amu = _kinetics_base_species_mass_amu(gas, self.pun_metadata)
        if mass_amu <= 0.0:
            return _zero_linearization(state)

        gas_idx = species_index[gas]
        surface_idx = species_index[surface]
        grain_idx = species_index[grain]
        surface_density = state.concentration[:, :, surface_idx]
        explicit_rate = float(self.term.parameters.get("A", 0.0))
        if explicit_rate > 0.0:
            rate = explicit_rate * surface_density
        else:
            rate = (
                _titan_thermal_velocity(state.temperature, mass_amu)
                * _titan_sticking_coefficient(gas, stick_x=1.0e-5, stick_h=0.3)
                * surface_density
            )
        gas_value = torch.clamp(state.concentration[:, :, gas_idx], min=0.0)
        flux = rate * gas_value

        tendency = torch.zeros_like(state.concentration)
        jacobian = torch.zeros(
            (state.ncol, state.nlyr, state.nspecies, state.nspecies),
            dtype=state.dtype,
            device=state.device,
        )
        tendency[:, :, grain_idx] = tendency[:, :, grain_idx] + flux
        jacobian[:, :, grain_idx, gas_idx] = jacobian[:, :, grain_idx, gas_idx] + rate
        return LocalSourceLinearization(tendency=tendency, jacobian=jacobian)


@dataclass
class KBTitanShiftedSublimationSource:
    """Titan grain release that deposits gas one vertical level above the grain."""

    titan_state: KBTitanState
    term: KBTitanSourceTerm
    pun_metadata: dict[str, Any] | None = None

    def linearize(self, state: AtmState2D) -> LocalSourceLinearization:
        species_index = {name: i for i, name in enumerate(self.titan_state.species)}
        if len(self.term.reactants) != 2 or len(self.term.products) != 1:
            return _zero_linearization(state)
        grain, _surface = self.term.reactants
        gas = self.term.products[0]
        if grain not in species_index or gas not in species_index:
            return _zero_linearization(state)
        if "vapor_A" not in self.term.parameters:
            return _zero_linearization(state)
        mass_amu = _kinetics_base_species_mass_amu(gas, self.pun_metadata)
        if mass_amu <= 0.0:
            return _zero_linearization(state)

        grain_idx = species_index[grain]
        gas_idx = species_index[gas]
        rate = _titan_sublimation_rate_profile(
            state, species_index, self.term.parameters, gas, mass_amu
        )
        grain_value = torch.clamp(state.concentration[:, :, grain_idx], min=0.0)
        flux = rate * grain_value

        tendency = torch.zeros_like(state.concentration)
        jacobian = torch.zeros(
            (state.ncol, state.nlyr, state.nspecies, state.nspecies),
            dtype=state.dtype,
            device=state.device,
        )
        if state.nlyr > 1:
            active_flux = flux[:, :-1]
            active_rate = rate[:, :-1]
            tendency[:, :-1, grain_idx] = tendency[:, :-1, grain_idx] - active_flux
            tendency[:, 1:, gas_idx] = tendency[:, 1:, gas_idx] + active_flux
            jacobian[:, :-1, grain_idx, grain_idx] = (
                jacobian[:, :-1, grain_idx, grain_idx] - active_rate
            )
        return LocalSourceLinearization(tendency=tendency, jacobian=jacobian)

    def global_operator(self, state: AtmState2D) -> SparseSystemMatrix | None:
        species_index = {name: i for i, name in enumerate(self.titan_state.species)}
        if len(self.term.reactants) != 2 or len(self.term.products) != 1:
            return None
        grain, _surface = self.term.reactants
        gas = self.term.products[0]
        if grain not in species_index or gas not in species_index:
            return None
        if "vapor_A" not in self.term.parameters or state.nlyr < 2:
            return None
        mass_amu = _kinetics_base_species_mass_amu(gas, self.pun_metadata)
        if mass_amu <= 0.0:
            return None

        grain_idx = species_index[grain]
        gas_idx = species_index[gas]
        rate = _titan_sublimation_rate_profile(
            state, species_index, self.term.parameters, gas, mass_amu
        )
        active = rate[:, :-1] != 0.0
        nz = active.nonzero(as_tuple=False)
        if nz.numel() == 0:
            return None
        row_ids = flatten_state_index(
            nz[:, 0],
            nz[:, 1] + 1,
            gas_idx,
            state.nlyr,
            state.nspecies,
        ).to(device=state.concentration.device)
        col_ids = flatten_state_index(
            nz[:, 0],
            nz[:, 1],
            grain_idx,
            state.nlyr,
            state.nspecies,
        ).to(device=state.concentration.device)
        values = rate[:, :-1][nz[:, 0], nz[:, 1]]
        return SparseSystemMatrix.from_coo(
            torch.stack([row_ids, col_ids]),
            values,
            ncol=state.ncol,
            nlyr=state.nlyr,
            nspecies=state.nspecies,
            device=state.concentration.device,
            dtype=state.dtype,
        )


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
        if term.kind in {
            "pun_thermal_reaction",
            "pun_ion_mass_action_reaction",
            "pun_dissociative_recombination",
        }:
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
        if _is_ch4_grain_loading(term):
            source = KBTitanProductOnlyCondensationSource(
                titan_state=titan_state,
                term=term,
                pun_metadata=pun_metadata,
            )
            atm_sources.append(source)
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
            if _is_ch4_shifted_grain_release(term):
                source = KBTitanShiftedSublimationSource(
                    titan_state=titan_state,
                    term=term,
                    pun_metadata=pun_metadata,
                )
            else:
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


def _zero_linearization(state: AtmState2D) -> LocalSourceLinearization:
    return LocalSourceLinearization(
        tendency=torch.zeros_like(state.concentration),
        jacobian=torch.zeros(
            (state.ncol, state.nlyr, state.nspecies, state.nspecies),
            dtype=state.dtype,
            device=state.device,
        ),
    )


def _is_ch4_grain_loading(term: KBTitanSourceTerm) -> bool:
    return (
        term.kind == "titan_condensation"
        and term.reactants == ["CH4", "SGA"]
        and term.products == ["GCH4"]
    )


def _is_ch4_shifted_grain_release(term: KBTitanSourceTerm) -> bool:
    return (
        term.kind == "titan_sublimation"
        and term.reactants == ["GCH4", "U"]
        and term.products == ["CH4"]
    )

def _build_titan_thermal_atm2d_source(
    titan_state: KBTitanState,
    species_index: dict[str, int],
    term: KBTitanSourceTerm,
) -> IndexedMassActionSource | None:
    reactants = [species_index[name] for name in term.reactants if name in species_index]
    products = [species_index[name] for name in term.products if name in species_index]
    # Drop the entire reaction if ANY reactant or product is missing from the
    # species set. Previously we kept reactions where only some products were
    # missing — that silently broke mass conservation: e.g.
    # ``2 NH2 + M → bN2H4 + M`` with ``bN2H4`` not in species reduced to
    # ``2 NH2 + M → M``, which destroys NH2 without producing anything.
    # At KB-level [NH2]=4e+9, that fake sink drained 8.6e+8 /cm³/s of NH2,
    # collapsing the NH chain into a low-equilibrium basin.
    if len(reactants) != len(term.reactants) or len(products) != len(term.products):
        return None
    if not products:
        return None
    reactant_coeffs = list(term.parameters.get("reactant_coefficients", []))
    product_coeffs = list(term.parameters.get("product_coefficients", []))
    if len(reactant_coeffs) != len(reactants):
        reactant_coeffs = [1] * len(reactants)
    if len(product_coeffs) != len(products):
        product_coeffs = [1] * len(products)

    # Check for KB UPDATE_CHEMB hand-coded overrides (kinetgen1X.F:6803-7384).
    # Many Titan reactions have Moses 2005 / Cheng 2013 formulas that REPLACE
    # the catalog (.pun-file) rate constant. Failing to apply these gives
    # 10× rate errors (see project-diagnostic-findings memory + the
    # kb-fortran-map skill for the catalog of overrides).
    from .chemb_overrides import has_titan_chemb_override, titan_chemb_rate_constant
    override_id = term.reaction_id if has_titan_chemb_override(term.reaction_id) else None

    def rate_provider(state: AtmState2D) -> torch.Tensor:
        density = titan_state.density.to(dtype=state.dtype, device=state.device)
        if override_id is not None:
            T = state.temperature
            # density tensor needs same shape as T for broadcasting
            d = density.expand_as(T) if density.shape != T.shape else density
            k = titan_chemb_rate_constant(override_id, T, d)
            if k is not None:
                return k
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
) -> IndexedFirstOrderSource | None:
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
        surface_density = state.concentration[:, :, species_index[surface]]
        explicit_rate = float(term.parameters.get("A", 0.0))
        if explicit_rate > 0.0:
            return explicit_rate * surface_density
        return (
            _titan_thermal_velocity(state.temperature, mass_amu)
            * _titan_sticking_coefficient(gas, stick_x=1.0e-5, stick_h=0.3)
            * surface_density
        )

    return IndexedFirstOrderSource(
        reactant=species_index[gas],
        products=[species_index[product]],
        rate=rate_provider,
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
        explicit_rate = float(condensation.parameters.get("A", 0.0))
        if explicit_rate > 0.0:
            return explicit_rate * sga
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

