from __future__ import annotations

import torch

from .models import KBTitanSourceTerm, KBTitanState
from .physics import (
    _kinetics_base_species_mass_amu,
    _pun_rate_constant,
    _titan_saturation_density,
    _titan_sticking_coefficient,
    _titan_sublimation_rate_profile,
    _titan_thermal_velocity,
)
from .radiation import _kinetics_base_pyharp_actinic_flux


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
        term
        for term in source_terms
        if term.kind
        in {
            "pun_thermal_reaction",
            "pun_ion_mass_action_reaction",
            "pun_dissociative_recombination",
        }
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
            if _is_ch4_grain_loading(term):
                _apply_titan_product_only_condensation_source(
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
                continue
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
            if _is_ch4_shifted_grain_release(term):
                _apply_titan_shifted_sublimation_source(
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
        product_coefficients: dict[int, int] = {}
        missing_product = False
        for name in term.products:
            if name not in species_index:
                missing_product = True
                continue
            index = species_index[name]
            product_coefficients[index] = product_coefficients.get(index, 0) + 1
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
        for product, coefficient in product_coefficients.items():
            production[:, :, product] = production[:, :, product] + coefficient * rate

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
    opacity_concentration = concentration
    if bool(term.parameters.get("freeze_actinic_flux", False)):
        initial_concentration = titan_state.concentration
        if initial_concentration.shape == concentration.shape:
            opacity_concentration = initial_concentration.to(dtype=dtype, device=device)
    actinic_flux = _kinetics_base_pyharp_actinic_flux(
        term,
        titan_state,
        opacity_concentration,
        species_index,
        flux_tensor,
        reaction_sigma.numel(),
        dtype,
        device,
    )
    if actinic_flux is None:
        return None

    # Secondary electron impact ionization: when set, each absorbed photon
    # creates (1 + n_sec) ion-electron pairs, where n_sec = (E_γ - threshold)/W
    # is the number of secondary ionizations a primary electron at kinetic
    # energy (E_γ - threshold) can create before thermalizing. W is the mean
    # energy per ion pair (Cravens-style). Wavelengths are in Ångstroms
    # (12400 eV·Å / λ_Å = photon energy in eV).
    secondary = term.parameters.get("secondary_impact")
    if isinstance(secondary, dict):
        try:
            threshold_eV = float(secondary["threshold_eV"])
            W_eV = float(secondary["W_eV"])
        except (KeyError, TypeError, ValueError):
            threshold_eV = None
            W_eV = None
        if threshold_eV is not None and W_eV is not None and W_eV > 0.0:
            wl_tensor = torch.tensor(wavelengths, dtype=dtype, device=device)
            safe_wl = torch.clamp(wl_tensor, min=0.1)
            photon_energy_eV = 12400.0 / safe_wl
            excess = torch.clamp(photon_energy_eV - threshold_eV, min=0.0)
            n_sec = excess / W_eV
            secondary_factor = 1.0 + n_sec
            return (
                actinic_flux
                * reaction_sigma.view(1, 1, -1)
                * secondary_factor.view(1, 1, -1)
            ).sum(dim=-1)

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

def _apply_titan_product_only_condensation_source(
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
    explicit_rate = float(term.parameters.get("A", 0.0))
    if explicit_rate > 0.0:
        rate_constant = explicit_rate * concentration[:, :, surface_idx]
    else:
        sticking = _titan_sticking_coefficient(gas, stick_x=stick_x, stick_h=stick_h)
        rate_constant = (
            sticking
            * _titan_thermal_velocity(temperature, mass_amu)
            * concentration[:, :, surface_idx]
        )
    delta = dt * rate_constant * torch.clamp(concentration[:, :, gas_idx], min=0.0)
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
    nsite = torch.tensor(1.5e15, dtype=temperature.dtype, device=temperature.device)
    ntot = 4.0 * concentration[:, :, sga_idx] * nsite
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

def _apply_titan_shifted_sublimation_source(
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
    if "vapor_A" not in term.parameters or concentration.shape[1] < 2:
        return

    grain_idx = species_index[grain]
    gas_idx = species_index[gas]
    mass_amu = _kinetics_base_species_mass_amu(gas, pun_metadata)
    if mass_amu <= 0.0:
        return
    rate_constant = _titan_sublimation_rate_profile(
        titan_state.state, species_index, term.parameters, gas, mass_amu
    ).to(dtype=concentration.dtype, device=concentration.device)
    delta = torch.minimum(
        dt * rate_constant[:, :-1] * concentration[:, :-1, grain_idx],
        concentration[:, :-1, grain_idx],
    )
    concentration[:, :-1, grain_idx] = concentration[:, :-1, grain_idx] - delta
    concentration[:, 1:, gas_idx] = concentration[:, 1:, gas_idx] + delta

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
    nsite = torch.tensor(1.5e15, dtype=temperature.dtype, device=temperature.device)
    ntot = 4.0 * concentration[:, :, sga_idx] * nsite
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
