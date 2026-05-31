from __future__ import annotations

import math
from typing import Any

import torch


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
    # KB clips T to max(T, t0) for the (T/t0)^b factor: when T < t0 (e.g.
    # T=64K at upper atmosphere for a recombination with reference 300K),
    # the power term becomes 1.0 instead of (t0/T)^|b| > 1 — without
    # clipping our model over-rates 5 cation+E recombinations by ~2.5x at
    # cold altitudes. The exp(c/T) Arrhenius factor uses raw T (KB doesn't
    # clip activation-energy attenuation). Confirmed via KB-state-injected
    # prod/loss diff (commit 9c04f75): rxn 776 (CH2+ + E -> C + 2H) at L30
    # was 2.7x over KB before this fix.
    t_eff = torch.where(temperature < t0, torch.full_like(temperature, t0), temperature)
    if b > 0.0:
        rate = a * torch.pow(t_eff / t0, b) * torch.exp(c / temperature)
    else:
        rate = a * torch.pow(t0 / t_eff, abs(b)) * torch.exp(c / temperature)

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

def _reaction_has_zero_primary_rate(reaction: Any) -> bool:
    if not reaction.rate_blocks:
        return True
    return reaction.rate_blocks[0].A == 0.0

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
    nsite = torch.tensor(1.5e15, dtype=state.dtype, device=state.device)
    site_capacity = 4.0 * state.concentration[:, :, sga_idx] * nsite
    ntot = torch.maximum(
        _titan_total_grain_ice_abundance(state, species_index),
        site_capacity,
    )
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


def _titan_total_grain_ice_abundance(
    state: AtmState2D,
    species_index: dict[str, int],
) -> torch.Tensor:
    """Return KB-style ``AIPT9502`` total abundance of G-prefixed ices."""

    total = torch.zeros_like(state.temperature)
    for name, index in species_index.items():
        if _is_titan_grain_ice_species(name):
            total = total + torch.clamp(state.concentration[:, :, index], min=0.0)
    return total


def _is_titan_grain_ice_species(name: str) -> bool:
    return name.startswith("G")

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

def _kinetics_base_species_mass_amu(
    name: str,
    pun_metadata: dict[str, Any] | None,
) -> float:
    # Prefer the name-based fallback table — it is authoritative for the
    # KB Titan networks. The PUN's molecular_weight field is unreliable for
    # moses00 (CH3 reads back as 3.0, C2H6 as 6.0 — H counts are dropped
    # somewhere in the parser/composition pipeline). When the species is
    # NOT in the fallback table, defer to the PUN metadata as a last
    # resort.
    name_mass = _fallback_species_mass_amu(name)
    if name_mass > 0:
        return name_mass
    metadata = None if pun_metadata is None else pun_metadata.get(name)
    if metadata is None:
        return 0.0
    if metadata.molecular_weight > 0:
        return float(metadata.molecular_weight)
    element_masses = [1.0, 4.0, 12.0, 14.0, 14.0, 16.0, 32.0, 32.0]
    return float(
        sum(count * element_masses[i] for i, count in enumerate(metadata.composition))
    )

def _fallback_species_mass_amu(name: str) -> float:
    """Return the species mass in amu from its name.

    Tries (1) a hardcoded table for common Titan-chemistry species, then
    (2) a chemical-formula regex parser for names like ``C2H4`` or
    ``CH3CN``. The parser strips leading positional prefixes (``1-``,
    ``1,3-``) and parenthetical state labels (``(1)CH2``, ``(3)CH2``,
    ``N(2D)``, ``O(1D)``) before matching ``[A-Z][a-z]?[0-9]*`` atom
    tokens. Element masses use IUPAC values for H, He, C, N, O — the
    set used by KB Titan networks.
    """
    fallback = {
        "H": 1.0, "H2": 2.0, "HE": 4.0,
        "C": 12.0, "CH": 13.0, "(1)CH2": 14.0, "(3)CH2": 14.0,
        "CH3": 15.0, "CH4": 16.0,
        "C2": 24.0, "C2H": 25.0, "C2H2": 26.0, "C2H2*": 26.0, "C2H3": 27.0,
        "C2H4": 28.0, "C2H5": 29.0, "C2H6": 30.0,
        "C3": 36.0, "C3H": 37.0, "C3H2": 38.0, "C3H3": 39.0,
        "CH3C2H": 40.0, "CH2CCH2": 40.0, "C3H5": 41.0, "C3H6": 42.0,
        "C3H7": 43.0, "C3H8": 44.0,
        "C4H": 49.0, "C4H2": 50.0, "C4H2*": 50.0, "C4H3": 51.0,
        "C4H4": 52.0, "C4H5": 53.0,
        "1-C4H6": 54.0, "1,2-C4H6": 54.0, "1,3-C4H6": 54.0,
        "C4H8": 56.0, "C4H9": 57.0, "C4H10": 58.0,
        "C6H": 73.0, "C6H2": 74.0, "C6H3": 75.0, "C6H6": 78.0,
        "C8H2": 98.0, "C8H3": 99.0,
        "N": 14.0, "N(2D)": 14.0, "NH": 15.0,
        "HCN": 27.0, "CHCN": 39.0, "CH3CN": 41.0, "HC3N": 51.0,
        "C3N": 50.0, "C2N2": 52.0, "C4N2": 76.0, "CN": 26.0,
        "O": 16.0, "O(1D)": 16.0, "O2": 32.0, "OH": 17.0,
        "H2O": 18.0, "CO": 28.0, "CO2": 44.0,
        "HCO": 29.0, "H2CO": 30.0, "CH2OH": 31.0, "CH3O": 31.0, "CH3OH": 32.0,
        "HCCO": 41.0, "H2CCO": 42.0, "CH3CO": 43.0, "CH3CHO": 44.0,
        "C2H4OH": 45.0,
        "M": 28.0, "JDUST": 0.0, "PROD": 0.0, "U": 0.0, "V": 0.0,
        "H2OCON": 18.0, "C4H2CON": 50.0, "C4H10CON": 58.0,
    }
    if name in fallback:
        return fallback[name]
    # Last-resort chemical-formula parser.
    import re
    element_amu = {"H": 1.0, "He": 4.0, "C": 12.0, "N": 14.0, "O": 16.0, "S": 32.0}
    stripped = re.sub(r"\([^)]*\)", "", name)
    stripped = re.sub(r"^\d+[,\d]*-", "", stripped)
    stripped = stripped.rstrip("*")
    total = 0.0
    for sym, count in re.findall(r"([A-Z][a-z]?)(\d*)", stripped):
        if sym in element_amu:
            total += element_amu[sym] * (int(count) if count else 1)
    return total

