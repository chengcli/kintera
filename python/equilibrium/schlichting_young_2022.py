"""Schlichting & Young (2022) core-mantle-atmosphere equilibrium case.

The paper specifies the reaction topology and pressure relation completely,
but refers several standard-state Gibbs energies to external NIST, MAGMA, and
experimental datasets. Consequently this module requires a ``log_k_model``
callable rather than silently substituting thermodynamic data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import torch

from ..kintera import EquilibriumOptions, EquilibriumTP, molar_masses_from_yaml

_CONFIG = Path(__file__).with_name("schlichting_young_2022.yaml")
_BASE_OPTIONS = EquilibriumOptions.from_yaml(str(_CONFIG))
COMPONENTS = tuple(_BASE_OPTIONS.components())
PHASES = tuple(_BASE_OPTIONS.phases())
REACTIONS = tuple(_BASE_OPTIONS.reactions())
_PHASE_IDS = tuple(_BASE_OPTIONS.phase_ids())
_GAS_PHASE = _BASE_OPTIONS.gas_phase()
_COMPONENT_INDEX = {name: i for i, name in enumerate(COMPONENTS)}

_MOLAR_MASS = tuple(molar_masses_from_yaml(str(_CONFIG)))


def make_options(
    scenario: Literal["reactive", "isolated-core"] = "reactive",
    *,
    max_iter: int = 100,
    ftol: float = 5e-3,
) -> EquilibriumOptions:
    """Create the generic core options for the paper's reaction system."""
    options = EquilibriumOptions.from_yaml(str(_CONFIG))
    if scenario == "isolated-core":
        keep = [i for i in range(18) if i not in (1, 3, 4, 6)]
        options.reactions([options.reactions()[j] for j in keep])
    elif scenario != "reactive":
        raise ValueError(f"unknown scenario: {scenario}")
    options.max_iter(max_iter).ftol(ftol)
    options.validate()
    return options


def initial_moles(
    hydrogen_mass_fraction: float = 0.04,
    metal_mass_fraction: float = 0.25,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Construct the paper's nominal composition on a one-kilogram basis."""
    if not 0 < hydrogen_mass_fraction < 1:
        raise ValueError("hydrogen_mass_fraction must lie between zero and one")
    if not 0 <= metal_mass_fraction < 1 - hydrogen_mass_fraction:
        raise ValueError("metal_mass_fraction leaves no silicate inventory")

    mass = torch.tensor(_MOLAR_MASS, dtype=dtype, device=device)
    result = torch.full((len(COMPONENTS),), 1.0e-20, dtype=dtype, device=device)
    fe_metal = _COMPONENT_INDEX["Fe[metal]"]
    result[fe_metal] = metal_mass_fraction / mass[fe_metal]

    silicate_mass = 1.0 - metal_mass_fraction - hydrogen_mass_fraction
    mgsio3 = _COMPONENT_INDEX["MgSiO3[silicate]"]
    mgo = _COMPONENT_INDEX["MgO[silicate]"]
    fesio3 = _COMPONENT_INDEX["FeSiO3[silicate]"]
    na2o = _COMPONENT_INDEX["Na2O[silicate]"]
    silicate_moles = silicate_mass / (
        0.921 * mass[mgsio3]
        + 0.032 * mass[mgo]
        + 0.035 * mass[fesio3]
        + 0.007 * mass[na2o]
    )
    result[mgsio3] = 0.921 * silicate_moles
    result[mgo] = 0.032 * silicate_moles
    result[fesio3] = 0.035 * silicate_moles
    result[na2o] = 0.007 * silicate_moles

    # The primary atmosphere is 99.9 mol% H2 and 0.1 mol% CO.
    h2_gas = _COMPONENT_INDEX["H2[gas]"]
    co_gas = _COMPONENT_INDEX["CO[gas]"]
    gas_moles = hydrogen_mass_fraction / (
        0.999 * mass[h2_gas] + 0.001 * mass[co_gas]
    )
    result[h2_gas] = 0.999 * gas_moles
    result[co_gas] = 0.001 * gas_moles
    return result


def surface_pressure(
    moles: torch.Tensor, planet_mass_earth: float | torch.Tensor = 4.0
) -> torch.Tensor:
    """Evaluate Equation 8 using the atmosphere mass implied by ``moles``."""
    mass = torch.as_tensor(_MOLAR_MASS, dtype=moles.dtype, device=moles.device)
    component_mass = moles * mass
    total_mass = component_mass.sum(dim=-1)
    gas = torch.tensor(
        [i for i, phase in enumerate(_PHASE_IDS) if phase == _GAS_PHASE],
        dtype=torch.long,
        device=moles.device,
    )
    atmosphere_fraction = (
        component_mass.index_select(-1, gas).sum(dim=-1) / total_mass
    )
    planet_mass = torch.as_tensor(
        planet_mass_earth, dtype=moles.dtype, device=moles.device
    )
    return 1.1e11 * atmosphere_fraction * planet_mass.pow(2.0 / 3.0)


@dataclass(frozen=True)
class SchlichtingYoungResult:
    moles: torch.Tensor
    pressure: torch.Tensor
    phase_totals: torch.Tensor
    phase_fractions: torch.Tensor
    gain: torch.Tensor
    diagnostics: torch.Tensor
    pressure_error: torch.Tensor
    outer_iterations: int


class SchlichtingYoung2022:
    """Python driver coupling the generic core to the paper's pressure law."""

    def __init__(
        self,
        log_k_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scenario: Literal["reactive", "isolated-core"] = "reactive",
        *,
        max_outer_iter: int = 30,
        pressure_tolerance: float = 0.1,
        pressure_damping: float = 0.5,
    ) -> None:
        if log_k_model is None:
            raise TypeError(
                "log_k_model is required because the paper refers some Gibbs "
                "energies to external datasets"
            )
        if max_outer_iter <= 0:
            raise ValueError("max_outer_iter must be positive")
        if pressure_tolerance <= 0:
            raise ValueError("pressure_tolerance must be positive")
        if not 0 < pressure_damping <= 1:
            raise ValueError("pressure_damping must lie in (0, 1]")
        self.scenario = scenario
        self.log_k_model = log_k_model
        self.max_outer_iter = max_outer_iter
        self.pressure_tolerance = pressure_tolerance
        self.pressure_damping = pressure_damping
        self.options = make_options(scenario)
        self.solver = EquilibriumTP(self.options)

    def solve(
        self,
        temp: torch.Tensor,
        moles: torch.Tensor,
        pressure: torch.Tensor | None = None,
        *,
        planet_mass_earth: float = 4.0,
    ) -> SchlichtingYoungResult:
        self.solver.to(moles.device, moles.dtype)
        pressure = (
            surface_pressure(moles, planet_mass_earth)
            if pressure is None
            else pressure.to(device=moles.device, dtype=moles.dtype)
        )
        state = moles
        gain = diagnostics = None
        pressure_error = torch.full_like(pressure, float("inf"))

        for _iteration in range(1, self.max_outer_iter + 1):
            log_k = self.log_k_model(temp, pressure)
            if self.scenario == "isolated-core" and log_k.size(-1) == 18:
                keep = [i for i in range(18) if i not in (1, 3, 4, 6)]
                log_k = log_k[..., keep]
            state, gain, diagnostics = self.solver(temp, pressure, state, log_k)
            target = surface_pressure(state, planet_mass_earth)
            pressure_error = (target - pressure).abs() / target.clamp_min(
                torch.finfo(target.dtype).tiny
            )
            chemistry_ok = diagnostics[..., 0].eq(0).all()
            if chemistry_ok and pressure_error.max() <= self.pressure_tolerance:
                pressure = target
                break
            pressure = torch.exp(
                (1.0 - self.pressure_damping) * torch.log(pressure)
                + self.pressure_damping * torch.log(target)
            )

        phase_totals = torch.stack(
            [
                state[..., [i for i, p in enumerate(_PHASE_IDS) if p == phase]].sum(
                    -1
                )
                for phase in range(len(PHASES))
            ],
            dim=-1,
        )
        phase_fractions = torch.empty_like(state)
        for phase in range(len(PHASES)):
            ids = [i for i, value in enumerate(_PHASE_IDS) if value == phase]
            phase_fractions[..., ids] = (
                state[..., ids] / phase_totals[..., phase, None]
            )

        return SchlichtingYoungResult(
            state,
            pressure,
            phase_totals,
            phase_fractions,
            gain,
            diagnostics,
            pressure_error,
            _iteration,
        )
