from pathlib import Path

import pytest
import torch
from kintera import (
    EquilibriumOptions,
    EquilibriumTP,
    atomic_mass,
    molar_mass,
    molar_masses_from_yaml,
)
from kintera.equilibrium import (
    COMPONENTS,
    SchlichtingYoung2022,
    initial_moles,
    make_options,
    surface_pressure,
)


def test_python_core_binding_is_functional():
    options = (
        EquilibriumOptions()
        .components(["A", "B"])
        .phases(["gas"])
        .phase_ids([0, 0])
        .reactions(["A <=> B"])
        .gas_phase(0)
    )
    solver = EquilibriumTP(options)
    moles = torch.tensor([0.8, 0.2], dtype=torch.float64)
    result, gain, diagnostics = solver(
        torch.tensor(1000.0, dtype=torch.float64),
        torch.tensor(1.0e5, dtype=torch.float64),
        moles,
        torch.zeros(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result, torch.tensor([0.5, 0.5], dtype=torch.float64), atol=1e-7, rtol=1e-7
    )
    torch.testing.assert_close(moles, torch.tensor([0.8, 0.2], dtype=torch.float64))
    assert gain.shape == (1, 1)
    assert diagnostics[0] == 0


def test_standalone_molar_mass_utilities():
    assert atomic_mass("H") == pytest.approx(1.008e-3)
    assert molar_mass({"H": 2.0, "O": 1.0}) == pytest.approx(18.015e-3)


def test_paper_topology_and_pressure_relation():
    options = make_options()
    options.validate()
    assert len(options.components()) == 25
    assert len(options.reactions()) == 18
    config = (
        Path(__file__).parents[1]
        / "python"
        / "equilibrium"
        / "schlichting_young_2022.yaml"
    )
    assert len(molar_masses_from_yaml(str(config))) == 25
    assert EquilibriumTP(options).buffer("stoich").shape == (25, 18)
    isolated = make_options("isolated-core")
    assert EquilibriumTP(isolated).buffer("stoich").shape == (25, 14)

    moles = initial_moles()
    pressure = surface_pressure(moles)
    assert moles.shape == (len(COMPONENTS),)
    assert pressure > 0


def test_paper_driver_accepts_external_thermodynamics():
    moles = initial_moles()
    options = make_options()
    stoich = EquilibriumTP(options).buffer("stoich").to(dtype=moles.dtype)
    phase_ids = options.phase_ids()

    def equilibrium_at_initial_state(temp, pres):
        del temp
        log_activity = torch.empty_like(moles)
        for phase in range(3):
            ids = [i for i, value in enumerate(phase_ids) if value == phase]
            log_activity[ids] = torch.log(moles[ids] / moles[ids].sum())
        gas = [i for i, value in enumerate(phase_ids) if value == 2]
        log_activity[gas] += torch.log(pres / 1.0e5)
        return log_activity @ stoich

    model = SchlichtingYoung2022(equilibrium_at_initial_state, max_outer_iter=2)
    result = model.solve(torch.tensor(4500.0, dtype=moles.dtype), moles)
    assert model.options.gas_phase() == 2
    assert result.diagnostics[0] == 0
    assert result.pressure_error < 0.1
    for phase in range(3):
        ids = [i for i, value in enumerate(model.options.phase_ids()) if value == phase]
        torch.testing.assert_close(
            result.phase_fractions[ids].sum(), torch.tensor(1.0, dtype=moles.dtype)
        )


def test_paper_driver_rejects_invalid_outer_solve_options():
    def model(temp, pres):
        del pres
        return torch.zeros(18, dtype=temp.dtype, device=temp.device)

    with pytest.raises(ValueError, match="max_outer_iter"):
        SchlichtingYoung2022(model, max_outer_iter=0)
    with pytest.raises(ValueError, match="pressure_damping"):
        SchlichtingYoung2022(model, pressure_damping=0.0)
