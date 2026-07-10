from pathlib import Path

import pytest
import torch
from kintera import (
    EquilibriumOptions,
    EquilibriumTP,
    atomic_mass,
    molar_mass,
)
from kintera.equilibrium import Nasa9LogK


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


def test_nasa9_yaml_equilibrium_case():
    config = Path(__file__).parents[1] / "examples" / "equilibrium_nasa9.yaml"
    options = EquilibriumOptions.from_yaml(str(config))
    solver = EquilibriumTP(options)
    model = Nasa9LogK(options)
    temp = torch.tensor(4000.0, dtype=torch.float64)
    pressure = torch.tensor(1.0e5, dtype=torch.float64)
    initial = torch.tensor([0.6, 0.2, 0.2], dtype=torch.float64)

    log_k = model(temp, pressure)
    result, _, diagnostics = solver(temp, pressure, initial, log_k)

    assert log_k.shape == (1,)
    assert torch.isfinite(log_k).all()
    assert diagnostics[0] == 0
    assert diagnostics[2] < 1.0e-7
    torch.testing.assert_close(
        result @ torch.tensor([2.0, 0.0, 2.0], dtype=result.dtype),
        initial @ torch.tensor([2.0, 0.0, 2.0], dtype=initial.dtype),
    )
    torch.testing.assert_close(
        result @ torch.tensor([0.0, 2.0, 1.0], dtype=result.dtype),
        initial @ torch.tensor([0.0, 2.0, 1.0], dtype=initial.dtype),
    )
