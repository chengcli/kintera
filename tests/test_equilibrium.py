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


def test_ill_conditioned_multiphase_newton_step():
    """Regression for normal-equation precision loss in a Young22 state."""
    components = [
        "MgO_silicate", "SiO2_silicate", "MgSiO3_silicate",
        "FeO_silicate", "FeSiO3_silicate", "Na2O_silicate",
        "Na2SiO3_silicate", "H2_silicate", "H2O_silicate",
        "CO_silicate", "CO2_silicate", "Fe_metal", "Si_metal",
        "O_metal", "H_metal", "H2_gas", "CO_gas", "CO2_gas",
        "CH4_gas", "O2_gas", "H2O_gas", "Fe_gas", "Mg_gas",
        "SiO_gas", "Na_gas",
    ]
    reactions = [
        "Na2SiO3_silicate <=> Na2O_silicate + SiO2_silicate",
        "0.5 SiO2_silicate + Fe_metal <=> FeO_silicate + 0.5 Si_metal",
        "MgSiO3_silicate <=> MgO_silicate + SiO2_silicate",
        "O_metal + 0.5 Si_metal <=> 0.5 SiO2_silicate",
        "2 H_metal <=> H2_silicate",
        "FeSiO3_silicate <=> FeO_silicate + SiO2_silicate",
        "2 H2O_silicate + Si_metal <=> SiO2_silicate + 2 H2_silicate",
        "CO_gas + 0.5 O2_gas <=> CO2_gas",
        "CH4_gas + 0.5 O2_gas <=> 2 H2_gas + CO_gas",
        "H2_gas + 0.5 O2_gas <=> H2O_gas",
        "FeO_silicate <=> Fe_gas + 0.5 O2_gas",
        "MgO_silicate <=> Mg_gas + 0.5 O2_gas",
        "SiO2_silicate <=> SiO_gas + 0.5 O2_gas",
        "Na2O_silicate <=> 2 Na_gas + 0.5 O2_gas",
        "H2_gas <=> H2_silicate",
        "H2O_gas <=> H2O_silicate",
        "CO_gas <=> CO_silicate",
        "CO2_gas <=> CO2_silicate",
    ]
    options = (
        EquilibriumOptions()
        .components(components)
        .phases(["silicate", "metal", "gas"])
        .phase_ids([0] * 11 + [1] * 4 + [2] * 10)
        .reactions(reactions)
        .gas_phase(2)
        .standard_pressure(1.0e5)
        .max_iter(1000)
        .ftol(1.0e-8)
        .mole_floor(1.0e-40)
    )
    initial = torch.tensor(
        [
            229.45501199233973, 7.170469124760617, 6604.002063904528,
            3.5852345623803084, 250.96641936662164, 50.19328387332432,
            7.170469124760617, 7.1704691247606165e-9,
            7.1704691247606165e-9, 7.1704691247606165e-9,
            7.1704691247606165e-9, 4476.516196031982,
            4.476516196031981e-9, 4.476516196031981e-9,
            4.476516196031981e-9, 19570.252531955153, 19.589842374329482,
            1.9589842374329483e-8, 1.9589842374329483e-8,
            1.9589842374329483e-8, 1.9589842374329483e-8,
            1.9589842374329483e-8, 1.9589842374329483e-8,
            1.9589842374329483e-8, 1.9589842374329483e-8,
        ],
        dtype=torch.float64,
    )
    log_k = torch.tensor(
        [
            -4.034640768501788, -7.62490310519855, -2.039313234293211,
            7.218657926794565, 0.7959817637548838, -0.1371317344272009,
            26.01075006664782, -4.50470154179624, 38.86385560946831,
            -3.708764843086439, 4.050968036542196, 1.772570237146461,
            5.579140721827175, 14.1841721338006, -13.701917787416148,
            -14.27932134370162, -16.21258232342021, -14.87214162753718,
        ],
        dtype=torch.float64,
    )
    result, _, diagnostics = EquilibriumTP(options)(
        torch.tensor(4500.0, dtype=torch.float64),
        torch.tensor(12599210499.047432, dtype=torch.float64),
        initial,
        log_k,
    )

    assert diagnostics[0] == 0
    assert diagnostics[2] < 1.0e-8
    assert torch.isfinite(result).all()
    assert torch.all(result > 0.0)
