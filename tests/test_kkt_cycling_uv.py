"""K19 (#54): a tests/jupiter.yaml UV cell (104.19 K, 1.95e6 Pa; NH3 gas absent, NH3(s) present)
whose first saturation step cycled in the feasible-origin active set and returned status 2.
Reference: the only admissible cloud assemblage (H2O(l) and NH4SH(s) present, NH3(s) gone),
found by enumerating all eight and solving saturation, H2O/N/S and energy conservation."""
import os
import pytest
import torch
from kintera import ThermoOptions, ThermoY

torch.set_default_dtype(torch.float64)

# mole fractions: dry, H2O, NH3, H2S, H2O(l), H2O(l,p), NH3(s), NH3(s,p), NH4SH(s)
X = [0.9452025167266452, 3.0793244963643104e-06, 0.0, 0.04432008639206904, 0.0,
     0.0033088825716865882, 0.007165383153972112, 0.0, 5.1831130667826045e-08]
T0, P0 = 104.18617226219762, 1946753.4681013762


@pytest.mark.parametrize("dev", ["cpu"] + ["cuda"] * torch.cuda.is_available())
def test_uv_cell_converges_to_the_equilibrium(dev):
    op = ThermoOptions.from_yaml(os.path.join(os.path.dirname(__file__), "jupiter.yaml"))
    thermo = ThermoY(op.max_iter(100).uv_solver("kkt"))
    thermo.to(torch.device(dev))
    inv_mu = thermo.buffer("inv_mu")
    c = torch.tensor(X, device=dev) * P0 / (8.31446 * T0) / sum(X[:4])  # mol/m^3
    rho = (c / inv_mu).sum().reshape(1)
    y = (c[1:] / inv_mu[1:] / rho).reshape(-1, 1).contiguous()
    V = thermo.compute("DY->V", [rho, y])
    U = thermo.compute("VT->U", [V, torch.tensor([T0], device=dev)])
    diag = torch.zeros(1, 1, device=dev)
    thermo.forward(rho, U, y, False, diag)
    assert diag.item() >= 0, diag.item()

    V = thermo.compute("DY->V", [rho, y])
    T = thermo.compute("VU->T", [V, U])
    c = (V * inv_mu)[0].cpu()
    assert abs(T.item() - 129.7948288705) < 1e-6
    assert abs(c[1].item() / 7.9060992401e-12 - 1) < 1e-6  # H2O vapour at saturation
    assert abs(c[4].item() - 6.9935101902e-03) < 1e-12
    assert c[6].item() < 1e-12  # NH3(s) evaporated
    assert abs(c[8].item() - 1.6273550482e+01) < 1e-8
