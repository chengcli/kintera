"""CUDA saturation adjustment whose per-thread workspace overflows a 32-thread block.

Eight vapour/cloud pairs need about 10 KB of shared workspace per cell, 325 KB for
a 32-thread block: more than any GPU's per-block limit, while one cell still fits.
The kernel launcher must shrink the block instead of refusing, and the CUDA result
must match the CPU one.
"""

import pytest
import torch
from kintera import ThermoOptions, ThermoX, ThermoY

torch.set_default_dtype(torch.float64)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NPAIR = 8
SPECIES = "\n".join(
    [f"  - {{name: dry, composition: {{N: 2}}, cv_R: 2.5}}"]
    + [f"  - {{name: V{i}, composition: {{H: 2, O: 1}}, cv_R: 2.5}}" for i in range(NPAIR)]
    + [f"  - {{name: V{i}(l), composition: {{H: 2, O: 1}}, cv_R: 9.0, u0_R: -3430.}}"
       for i in range(NPAIR)])
REACTIONS = "\n".join(
    f"  - {{equation: V{i} <=> V{i}(l), type: nucleation, rate-constant: "
    f"{{formula: antoine, A: 5.40221, B: {1700. + 40. * i}, C: -31.737}}}}"
    for i in range(NPAIR))
CARD = f"""reference-state: {{Tref: 0., Pref: 1.e5}}
dynamics: {{equation-of-state: {{max-iter: 50, ftol: 1.e-12}}}}
species:
{SPECIES}
reactions:
{REACTIONS}
"""
TEMP = torch.linspace(250., 400., 200)


def _options(tmp_path):
    (path := tmp_path / "pairs.yaml").write_text(CARD)
    return ThermoOptions.from_yaml(str(path))


def _tp(op, device):
    thermo = ThermoX(op)
    thermo.to(torch.device(device))
    xfrac = torch.zeros(TEMP.numel(), 1 + 2 * NPAIR, device=device)
    xfrac[:, 1:1 + NPAIR] = 0.02
    xfrac[:, 0] = 1. - xfrac.sum(-1)
    temp = TEMP.to(device)
    thermo.forward(temp, torch.full_like(temp, 1.e5), xfrac)
    return xfrac.cpu()


def _uv(op, device):
    thermo = ThermoY(op)
    thermo.to(torch.device(device))
    yfrac = torch.zeros(2 * NPAIR, TEMP.numel(), device=device)
    yfrac[:NPAIR] = 0.02
    rho = torch.ones(TEMP.numel(), device=device)
    ivol = thermo.compute("DY->V", [rho, yfrac])
    thermo.forward(rho, thermo.compute("VT->U", [ivol, TEMP.to(device)]), yfrac)
    return yfrac.cpu()


def test_equilibrate_tp_cuda_matches_cpu(tmp_path):
    op = _options(tmp_path)
    cpu, gpu = _tp(op, "cpu"), _tp(op, "cuda")
    cloudy = cpu[:, 1 + NPAIR:].gt(0)
    assert 0 < cloudy.sum() < cloudy.numel()
    torch.testing.assert_close(gpu, cpu, rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("solver", ["partition", "kkt"])
def test_equilibrate_uv_cuda_matches_cpu(tmp_path, solver):
    op = _options(tmp_path).uv_solver(solver)
    cpu, gpu = _uv(op, "cpu"), _uv(op, "cuda")
    cloudy = cpu[NPAIR:].gt(0)
    assert 0 < cloudy.sum() < cloudy.numel()
    torch.testing.assert_close(gpu, cpu, rtol=1e-10, atol=1e-14)
