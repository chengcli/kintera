"""A null NucleationOptions that bypasses ThermoX/ThermoY reset() must not crash.

reset() replaces a null nucleation with the empty default, but a caller can
still set it to None after construction (on the ThermoOptions they passed in,
or through th.options), or pass None straight to kintera.relative_humidity.
Each path either refuses None with a Python exception or gives the result of
the empty default. Each case runs in a fresh subprocess, because a segfault
would kill pytest.
"""

import os
import subprocess
import sys

import pytest
import torch

SCRIPT = """
import sys, torch, kintera
from kintera import ThermoOptions, ThermoX, ThermoY
torch.set_default_dtype(torch.float64)
case, nucleation, device = sys.argv[1], sys.argv[2], torch.device(sys.argv[3])
kintera.set_species_names(["dry"])
kintera.set_species_weights([29.e-3])
op = ThermoOptions()
op.vapor_ids([0])
op.cref_R([2.5])
op.uref_R([0.0])
op.sref_R([0.0])
op.Tref(300.0)
op.Pref(1.e5)
none = nucleation == "none"
temp = torch.full((2, 3), 300.0, device=device)
pres = torch.full((2, 3), 1.e5, device=device)
if case in ("x_op", "x_options"):
    th = ThermoX(op)
    th.to(device)
    if none:
        (op if case == "x_op" else th.options).nucleation(None)
    xfrac = torch.ones((2, 3, 1), device=device)
    conc = th.compute("TPX->V", [temp, pres, xfrac])
    out = th.compute("TPV->S", [temp, pres, conc])
elif case == "y_op":
    th = ThermoY(op)
    th.to(device)
    if none:
        op.nucleation(None)
    conc = torch.full((2, 3, 1), 40.0, device=device)
    out = th.compute("PVT->S", [pres, conc, temp])
elif case == "relative_humidity":
    conc = torch.full((2, 3, 1), 40.0, device=device)
    stoich = torch.zeros((1, 0), device=device)
    out = kintera.relative_humidity(temp, conc, stoich,
                                    None if none else op.nucleation())
print(repr(out.flatten().tolist()))
"""

CASES = ["x_op", "x_options", "y_op", "relative_humidity"]


def _run(case, nucleation, device):
    return subprocess.run(
        [sys.executable, "-c", SCRIPT, case, nucleation, device],
        capture_output=True, text=True, env=os.environ.copy())


def _none_nucleation(case, device):
    ref = _run(case, "default", device)
    assert ref.returncode == 0, ref.stderr
    out = _run(case, "none", device)
    if out.returncode == 1 and "Traceback" in out.stderr:
        return  # refused with a Python exception: acceptable
    assert out.returncode == 0, (
        f"{case} with nucleation None on {device}: returncode "
        f"{out.returncode} (-11 = SIGSEGV)\nstderr:\n{out.stderr}")
    torch.testing.assert_close(
        torch.tensor(eval(out.stdout.strip().splitlines()[-1])),
        torch.tensor(eval(ref.stdout.strip().splitlines()[-1])),
        rtol=1e-12, atol=0.)


@pytest.mark.parametrize("case", CASES)
def test_none_nucleation_after_reset(case):
    _none_nucleation(case, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("case", CASES)
def test_none_nucleation_after_reset_cuda(case):
    _none_nucleation(case, "cuda:0")
