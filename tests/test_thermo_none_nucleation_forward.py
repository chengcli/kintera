"""ThermoX/ThermoY forward with nucleation set to None after reset() must not crash.

The species have a vapor and a cloud and one nucleation reaction, so forward
reaches the saturation solver before the mutation. After construction the
nucleation is set to None, on the ThermoOptions passed in or through
th.options, and forward is called. It either refuses None with a Python
exception or gives the result of an empty NucleationOptions set the same way.
Each case runs in a fresh subprocess, because a segfault would kill pytest.
"""

import os
import subprocess
import sys

import pytest
import torch

SCRIPT = """
import sys, torch, kintera
from kintera import NucleationOptions, Reaction, ThermoOptions, ThermoX, ThermoY
torch.set_default_dtype(torch.float64)
case, where, nucleation, device = sys.argv[1:4] + [torch.device(sys.argv[4])]
kintera.set_species_names(["dry", "H2O", "H2O(l)"])
kintera.set_species_weights([29.e-3, 18.e-3, 18.e-3])
nuc = NucleationOptions()
nuc.reactions([Reaction("H2O <=> H2O(l)")])
nuc.minT([200.0])
nuc.maxT([400.0])
nuc.logsvp(["h2o_ideal"])
op = ThermoOptions().max_iter(15).ftol(1.e-8)
op.vapor_ids([0, 1])
op.cloud_ids([2])
op.cref_R([2.5, 2.5, 9.0])
op.uref_R([0.0, 0.0, -3430.])
op.sref_R([0.0, 0.0, 0.0])
op.Tref(300.0)
op.Pref(1.e5)
op.nucleation(nuc)
th = (ThermoX if case == "x" else ThermoY)(op)
th.to(device)
(op if where == "op" else th.options).nucleation(
    None if nucleation == "none" else NucleationOptions())
temp = torch.full((2, 3), 300.0, device=device)
if case == "x":
    pres = torch.full((2, 3), 1.e5, device=device)
    xfrac = torch.tensor([0.9, 0.08, 0.02], device=device).expand(2, 3, 3)
    xfrac = xfrac.contiguous()
    th.forward(temp, pres, xfrac)
    out = xfrac
else:
    conc = torch.tensor([30.0, 10.0, 1.0], device=device).expand(2, 3, 3)
    conc = conc.contiguous()
    mu = torch.tensor([29.e-3, 18.e-3, 18.e-3], device=device)
    yfrac = th.compute("V->Y", [conc])
    th.forward(conc @ mu, th.compute("VT->U", [conc, temp]), yfrac)
    out = yfrac
print(repr(out.flatten().tolist()))
"""

CASES = [("x", "op"), ("x", "options"), ("y", "op"), ("y", "options")]


def _run(case, where, nucleation, device):
    return subprocess.run(
        [sys.executable, "-c", SCRIPT, case, where, nucleation, device],
        capture_output=True, text=True, env=os.environ.copy())


def _none_nucleation(case, where, device):
    ref = _run(case, where, "empty", device)
    assert ref.returncode == 0, ref.stderr
    out = _run(case, where, "none", device)
    if out.returncode == 1 and "Traceback" in out.stderr:
        return  # refused with a Python exception: acceptable
    assert out.returncode == 0, (
        f"Thermo{case.upper()} forward with {where}.nucleation(None) on "
        f"{device}: returncode {out.returncode} (-11 = SIGSEGV)\n"
        f"stderr:\n{out.stderr}")
    torch.testing.assert_close(
        torch.tensor(eval(out.stdout.strip().splitlines()[-1])),
        torch.tensor(eval(ref.stdout.strip().splitlines()[-1])),
        rtol=1e-12, atol=0.)


@pytest.mark.parametrize("case,where", CASES)
def test_none_nucleation_forward(case, where):
    _none_nucleation(case, where, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("case,where", CASES)
def test_none_nucleation_forward_cuda(case, where):
    _none_nucleation(case, where, "cuda:0")
