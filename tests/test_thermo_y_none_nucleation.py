"""ThermoY with an explicit .nucleation(None) must not crash in PVT->S.

The ThermoY counterpart of test_thermo_none_nucleation.py (#124). Either None
is refused with a Python exception, or PVT->S gives the entropy of the empty
default. Each case runs in a fresh subprocess, because a segfault would kill
pytest.
"""

import os
import subprocess
import sys

import pytest
import torch

SCRIPT = """
import sys, torch, kintera
from kintera import ThermoOptions, ThermoY
torch.set_default_dtype(torch.float64)
nucleation, device = sys.argv[1], torch.device(sys.argv[2])
kintera.set_species_names(["dry"])
kintera.set_species_weights([29.e-3])
op = ThermoOptions()
op.vapor_ids([0])
op.cref_R([2.5])
op.uref_R([0.0])
op.sref_R([0.0])
op.Tref(300.0)
op.Pref(1.e5)
if nucleation == "none":
    op.nucleation(None)
th = ThermoY(op)
th.to(device)
temp = torch.full((2, 3), 300.0, device=device)
pres = torch.full((2, 3), 1.e5, device=device)
conc = torch.full((2, 3, 1), 40.0, device=device)
entropy = th.compute("PVT->S", [pres, conc, temp])
print(repr(entropy.flatten()[0].item()))
"""


def _entropy(nucleation, device):
    return subprocess.run([sys.executable, "-c", SCRIPT, nucleation, device],
                          capture_output=True, text=True, env=os.environ.copy())


def _none_nucleation(device):
    ref = _entropy("default", device)
    assert ref.returncode == 0, ref.stderr
    out = _entropy("none", device)
    if out.returncode == 1 and "Traceback" in out.stderr:
        return  # refused with a Python exception: acceptable
    assert out.returncode == 0, (
        f"PVT->S with nucleation(None) on {device}: returncode "
        f"{out.returncode} (-11 = SIGSEGV)\nstderr:\n{out.stderr}")
    torch.testing.assert_close(
        torch.tensor(float(out.stdout.strip().splitlines()[-1])),
        torch.tensor(float(ref.stdout.strip().splitlines()[-1])),
        rtol=1e-12, atol=0.)


def test_thermo_y_none_nucleation_entropy():
    _none_nucleation("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_thermo_y_none_nucleation_entropy_cuda():
    _none_nucleation("cuda:0")
