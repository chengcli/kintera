"""ThermoX/ThermoY built from a default ThermoOptions() must not crash.

The construction follows examples/example_earth_dry.py: a Python ThermoOptions()
with no .nucleation(...), then ThermoX(op) or ThermoY(op). Each construction runs
in a fresh subprocess, because a segfault would kill pytest.
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
kind, device = sys.argv[1], sys.argv[2]
kintera.set_species_names(["dry"])
kintera.set_species_weights([29.e-3])
op = ThermoOptions()
op.vapor_ids([0])
op.cref_R([2.5])
op.uref_R([0.0])
op.sref_R([0.0])
op.Tref(300.0)
op.Pref(1.e5)
th = (ThermoX if kind == "X" else ThermoY)(op)
th.to(torch.device(device))
mu = th.mu[0] if kind == "X" else 1. / th.inv_mu[0]
print(mu.device.type, repr(mu.item()))
"""


def _default_options(kind, device):
    out = subprocess.run([sys.executable, "-c", SCRIPT, kind, device],
                         capture_output=True, text=True, env=os.environ.copy())
    assert out.returncode == 0, (
        f"Thermo{kind}(ThermoOptions()) on {device}: returncode "
        f"{out.returncode} (-11 = SIGSEGV)\nstderr:\n{out.stderr}")
    dev, mu = out.stdout.strip().splitlines()[-1].split()
    assert dev == torch.device(device).type
    torch.testing.assert_close(torch.tensor(float(mu)), torch.tensor(29.e-3),
                               rtol=1e-12, atol=0.)


def test_thermo_x_default_options():
    _default_options("X", "cpu")


def test_thermo_y_default_options():
    _default_options("Y", "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_thermo_x_default_options_cuda():
    _default_options("X", "cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_thermo_y_default_options_cuda():
    _default_options("Y", "cuda:0")



SATURATE = """  # K3 (#54): one nucleation reaction condenses a supersaturated cell
import sys, torch, kintera
from kintera import NucleationOptions, Reaction, ThermoOptions, ThermoX, ThermoY
torch.set_default_dtype(torch.float64)
kind, dev = sys.argv[1], torch.device(sys.argv[2])
kintera.set_species_names(["dry", "H2O", "H2O(l)"])
kintera.set_species_weights([29.e-3, 18.e-3, 18.e-3])
op = ThermoOptions().vapor_ids([0, 1]).cloud_ids([2]).cref_R([2.5, 3., 9.])
op.uref_R([0., 0., -6786.66]).sref_R([0.] * 3).Tref(300.).Pref(1.e5)
op.nucleation(NucleationOptions().reactions([Reaction("H2O => H2O(l)")]).logsvp(["h2o_ideal"]))
th, T, one = (ThermoX if kind == "X" else ThermoY)(op), torch.tensor([300.]), torch.ones(1)
th.to(dev); T, one = T.to(dev), one.to(dev)
if kind == "X":
    y = torch.tensor([[.9, .1, 0.]], device=dev)
    th.forward(T, 1.e5 * one, y)
    conc = th.compute("TPX->V", [T, 1.e5 * one, y])
else:
    y = torch.tensor([[.05], [0.]], device=dev)
    U = th.compute("VT->U", [th.compute("DY->V", [one, y]), T])
    th.forward(one, U, y)
    conc = th.compute("DY->V", [one, y]) * th.buffer("inv_mu")
    T = th.compute("VU->T", [conc / th.buffer("inv_mu"), U])
rh = kintera.relative_humidity(T, conc, th.buffer("stoich"), op.nucleation())
print(y.flatten()[-1].item(), rh.flatten()[0].item())
"""


@pytest.mark.parametrize("kind", ["X", "Y"])
@pytest.mark.parametrize("dev", ["cpu"] + ["cuda"] * torch.cuda.is_available())
def test_nucleation_reaction_saturates(kind, dev):
    out = subprocess.run([sys.executable, "-c", SATURATE, kind, dev], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    cloud, rh = map(float, out.stdout.split()[-2:])
    assert cloud > 0 and abs(rh - 1.) < 1e-6, out.stdout
