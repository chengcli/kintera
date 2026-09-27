"""Kinetics must not crash on KineticsOptions whose sub-options are null.

KineticsOptions() leaves arrhenius, coagulation, evaporation, ... null, a
caller can set one to None, and KineticsOptions.from_yaml returns None for a
card without reference-state. The Kinetics constructor dereferences them all.
Each path either refuses with a Python exception or builds a Kinetics that
runs; with evaporation(None) on a card without evaporation reactions the
rate must equal the default's. Each case runs in a fresh subprocess, because
a segfault would kill pytest.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

CARD = """
reference-state: {Tref: 300., Pref: 1.e5}
species:
  - {name: O2, composition: {O: 2}, cv_R: 2.5}
  - {name: O, composition: {O: 1}, cv_R: 1.5}
  - {name: O3, composition: {O: 3}, cv_R: 3.0}
reactions:
  - {equation: O + O2 <=> O3, type: arrhenius,
     rate-constant: {A: 1.7e-14, b: -2.4, Ea_R: 0.}}
"""

SCRIPT = """
import json, sys, torch
from kintera import Kinetics, KineticsOptions
torch.set_default_dtype(torch.float64)
case, card, device = sys.argv[1], sys.argv[2], torch.device(sys.argv[3])
if case == "plain":
    kin = Kinetics(KineticsOptions())
elif case == "none":  # the card has no reference-state: from_yaml gives None
    kin = Kinetics(KineticsOptions.from_yaml(card))
else:
    op = KineticsOptions.from_yaml(card)
    if case == "evaporation_none":
        op.evaporation(None)
    kin = Kinetics(op)
kin.to(device)
if case in ("default", "evaporation_none"):
    temp = torch.tensor([250.], device=device)
    pres = torch.tensor([1.e5], device=device)
    conc = torch.tensor([[2.e-2, 1.e-3, 1.e-4]], device=device)
    print(json.dumps(kin.forward(temp, pres, conc)[0].cpu().tolist()))
"""


def _run(case, card, device):
    return subprocess.run([sys.executable, "-c", SCRIPT, case, card, device],
                          capture_output=True, text=True, env=os.environ.copy())


def _no_crash(out, what):
    if out.returncode == 1 and "Traceback" in out.stderr:
        return False  # refused with a Python exception: acceptable
    assert out.returncode == 0, (
        f"{what}: returncode {out.returncode} (-11 = SIGSEGV)\n"
        f"stderr:\n{out.stderr}")
    return True


def _evaporation_none(card, device):
    ref = _run("default", card, device)
    assert ref.returncode == 0, ref.stderr
    out = _run("evaporation_none", card, device)
    if _no_crash(out, f"Kinetics after evaporation(None) on {device}"):
        assert json.loads(out.stdout.strip().splitlines()[-1]) == \
            json.loads(ref.stdout.strip().splitlines()[-1])


@pytest.fixture
def card(tmp_path):
    path = tmp_path / "ox.yaml"
    path.write_text(CARD)
    return str(path)


def test_evaporation_none(card):
    _evaporation_none(card, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_evaporation_none_cuda(card):
    _evaporation_none(card, "cuda:0")


def test_plain_options(card):
    _no_crash(_run("plain", card, "cpu"), "Kinetics(KineticsOptions())")


def test_none_options(tmp_path):
    path = tmp_path / "no_reference_state.yaml"
    path.write_text(CARD.replace("reference-state: {Tref: 300., Pref: 1.e5}\n", ""))
    _no_crash(_run("none", str(path), "cpu"), "Kinetics(from_yaml(card without reference-state))")
