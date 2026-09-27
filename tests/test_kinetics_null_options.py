"""Kinetics must not crash on KineticsOptions whose sub-options are null.

KineticsOptions() leaves arrhenius, coagulation, evaporation, ... null, a
caller can set one to None, and KineticsOptions.from_yaml returns None for a
card without reference-state. The Kinetics constructor dereferences them all.
Each path either refuses with a Python exception or builds a Kinetics that
runs; with any of the eight sub-options set to None on this card the rate
must equal the default's (none for arrhenius, the card's only reaction
type), and after Kinetics(op) the sub-option must read
back as empty options, not None. Each case runs in a fresh subprocess,
because a segfault would kill pytest.
"""

import json
import os
import subprocess
import sys

import pytest
import torch

SUB_OPTIONS = ["arrhenius", "coagulation", "evaporation", "three_body",
               "lindemann_falloff", "troe_falloff", "sri_falloff", "kb_falloff"]

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
else:  # "default" or "<sub-option>_none"
    op = KineticsOptions.from_yaml(card)
    name = case[:-len("_none")] if case.endswith("_none") else None
    if name:
        getattr(op, name)(None)
    kin = Kinetics(op)
kin.to(device)
if case not in ("plain", "none"):
    temp = torch.tensor([250.], device=device)
    pres = torch.tensor([1.e5], device=device)
    conc = torch.tensor([[2.e-2, 1.e-3, 1.e-4]], device=device)
    rate = kin.forward(temp, pres, conc)[0].cpu().tolist()
    print(json.dumps({"rate": rate,
                      "none_after": name is not None and getattr(op, name)() is None}))
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


def _sub_option_none(card, device, name):
    ref = _run("default", card, device)
    assert ref.returncode == 0, ref.stderr
    out = _run(f"{name}_none", card, device)
    if _no_crash(out, f"Kinetics after {name}(None) on {device}"):
        got = json.loads(out.stdout.strip().splitlines()[-1])
        # the card's one reaction is Arrhenius: dropping it leaves no rates
        want = [[]] if name == "arrhenius" else \
            json.loads(ref.stdout.strip().splitlines()[-1])["rate"]
        assert got["rate"] == want
        assert not got["none_after"], f"{name} still None after Kinetics(op)"


@pytest.fixture
def card(tmp_path):
    path = tmp_path / "ox.yaml"
    path.write_text(CARD)
    return str(path)


@pytest.mark.parametrize("name", SUB_OPTIONS)
def test_sub_option_none(card, name):
    _sub_option_none(card, "cpu", name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("name", SUB_OPTIONS)
def test_sub_option_none_cuda(card, name):
    _sub_option_none(card, "cuda:0", name)


def test_plain_options(card):
    _no_crash(_run("plain", card, "cpu"), "Kinetics(KineticsOptions())")


def test_none_options(tmp_path):
    path = tmp_path / "no_reference_state.yaml"
    path.write_text(CARD.replace("reference-state: {Tref: 300., Pref: 1.e5}\n", ""))
    _no_crash(_run("none", str(path), "cpu"), "Kinetics(from_yaml(card without reference-state))")
