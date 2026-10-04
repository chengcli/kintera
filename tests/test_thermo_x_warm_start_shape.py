"""ThermoX warm starts are valid only for their original shape and device."""

import os
import subprocess
import sys

import pytest
import torch
from kintera import ThermoOptions, ThermoX

torch.set_default_dtype(torch.float64)

CARD = """reference-state: {Tref: 300., Pref: 1.e5}
species:
- {name: dry, composition: {N: 1.56, O: 0.42}, cv_R: 2.5}
- {name: A, composition: {H: 2, O: 1}, cv_R: 1.5, u0_R: 0.}
- {name: B, composition: {H: 3, N: 1}, cv_R: 1.5, u0_R: 0.}
- {name: A(l), composition: {H: 2, O: 1}, cv_R: 7.5, u0_R: -6786.66}
- {name: B(l), composition: {H: 3, N: 1}, cv_R: 7.5, u0_R: -5000.00}
reactions:
- {equation: 'A <=> A(l)', type: nucleation, rate-constant: {formula: antoine, A: 1.5, B: 900., C: 0.}}
- {equation: 'B <=> B(l)', type: nucleation, rate-constant: {formula: antoine, A: 0.0, B: 900., C: 0.}}
"""


def _state(n, device):
    temp = torch.full((n,), 300.0, device=device)
    pres = torch.full((n,), 1.0e5, device=device)
    xfrac = torch.tensor([0.96, 0.02, 0.02, 0.0, 0.0], device=device)
    return temp, pres, xfrac.expand(n, 5).clone()


def _matches_cold(thermo, temp, pres, xfrac):
    warm = xfrac.clone()
    thermo.forward(temp, pres, warm, True)
    cold = xfrac.clone()
    thermo.forward(temp, pres, cold, False)
    torch.testing.assert_close(warm, cold, rtol=0.0, atol=0.0)


def _prime_two_active_sets(thermo, device):
    temp = torch.full((2,), 300.0, device=device)
    pres = torch.tensor([1.0e5, 1.0e6], device=device)
    xfrac = torch.tensor(
        [[0.96, 0.02, 0.02, 0.0, 0.0], [0.98, 0.02, 1.0e-20, 0.0, 0.0]],
        device=device,
    )
    thermo.forward(temp, pres, xfrac, False)


def child(card, action):
    thermo = ThermoX(ThermoOptions.from_yaml(card))
    device = "cuda" if action == "cuda-shape" else "cpu"
    thermo.to(torch.device(device))
    _prime_two_active_sets(thermo, device)
    if action.endswith("shape"):
        _matches_cold(thermo, *_state(3, device))
        return 0

    thermo.to(torch.device("cuda"))
    _matches_cold(thermo, *_state(2, "cuda"))
    thermo.to(torch.device("cpu"))
    _matches_cold(thermo, *_state(2, "cpu"))
    return 0


def _run_child(card, action):
    run = subprocess.run(
        [sys.executable, os.path.abspath(__file__), str(card), action],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert run.returncode == 0, (
        "%s: rc %d (negative = killed by that signal)\n%s%s"
        % (action, run.returncode, run.stdout[-2000:], run.stderr[-2000:])
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_warm_start_on_a_larger_batch(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no cuda")
    (card := tmp_path / "two_clouds.yaml").write_text(CARD)
    _run_child(card, device + "-shape")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_warm_start_after_device_move(tmp_path):
    (card := tmp_path / "two_clouds.yaml").write_text(CARD)
    _run_child(card, "migrate")


if __name__ == "__main__":
    sys.exit(child(sys.argv[1], sys.argv[2]))
