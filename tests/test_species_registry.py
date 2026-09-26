"""A card's thermodynamics must not depend on which card the process loaded first.

Card A (dry = N2, vapour H2O) is loaded, then card B, which uses the same species names
with a different dry gas (dry = H2). B's ThermoY must use B's molar masses, as it does
when B is the only card in the process (reference: B alone in a fresh subprocess).
"""

import json
import os
import subprocess
import sys

import pytest
import torch
import kintera
from kintera import ThermoOptions, ThermoY

torch.set_default_dtype(torch.float64)

CARD = """reference-state: {{Tref: 300., Pref: 1.e5}}
species:
  - {{name: dry, composition: {dry}, cv_R: 2.5}}
  - {{name: H2O, composition: {{H: 2, O: 1}}, cv_R: 2.5}}
  - {{name: H2O(l), composition: {{H: 2, O: 1}}, cv_R: 9.0, u0_R: -3430.}}
reactions:
  - {{equation: H2O <=> H2O(l), type: nucleation,
     rate-constant: {{formula: h2o_ideal}}}}
"""
CARD_A = CARD.format(dry="{N: 2}")
CARD_B = CARD.format(dry="{H: 2}")
YFRAC = [[0.01, 0.2, 0.5], [0.001, 0.05, 0.1]]  # (ny = 2: H2O, H2O(l)) x 3 cells

FRESH = """
import json, sys, torch
from kintera import ThermoOptions, ThermoY
torch.set_default_dtype(torch.float64)
th = ThermoY(ThermoOptions.from_yaml(sys.argv[1]))
print(json.dumps({"mu": (1. / th.inv_mu).tolist(),
                  "X": th.compute("Y->X", (torch.tensor(%r),)).tolist()}))
""" % (YFRAC,)


def _write(tmp_path, name, card):
    (path := tmp_path / name).write_text(card)
    return str(path)


def _alone(path):
    out = subprocess.run([sys.executable, "-c", FRESH, path], check=True,
                         capture_output=True, text=True, env=os.environ.copy())
    return json.loads(out.stdout.strip().splitlines()[-1])


def _second_card(tmp_path, device):
    ThermoY(ThermoOptions.from_yaml(_write(tmp_path, "a.yaml", CARD_A)))
    path_b = _write(tmp_path, "b.yaml", CARD_B)
    ref = _alone(path_b)
    th = ThermoY(ThermoOptions.from_yaml(path_b))
    th.to(torch.device(device))
    X = th.compute("Y->X", (torch.tensor(YFRAC, device=device),))
    mu = (1. / th.inv_mu).cpu()
    print("device", X.device, "| B alone: mu", ref["mu"], "X[0]", ref["X"][0],
          "| B after A: mu", mu.tolist(), "X[0]", X[0].tolist())
    torch.testing.assert_close(X.cpu(), torch.tensor(ref["X"]), rtol=1e-12, atol=0.)
    torch.testing.assert_close(mu, torch.tensor(ref["mu"]), rtol=1e-12, atol=0.)


def test_second_card_uses_its_own_molar_masses(tmp_path):
    _second_card(tmp_path, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_second_card_uses_its_own_molar_masses_cuda(tmp_path):
    _second_card(tmp_path, "cuda:0")
