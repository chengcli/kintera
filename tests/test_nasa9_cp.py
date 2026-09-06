"""use-nasa9-cp gate: NASA-9 heat capacity for species that carry coefficients.

Flag on: cp/R ("TV->cp") and cv/R ("VT->cv") equal the NASA-9 polynomial of data/nasa9.dat,
evaluated here with numpy, in both ranges (low < 1000 K <= high); U equals the constant-cv
value at 300 K and dU/dT = cv.  Flag off or absent: cp/R is the constant cv_R + 1.
Each species runs in its own interpreter: kintera's species registry is global.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TEMPS = [200.0, 300.0, 650.0, 999.0, 1001.0, 3000.0, 6000.0]
SPECIES = {"H2": ("{H: 2}", 2.5), "CH4": ("{C: 1, H: 4}", 3.3)}
CARD = """
reference-state: {Tref: 300.0, Pref: 1.0e5%s}
species:
- {name: %s, composition: %s, cv_R: %s}
"""

PROBE = r"""
import json, sys, torch
torch.set_default_dtype(torch.float64)
from kintera import ThermoX, ThermoY, ThermoOptions
R, T, out = 8.31446, torch.tensor(json.loads(sys.argv[1])), {}  # R = kintera Rgas
for key, card in zip(("on", "off", "default"), sys.argv[2:]):
    op = ThermoOptions.from_yaml(card)
    tx, ty = ThermoX(op), ThermoY(op)
    V = tx.compute("TPX->V", (T, 1e5 * torch.ones_like(T), torch.ones(T.shape + (1,))))
    c, ivol = V[..., 0] * R, V / ty.inv_mu
    U = lambda t: ty.compute("VT->U", (ivol, t)) / c
    out[key] = {"cp": (tx.compute("TV->cp", (T, V)) / c).tolist(),
                "cv": (ty.compute("VT->cv", (ivol, T)) / c).tolist(), "U": U(T).tolist(),
                "dUdT": ((U(T * (1 + 1e-6)) - U(T * (1 - 1e-6))) / (2e-6 * T)).tolist()}
print(json.dumps(out))
"""


def nasa9_cp_R(name, T):
    lines = (Path(__file__).parents[1] / "data" / "nasa9.dat").read_text().splitlines()
    i = [ln.strip() for ln in lines].index(name)
    v = np.array(" ".join(lines[i + 1 : i + 5]).split(), dtype=float)
    a = np.where((T < 1000.0)[:, None], v[None, 0:7], v[None, 10:17])
    return (a * T[:, None] ** np.arange(-2, 5)).sum(-1)


@pytest.fixture(scope="module", params=sorted(SPECIES))
def probe(request, tmp_path_factory):
    name, (comp, cv_R) = request.param, SPECIES[request.param]
    d = tmp_path_factory.mktemp(name)
    cards = []
    for key, flag in (("on", ", use-nasa9-cp: true"), ("off", ", use-nasa9-cp: false"),
                      ("default", "")):
        cards.append(str(d / (key + ".yaml")))
        Path(cards[-1]).write_text(CARD % (flag, name, comp, cv_R))
    res = subprocess.run([sys.executable, "-c", PROBE, json.dumps(TEMPS)] + cards,
                         capture_output=True, text=True, check=True)
    return name, cv_R, json.loads(res.stdout.strip().splitlines()[-1])


def test_nasa9_on_follows_the_polynomial(probe):  # rtol 1e-10 (cp, cv, U), 1e-6 (FD dU/dT)
    name, cv_R, got = probe
    want = nasa9_cp_R(name, np.array(TEMPS))
    assert np.abs(want - (cv_R + 1)).max() > 0.1  # the check discriminates from the constant
    np.testing.assert_allclose(got["on"]["cp"], want, rtol=1e-10, atol=0)
    np.testing.assert_allclose(got["on"]["cv"], want - 1.0, rtol=1e-10, atol=0)
    np.testing.assert_allclose(got["on"]["dUdT"], want - 1.0, rtol=1e-6, atol=0)
    i300 = TEMPS.index(300.0)
    np.testing.assert_allclose(got["on"]["U"][i300], got["off"]["U"][i300], rtol=1e-12, atol=0)


def test_nasa9_off_keeps_the_constant_cp(probe):
    name, cv_R, got = probe
    assert got["off"] == got["default"]  # explicit false == key absent, bit for bit
    np.testing.assert_allclose(got["off"]["cp"], cv_R + 1.0, rtol=1e-14, atol=0)
    np.testing.assert_allclose(got["off"]["cv"], cv_R, rtol=1e-14, atol=0)
