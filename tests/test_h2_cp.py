"""use-h2-cp gate: the rotational partition-function heat capacity of H2.

cp/R of a pure H2 gas, from a finite difference of VT->U, at the values the option's commit
quotes: 50 K equilibrium 4.57 (the ortho/para conversion peak), 50 K normal 2.50 (rotation
frozen, translation only) and 300 K 3.45; plus the frozen limit at 20 K in normal mode.
The expected values pin this implementation (tolerance 2e-3).

Each ortho/para mode runs in its own interpreter: kintera's species registry is global.
"""

import json
import subprocess
import sys

CARD = """
reference-state: {Tref: 300.0, Pref: 1.0e5, use-h2-cp: true, h2-cp-mode: %s}
species:
- {name: H2, composition: {H: 2}, cv_R: 2.5}
"""

PROBE = r"""
import json, sys, torch
torch.set_default_dtype(torch.float64)
from kintera import ThermoY, ThermoOptions
th = ThermoY(ThermoOptions.from_yaml(sys.argv[1]))
R = 8.314462618
mu = 1.0 / float(th.inv_mu[0])
out = {}
for Tv in (20.0, 50.0, 300.0):
    rho = torch.tensor([1e5 * mu / (R * Tv)]); T = torch.tensor([Tv])
    V = th.compute("DY->V", (rho, torch.zeros((0, 1))))
    U0 = th.compute("VT->U", (V, T)); U1 = th.compute("VT->U", (V, T * 1.000001))
    out[str(Tv)] = ((U1 - U0) / (T * 1e-6) / (rho / mu * R)).item() + 1.0  # cp/R = cv/R + 1
print(json.dumps(out))
"""

EXPECTED = {  # cp/R
    "equilibrium": {"50.0": 4.5684, "300.0": 3.4526},
    "normal": {"20.0": 2.5000, "50.0": 2.5038, "300.0": 3.4522},
}


def _cp_R(tmp_path, mode):
    card = tmp_path / ("h2_%s.yaml" % mode)
    card.write_text(CARD % mode)
    res = subprocess.run([sys.executable, "-c", PROBE, str(card)],
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_equilibrium_mode_has_the_conversion_peak(tmp_path):
    got = _cp_R(tmp_path, "equilibrium")
    for T, want in EXPECTED["equilibrium"].items():
        assert abs(got[T] - want) < 2e-3, (T, got[T], want)


def test_normal_mode_freezes_rotation(tmp_path):
    got = _cp_R(tmp_path, "normal")
    for T, want in EXPECTED["normal"].items():
        assert abs(got[T] - want) < 2e-3, (T, got[T], want)


def test_h2_cp_mode_typo_is_rejected(tmp_path):
    import pytest
    from kintera import ThermoOptions

    card = tmp_path / "h2_typo.yaml"
    card.write_text(CARD % "Normal")
    with pytest.raises(RuntimeError, match="h2-cp-mode"):
        ThermoOptions.from_yaml(str(card))
