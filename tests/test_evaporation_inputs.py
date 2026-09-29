"""Evaporation inputs of tests/earth.yaml and tests/jupiter.yaml (#135).

These two files read like model configurations and get copied into runs.
The evaporation diffusivity is D = diff_c (T/Tref)^diff_T (P/Pref)^diff_P;
diff_T = diff_P = 0 freezes it at diff_c at every height, where kintera's
defaults are the gas-kinetic 1.75 and -1. Each condensate also takes its
own molar volume vm: NH3 ice 20.8e-6 m^3/mol (17.03 g/mol / 0.82 g/cm^3),
not water's 18e-6. Each file is read in its own process:
the species table is process-global and keeps the first file's species.
"""

import json
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
DIFF_T, DIFF_P = 1.75, -1.0  # kintera's defaults
NH3_ICE_VM = 20.8e-6  # m^3/mol

READ = """
import json, sys
import kintera as kt
ev = kt.KineticsOptions.from_yaml(sys.argv[1]).evaporation()
print(json.dumps({r.equation(): [dT, dP, vm] for r, dT, dP, vm in
                  zip(ev.reactions(), ev.diff_T(), ev.diff_P(), ev.vm())}))
"""


def evaporation(name):
    """{equation: [diff_T, diff_P, vm]} of the file's evaporation reactions."""
    out = subprocess.run([sys.executable, "-c", READ, str(HERE / name)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", ["earth.yaml", "jupiter.yaml"])
def test_evaporation_takes_the_default_diffusivity_exponents(name):
    reactions = evaporation(name)
    assert reactions
    for eq, (dT, dP, _) in reactions.items():
        assert (dT, dP) == pytest.approx((DIFF_T, DIFF_P), rel=1e-12), f"{name}: {eq}"


def test_ammonia_condensate_has_its_own_molar_volume():
    vm = evaporation("jupiter.yaml")["NH3(s,p) => NH3"][2]
    assert vm == pytest.approx(NH3_ICE_VM, rel=1e-12)
