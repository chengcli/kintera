"""The example scripts run.

Three of them did not, each for its own reason, and none of them was a regression:

  example_chapman_cycle.py  called Arrhenius.forward with three arguments; the
                            binding requires a fourth, `other`, with no default.
  example_earth_dry.py      called extrapolate_ad, which was split into
                            extrapolate_dlnp and extrapolate_dz. The replacement
                            takes dz before grav, the reverse of the old call, so
                            the two had to be reordered and not merely renamed.
  test_type_hints.py        raised Skipped at import from a module-level
                            pytest.skip, so running it directly exited 1 instead
                            of reaching its own __main__ banner.

Each is copied to a scratch directory and run there, so a script that writes
output cannot touch the source tree. They are run as `python <name>.py`, which is
how a reader runs an example.

What these checks catch: a script that raises, dies on a signal or times out, and
an earth column that comes back isothermal because extrapolate_dz never wrote
through its layer slices.

What they do not catch: dz and grav passed in the wrong order. With ds_dz left at
its default of zero, as example_earth_dry.py leaves it, the hydrostatic step uses
the two only through the product grav * dz -- in the isothermal first guess and
in p0 - 0.5 * (rho + rho0) * grav * dz -- so exchanging them cannot change the
answer. Measured on this script's dz = 100 m and grav = 9.8 m/s^2, the swapped
run returns temperature and pressure bit for bit identical and every check below
still passes. Only a non-zero ds_dz, which multiplies dz on its own, would make
the order observable.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCRIPTS = ("example_chapman_cycle.py", "example_earth_dry.py", "test_type_hints.py")
TIMEOUT = 180


@pytest.mark.parametrize("name", SCRIPTS)
def test_example_exits_zero(name, tmp_path):
    src = EXAMPLES / name
    assert src.exists(), f"{name} is missing from examples/"

    work = tmp_path / "examples"
    shutil.copytree(EXAMPLES, work)

    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MPLBACKEND="Agg")

    r = subprocess.run([sys.executable, name], cwd=work, env=env,
                       capture_output=True, text=True, timeout=TIMEOUT)

    assert r.returncode >= 0, f"{name} died on signal {-r.returncode}"
    assert r.returncode == 0, (
        f"{name} exited {r.returncode}\n"
        f"--- stdout ---\n{r.stdout[-2000:]}\n--- stderr ---\n{r.stderr[-2000:]}")


def test_earth_dry_column_is_not_isothermal(tmp_path):
    """example_earth_dry.py must produce a real adiabat, not a flat column.

    With extrapolate_dz rebinding its slices the script still exited 0 once the call
    was corrected, but left every layer at the surface temperature. Exit status alone
    would not have caught that.
    """
    work = tmp_path / "examples"
    shutil.copytree(EXAMPLES, work)

    probe = work / "_probe_earth_dry.py"
    # the profile is built under `if __name__ == "__main__"`, so run it as main
    probe.write_text(
        "import runpy\n"
        "g = runpy.run_path('example_earth_dry.py', run_name='__main__')\n"
        "temp = g['temp']\n"
        "print('SPREAD', float(temp.max() - temp.min()))\n"
    )

    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MPLBACKEND="Agg")

    r = subprocess.run([sys.executable, probe.name], cwd=work, env=env,
                       capture_output=True, text=True, timeout=TIMEOUT)
    assert r.returncode == 0, f"probe exited {r.returncode}\n{r.stderr[-2000:]}"

    spread = float([l for l in r.stdout.splitlines() if l.startswith("SPREAD")][-1].split()[1])
    assert spread > 1.0, (
        f"the whole column is within {spread} K of itself; "
        f"extrapolate_dz is not updating the layer slices")
