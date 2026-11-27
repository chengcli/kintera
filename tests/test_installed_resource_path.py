"""An installed kintera resolves nasa9.dat through the C++ library from any directory.

A C++ consumer (snapy's run_hydro) never runs kintera/__init__.py, so the library starts
from its own search path '.' (find_resource.cpp); set_search_paths(".") restores that state.
The probe runs in a fresh interpreter from an empty directory with no venv variables.
"""

import os
import subprocess
import sys
from pathlib import Path

import kintera

PROBE = """
import kintera
kintera.set_search_paths(".")
print(kintera.find_resource("nasa9.dat"))
"""


def test_library_resolves_packaged_nasa9_from_empty_dir(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "CONDA_PREFIX")}
    res = subprocess.run([sys.executable, "-c", PROBE], cwd=tmp_path, env=env,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    found = Path(res.stdout.strip().splitlines()[-1])
    assert found.samefile(Path(kintera.__file__).with_name("data") / "nasa9.dat")
