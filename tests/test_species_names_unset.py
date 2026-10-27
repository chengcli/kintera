"""`SpeciesThermo.species()` must refuse unset names instead of segfaulting.

The ids are settable on their own, and `species()` falls back to the global
`species_names` whenever `names()` does not already cover them. That table is empty
until the registry is populated, so indexing it with an id read past the end of an
empty vector and took the interpreter down with signal 11 -- no traceback, no
diagnostic, and nothing a caller could catch.

A library has to raise here. The probe asks for the names with the ids set and the
table empty, and requires a Python exception and a live interpreter.

Run in a child interpreter: the crash is fatal, and the registry is process-global.
"""

import subprocess
import sys
import textwrap

SCRIPT = """
from kintera import SpeciesThermo

st = SpeciesThermo()
st.vapor_ids([1, 2, 3])
try:
    names = st.species()
except Exception as exc:
    print("RAISED", type(exc).__name__, str(exc).replace(chr(10), " ")[:200])
else:
    print("RETURNED", names)
"""


def test_species_with_unset_names_raises_instead_of_crashing():
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(SCRIPT)],
                       capture_output=True, text=True)

    assert r.returncode >= 0, (
        f"interpreter died on signal {-r.returncode} "
        f"({'SIGSEGV' if r.returncode == -11 else 'fatal'}); "
        f"species() read past the end of the global species names")
    assert r.returncode == 0, f"probe failed ({r.returncode}):\n{r.stderr}"

    out = r.stdout.strip()
    assert out.startswith("RAISED"), (
        f"species() should refuse unset names, got: {out}")
    assert "out of range" in out, f"unhelpful message: {out}"
