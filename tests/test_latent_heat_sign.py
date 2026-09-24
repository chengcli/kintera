"""A condensate whose internal energy exceeds its vapour's must be reported.

`u0_R` is u/R at Tref, so the latent heat of `vapor => cloud` is u0_R[vapor] -
u0_R[cloud].  A condensate with no `u0_R` therefore has latent heat 0 -- condensation
absorbs energy instead of releasing it.

Each card runs in its OWN interpreter: kintera's species registry is global and
`ensure_species_initialized` is a no-op after the first card, so two cards loaded in
one process would both be scored against the first one's energies.
"""

import subprocess
import sys
import textwrap

CARD = """
reference-state: {{Tref: 300.0, Pref: 1.0e5}}
species:
- name: dry
  composition: {{H: 2}}
  cv_R: 2.5
- name: vapor
  composition: {{H: 2, O: 1}}
  cv_R: 3.5
- name: cloud
  composition: {{H: 2, O: 1}}
  cv_R: 9.0{u0}
reactions:
- equation: vapor => cloud
  type: nucleation
  rate-constant: {{formula: h2o_ideal, T3: 273.16, P3: 611.7, beta: 24.845, delta: 4.986}}
"""


def _load(tmp_path, u0):
    """Load one card in a fresh interpreter; return everything it printed."""
    card = tmp_path / ("card_%s.yaml" % ("offset" if u0 else "bare"))
    card.write_text(textwrap.dedent(CARD.format(u0=("\n  u0_R: %s" % u0) if u0 else "")))
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; from kintera import ThermoOptions; ThermoOptions.from_yaml(sys.argv[1])",
         str(card)],
        capture_output=True, text=True, check=True).stdout


def test_endothermic_condensate_is_reported(tmp_path):
    out = _load(tmp_path, None)
    assert "latent heat" in out and "vapor => cloud" in out, out


def test_physical_condensate_is_silent(tmp_path):
    assert "latent heat" not in _load(tmp_path, "-3430."), "warned on a physical card"
