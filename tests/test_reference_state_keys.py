"""reference-state keys are whitelisted: a typo raises and names the closest valid key."""

import pytest
from kintera import ThermoOptions

CARD = """
reference-state: {Tref: 300.0, Pref: 1.0e5, %s}
species:
- {name: dry, composition: {H: 2}, cv_R: 2.5}
"""


@pytest.mark.parametrize("typo, key", [("Tfer", "Tref"), ("use-h2cp", "use-h2-cp")])
def test_unknown_key_is_rejected_with_a_suggestion(tmp_path, typo, key):
    card = tmp_path / "typo.yaml"
    card.write_text(CARD % ("%s: 1" % typo))
    msg = "unknown key 'reference-state/%s'; did you mean '%s'" % (typo, key)
    with pytest.raises(RuntimeError, match=msg):
        ThermoOptions.from_yaml(str(card))


def test_every_valid_key_is_accepted(tmp_path):
    card = tmp_path / "valid.yaml"
    card.write_text(CARD % "use-nasa9-cp: false, use-h2-cp: false, h2-cp-mode: normal")
    assert ThermoOptions.from_yaml(str(card)).Tref() == 300.0
