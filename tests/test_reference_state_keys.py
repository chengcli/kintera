"""reference-state keys are whitelisted: a typo raises and names the closest valid key."""

import pytest
from kintera import KineticsOptions, ThermoOptions

CARD = """
reference-state: {Tref: 300.0, Pref: 1.0e5, %s}
species:
- {name: dry, composition: {H: 2}, cv_R: 2.5}
"""
LOADERS = pytest.mark.parametrize("loader", [ThermoOptions, KineticsOptions])


@LOADERS
@pytest.mark.parametrize("typo, key", [("Tfer", "Tref"), ("use-h2cp", "use-h2-cp")])
def test_unknown_key_is_rejected_with_a_suggestion(tmp_path, loader, typo, key):
    card = tmp_path / "typo.yaml"
    card.write_text(CARD % ("%s: 1" % typo))
    msg = "unknown key 'reference-state/%s'; did you mean '%s'" % (typo, key)
    with pytest.raises(RuntimeError, match=msg):
        loader.from_yaml(str(card))


@LOADERS
def test_every_valid_key_is_accepted(tmp_path, loader):
    card = tmp_path / "valid.yaml"
    card.write_text(CARD % "use-nasa9-cp: false, use-h2-cp: false, h2-cp-mode: normal")
    assert loader.from_yaml(str(card)).Tref() == 300.0


@LOADERS
@pytest.mark.parametrize("block, kind", [("[300.0, 1.0e5]", "list"), ("300.0", "scalar")])
def test_non_map_block_is_rejected(tmp_path, loader, block, kind):
    card = tmp_path / "block.yaml"
    card.write_text(CARD.replace("{Tref: 300.0, Pref: 1.0e5, %s}", block))
    msg = "'reference-state' must be a map of key: value pairs.*got a %s" % kind
    with pytest.raises(RuntimeError, match=msg):
        loader.from_yaml(str(card))


@LOADERS
def test_empty_block_keeps_defaults(tmp_path, loader):
    card = tmp_path / "empty.yaml"
    card.write_text(CARD.replace("{Tref: 300.0, Pref: 1.0e5, %s}", ""))
    assert loader.from_yaml(str(card)).Tref() > 0.0
