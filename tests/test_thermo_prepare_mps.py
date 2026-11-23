from pathlib import Path

import pytest
import torch

from kintera import ThermoOptions, ThermoY


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_prepare_moist_column(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires MPS")

    options = ThermoOptions.from_yaml(str(Path(__file__).with_name("earth.yaml")))
    thermo = ThermoY(options)
    cells = torch.empty((1, 4, 32), dtype=torch.float32, device=device)
    thermo.prepare(cells)
    epoch = thermo.tables()
    assert epoch >= 0
    storage = thermo.make_active_set_storage()
    reactions, nactive = thermo.snapshot_active_set(storage)
    assert reactions.device == nactive.device == cells.device
    assert reactions.shape[:-1] == nactive.shape == cells.shape
    assert reactions.size(-1) > 0
    assert nactive.count_nonzero().item() == 0
    thermo.reset_failures()
    assert thermo.failures().device == cells.device
    thermo.prepare(cells)
    assert thermo.tables() == epoch
