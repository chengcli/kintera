"""A cell the saturation adjustment leaves unadjusted must be reported on every device.

ThermoY.forward marks such a cell with diag < 0, and when the caller passes no diag (as a
dynamical core does) the only report is a TORCH_WARN. The warning is issued on CPU only, so on
CUDA the same failure passes in silence. max-iter = 1 makes a strongly supersaturated NH4SH
cell fail deterministically.
"""
import pytest
import torch
from kintera import ThermoOptions, ThermoY

torch.set_default_dtype(torch.float64)

CARD = """reference-state: {Tref: 300., Pref: 1.e5}
dynamics: {equation-of-state: {max-iter: 1}}
species:
  - {name: dry, composition: {H: 1.5, He: 0.15}, cv_R: 2.5}
  - {name: NH3, composition: {N: 1, H: 3}, cv_R: 2.5, u0_R: 0.}
  - {name: H2S, composition: {H: 2, S: 1}, cv_R: 2.5, u0_R: 0.}
  - {name: NH3(s), composition: {N: 1, H: 3}, cv_R: 9.6, u0_R: -5520.}
  - {name: NH4SH(s), composition: {N: 1, H: 5, S: 1}, cv_R: 9.6, u0_R: -1.2e4}
reactions:
  - {equation: NH3 <=> NH3(s), type: nucleation, rate-constant: {formula: nh3_ideal}}
  - {equation: NH3 + H2S <=> NH4SH(s), type: nucleation, rate-constant: {formula: nh3_h2s_lewis}}
"""
# NH4SH 150x supersaturated, no cloud yet (test_saturation_adjustment_nh4sh.py, first cell)
CELL = (208.72174072916363, 0.8801790168247441, 5.279766354738258e-04, 1.428311049569326e-02,
        1.0099328092205351e-07, 0.)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_unadjusted_cell_is_reported(tmp_path, capfd, device):
    (path := tmp_path / "nh4sh.yaml").write_text(CARD)
    th = ThermoY(ThermoOptions.from_yaml(str(path)))
    th.to(torch.device(device))
    temp, rho, *y = torch.tensor([CELL], device=device).t()
    yfrac = torch.stack(y)
    intEng = th.compute("VT->U", [th.compute("DY->V", [rho, yfrac]), temp])

    diag = torch.zeros(1, 1, device=device)
    th.forward(rho, intEng, yfrac.clone(), False, diag)
    assert diag[0, 0] < 0, f"max-iter 1 must leave this cell unadjusted, diag = {diag.item()}"

    capfd.readouterr()
    th.forward(rho, intEng, yfrac.clone(), False)  # no diag, as a dynamical core calls it
    if device == "cuda":
        torch.cuda.synchronize()
    err = capfd.readouterr().err  # TORCH_WARN goes to stderr
    assert "saturation adjustment failed" in err, f"{device}: an unadjusted cell was not reported"
