"""Saturation adjustment with two clouds that share a vapour (NH3 ice, NH4SH), so the KKT path runs:
each Newton step has two reaction extents and four species non-negativity rows, any three of them
linearly dependent. An adjusted cell has no reaction supersaturated, no cloud in undersaturated air,
no negative concentration, and conserves N and S; a ThermoY cell that is not adjusted has diag < 0.
"""
import torch
from kintera import ThermoOptions, ThermoX, ThermoY, relative_humidity

torch.set_default_dtype(torch.float64)

CARD = """reference-state: {Tref: 300., Pref: 1.e5}
dynamics: {equation-of-state: {max-iter: MAXITER}}
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
# (T [K], rho [kg/m^3], mass fractions of NH3, H2S, NH3(s), NH4SH(s))
CELLS = [
    (208.72174072916363, 0.8801790168247441, 5.279766354738258e-04, 1.428311049569326e-02, 1.0099328092205351e-07, 0.),  # NH4SH 150x supersaturated, no cloud yet
    (162.72769929280207, 1.237723071684808, 3.328274229036432e-06, 8.2547022144026e-06, 1.1644586821086675e-07, 1.2924097121314215e-03),  # NH4SH supersaturated
    (214.46839123415936, 0.07097992788360383, 1.5019306038899767e-04, 1.8675641220304122e-02, 2.7610173495375953e-04, 1.044385688850282e-03),  # both clouds must evaporate
    (229.91058717234685, 0.11580839644086043, 1.0678578796182898e-06, 9.615007243162296e-03, 4.924820461033533e-05, 1.4954502082092486e-03),  # both clouds must evaporate
]


def _adjust(tmp_path, temp, rho, yfrac, max_iter):
    (path := tmp_path / "nh4sh.yaml").write_text(CARD.replace("MAXITER", str(max_iter)))
    th = ThermoY(ThermoOptions.from_yaml(str(path)))
    inv_mu, ivol = th.buffer("inv_mu"), th.compute("DY->V", [rho, yfrac])
    intEng = th.compute("VT->U", [ivol, temp])
    diag = torch.zeros(temp.numel(), 1)
    th.forward(rho, intEng, yfrac, False, diag)
    ivol1 = th.compute("DY->V", [rho, yfrac])
    conc, temp1 = ivol1 * inv_mu, th.compute("VU->T", [ivol1, intEng])
    rh = relative_humidity(temp1, conc, th.buffer("stoich"), th.options.nucleation())
    return ivol * inv_mu, conc, rh, diag[:, 0]


def _unadjusted(conc0, conc, rh, diag, tol=1.e-5):
    floor = 1.e-12 * conc[:, 1:].sum(-1, keepdim=True)  # round-off, relative to the non-dry total
    bad = (rh > 1. + tol).any(-1) | ((rh < 1. - tol) & (conc[:, 3:] > floor)).any(-1)
    bad |= (conc < -floor).any(-1)
    for rows in ([1, 3, 4], [2, 4]):  # nitrogen, sulfur
        torch.testing.assert_close(conc[:, rows].sum(-1), conc0[:, rows].sum(-1), rtol=1.e-12, atol=0.)
    return bad


def test_listed_cells_are_adjusted_at_default_budget(tmp_path):
    temp, rho, *y = torch.tensor(CELLS).t()
    conc0, conc, rh, diag = _adjust(tmp_path, temp, rho, torch.stack(y), 10)
    assert not _unadjusted(conc0, conc, rh, diag).any() and (diag >= 0).all(), f"diag {diag} rh {rh}"


def test_no_cell_fails_silently(tmp_path):
    g = torch.Generator().manual_seed(20260923)
    u = lambda: torch.rand(2000, generator=g)
    logu = lambda lo, hi: 10. ** (lo + (hi - lo) * u())
    temp = 100. + 200. * u(); rho = logu(4., 6.) * 2.3e-3 / (8.314462618 * temp)
    y = [logu(-6., -1.5), logu(-6., -1.5)] + [torch.where(u() < 0.5, 0., logu(-7., -2.)) for _ in "ab"]
    conc0, conc, rh, diag = _adjust(tmp_path, temp, rho, torch.stack(y), 50)
    bad = _unadjusted(conc0, conc, rh, diag)
    assert not (bad & (diag >= 0)).any(), f"{int((bad & (diag >= 0)).sum())} cells unadjusted, diag >= 0"
    assert int((diag < 0).sum()) <= 10, f"{int((diag < 0).sum())} of 2000 cells failed"


def test_thermox_keeps_condensates_non_negative(tmp_path):
    (path := tmp_path / "nh4sh_tp.yaml").write_text(CARD.replace("MAXITER", "50"))
    g = torch.Generator().manual_seed(7)
    x = 10. ** (-6. + 4.5 * torch.rand(2000, 4, generator=g))
    x[:, 2:] *= torch.rand(2000, 2, generator=g) < 0.5
    x = torch.cat([1. - x.sum(-1, keepdim=True), x], -1)
    temp, pres = 100. + 200. * torch.rand(2000, generator=g), 10. ** (4. + 2. * torch.rand(2000, generator=g))
    ThermoX(ThermoOptions.from_yaml(str(path))).forward(temp, pres, x)
    assert (x[:, 1:] >= -1.e-12 * x[:, 1:].sum(-1, keepdim=True)).all(), x[:, 1:].min(0).values
