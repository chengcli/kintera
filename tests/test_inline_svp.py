"""Inline saturation vapour pressure: rate-constant {formula: ideal|antoine}."""

import pytest
import torch
from kintera import ThermoOptions, ThermoX, ThermoY, constants, relative_humidity

torch.set_default_dtype(torch.float64)

CARD = """reference-state: {Tref: 0., Pref: 1.e5}
dynamics: {equation-of-state: {max-iter: 50, ftol: 1.e-12}}
species:
  - {name: dry, composition: {N: 2}, cv_R: 2.5}
  - {name: H2O, composition: {H: 2, O: 1}, cv_R: 2.5}
  - {name: H2O(l), composition: {H: 2, O: 1}, cv_R: 9.0, u0_R: -3430.}
reactions:
  - {equation: H2O <=> H2O(l), type: nucleation, rate-constant: RATE}
"""
# named curve, inline 'ideal' with its constants; nh3 guards the h2o_ideal sentinel
CURVES = {"h2o": ("{formula: h2o_ideal}", "{formula: ideal, T3: 273.16, P3: "
                  "611.7, beta: 24.845, gamma: 4.986009, betas: 22.98, gammas: 0.52}"),
          "nh3": ("{formula: nh3_ideal}", "{formula: ideal, T3: 195.4, P3: 6060., "
                  "beta: 20.08, gamma: 5.62, betas: 20.64, gammas: 1.43}")}
A, B, C = 5.40221, 1838.675, -31.737  # Antoine coefficients (bar, K)
ANTOINE = f"{{formula: antoine, A: {A}, B: {B}, C: {C}}}"
TEMP = torch.cat([torch.linspace(150., 400., 2501), torch.tensor([273.16, 195.4])])
PRES = torch.full_like(TEMP, 1.e5)  # TEMP: both branches + triple points
UNIT = torch.tensor([0., 1., 0.]).repeat(TEMP.numel(), 1)  # 1 mol/m^3 of vapour

def options(tmp_path, rate):
    (path := tmp_path / f"card{hash(rate)}.yaml").write_text(CARD.replace("RATE", rate))
    return ThermoOptions.from_yaml(str(path))

def humidity(thermo, conc=UNIT):  # rh(UNIT) = R T / svp
    return relative_humidity(TEMP, conc, thermo.buffer("stoich"),
                             thermo.options.nucleation())[..., 0]

def equilibrate_tp(thermo):
    xfrac = torch.tensor([0.9, 0.1, 0.]).repeat(TEMP.numel(), 1)
    thermo.forward(TEMP, PRES, xfrac)
    return xfrac

@pytest.mark.parametrize("gas", CURVES)
def test_ideal_logsvp_bitwise(tmp_path, gas):
    assert torch.equal(*(humidity(ThermoX(options(tmp_path, r))) for r in CURVES[gas]))

@pytest.mark.parametrize("gas", CURVES)
def test_ideal_equilibrate_tp_bitwise(tmp_path, gas):
    xn, xi = (equilibrate_tp(ThermoX(options(tmp_path, r))) for r in CURVES[gas])
    assert torch.equal(xi, xn) and 0 < xn[..., 2].gt(0).sum() < TEMP.numel()

@pytest.mark.parametrize("gas", CURVES)
@pytest.mark.parametrize("solver", ["partition", "kkt"])
def test_ideal_equilibrate_uv_bitwise(tmp_path, solver, gas):
    out = []
    for rate in CURVES[gas]:
        thermo = ThermoY(options(tmp_path, rate).uv_solver(solver))
        yfrac = torch.tensor([[0.05], [0.0]]).repeat(1, TEMP.numel())
        rho = torch.ones_like(TEMP)
        ivol = thermo.compute("DY->V", [rho, yfrac])
        thermo.forward(rho, thermo.compute("VT->U", [ivol, TEMP]), yfrac)
        out.append(yfrac)
    assert torch.equal(out[1], out[0]) and 0 < out[0][1].gt(0).sum() < TEMP.numel()

def test_antoine_logsvp_analytic(tmp_path):
    t = TEMP.numpy()
    expected = constants.Rgas * t / (1.e5 * 10. ** (A - B / (t + C)))
    got = humidity(ThermoX(options(tmp_path, ANTOINE))).numpy()
    err = abs(got / expected - 1.).max()
    print(f"antoine logsvp: max relative error {err:.3e}")
    assert err < 1.e-12

def test_antoine_equilibrate_tp_saturates(tmp_path):
    thermo = ThermoX(options(tmp_path, ANTOINE))
    xfrac = equilibrate_tp(thermo)
    rh = humidity(thermo, thermo.compute("TPX->V", [TEMP, PRES, xfrac]))
    cloudy = xfrac[..., 2] > 0
    err = (rh[cloudy] - 1.).abs().max().item()
    print(f"antoine equilibrate_tp: max |rh - 1| over cloudy cells {err:.3e}")
    assert err < 1.e-8 and not cloudy.all() and (rh[~cloudy] < 1.).all()

def test_inline_formula_set_from_python_raises(tmp_path):
    op = options(tmp_path, CURVES["h2o"][0])
    stoich = ThermoX(op).buffer("stoich")
    op.nucleation().logsvp(["ideal"])  # a python-set inline name has no parameters
    msg = "inline 'ideal'/'antoine' formulas must be defined in YAML"
    with pytest.raises(RuntimeError, match=msg):  # LogSVPFunc::init
        relative_humidity(TEMP, UNIT, stoich, op.nucleation())
    with pytest.raises(RuntimeError, match=msg):  # make_svp_spec
        equilibrate_tp(ThermoX(op))

def test_ideal_rejects_nonpositive_p3(tmp_path):
    with pytest.raises(RuntimeError, match="requires T3 > 0 and P3 > 0"):
        options(tmp_path, CURVES["h2o"][1].replace("P3: 611.7", "P3: 0."))
