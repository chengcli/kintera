"""Saturation adjustment must converge when an absent reactant sits on a degenerate bound.

Card-40 thermo (NH3 ice, NH4SH, water). In each state below NH3 vapour is zero and both
NH3-consuming reactions are active; the NH3 >= 0 row is then met to within one ulp of the
extents, and the KKT active set used to cycle on that round-off violation until its iteration
limit, returning the cell unadjusted with diag < 0 (ISSUES K13). Rows are
(T [K], concentrations [mol/m^3] in species order).
"""
import math
import torch
from kintera import ThermoY, ThermoOptions

torch.set_default_dtype(torch.float64)
RGAS = 8.31446261815324

YAML = """
reference-state: {Tref: 0.0, Pref: 1.e5}
species:
- name: dry
  composition: {H: 1.6667, He: 0.16667}
  cv_R: 2.5
- name: H2O
  composition: {H: 2, O: 1}
  cv_R: 2.5
  u0_R: 0.0
- name: NH3
  composition: {N: 1, H: 3}
  cv_R: 2.5
  u0_R: 0.0
- name: H2S
  composition: {H: 2, S: 1}
  cv_R: 2.5
  u0_R: 0.0
- name: H2O(l)
  composition: {H: 2, O: 1}
  cv_R: 8.486009
  u0_R: -6786.6602
- name: H2O(l,p)
  composition: {H: 2, O: 1}
  cv_R: 8.486009
  u0_R: -6786.6602
- name: NH3(s)
  composition: {N: 1, H: 3}
  cv_R: 4.93
  u0_R: -4033.056
- name: NH3(s,p)
  composition: {N: 1, H: 3}
  cv_R: 4.93
  u0_R: -4033.056
- name: NH4SH(s)
  composition: {N: 1, H: 5, S: 1}
  cv_R: 7.0
  u0_R: -10833.618
- name: NH4SH(s,p)
  composition: {N: 1, H: 5, S: 1}
  cv_R: 7.0
  u0_R: -10833.618
reactions:
- equation: H2O => H2O(l)
  type: nucleation
  rate-constant: {formula: h2o_ideal}
- equation: H2O(l) => H2O(l,p)
  type: coagulation
  rate-constant: {A: 1.0e-05, b: 0, Ea_R: 0.0}
- equation: H2O(l,p) => H2O
  type: evaporation
  rate-constant: {formula: h2o_ideal, diff_c: 2.0e-05, diff_T: 0.0, diff_P: 0.0, vm: 1.8e-05, diameter: 0.0001}
- equation: NH3 => NH3(s)
  type: nucleation
  rate-constant: {formula: nh3_ideal}
- equation: NH3(s) => NH3(s,p)
  type: coagulation
  rate-constant: {A: 1.0e-05, b: 0, Ea_R: 0.0}
- equation: NH3(s,p) => NH3
  type: evaporation
  rate-constant: {formula: nh3_ideal, diff_c: 2.0e-05, diff_T: 0.0, diff_P: 0.0, vm: 1.8e-05, diameter: 0.0001}
- equation: NH3 + H2S <=> NH4SH(s)
  type: nucleation
  rate-constant: {formula: nh3_h2s_lewis}
- equation: NH4SH(s) => NH4SH(s,p)
  type: coagulation
  rate-constant: {A: 1.0e-05, b: 0, Ea_R: 0.0}
- equation: NH4SH(s,p) => NH3 + H2S
  type: evaporation
  rate-constant: {formula: nh3_h2s_lewis, diff_c: 2.0e-05, diff_T: 0.0, diff_P: 0.0, vm: 4.37e-05, diameter: 0.0001}
dynamics:
  equation-of-state: {max-iter: 20, ftol: 1.0e-06}
"""

CELLS = [
    (110.75017331617806, [85.35088402289688, 0.014859034461403325, 0.0, 0.024322199120572263, 0.0008790675918259672, 4.730584845932525e-05, 0.0008336349990618182, 2.3769386447302556e-05, 2.4671307479466126e-06, 2.2721988970655326e-05]),
    (111.73204885933957, [569.5955280783662, 0.000782515937538375, 0.0, 0.0254794845280547, 0.0, 0.00014138938746626308, 0.006345515530010961, 0.0002690245253959461, 0.003884315455991907, 6.357830800661039e-05]),
    (120.09949439012965, [12.855985526922757, 0.00012322467053248586, 0.0, 0.015327503815977894, 0.0, 6.981897950421539e-06, 4.239620024083938e-06, 8.748301546441749e-06, 1.2439080093606413e-05, 1.6335107550212979e-06]),
    (120.47342843860712, [164.43585431528703, 0.01200132548292692, 0.0, 0.05758833309108901, 0.0, 1.7324189949255654e-05, 4.686631631090797e-06, 4.694608549604164e-05, 9.96469478965261e-05, 9.408926575825834e-05]),
    (110.72187366656257, [709.7768757621728, 2.0476653964663356, 0.0, 0.30781718301684113, 0.0020428570636798864, 0.00014660524885596094, 4.327257451287053e-06, 5.6419200305339586e-05, 0.021433919546139036, 0.0006828962723402724]),
]


def _svp(T):
    ideal = lambda t, tr, pr, bl, gl, bs, gs: ((1 - tr / t) * bl - gl * math.log(t / tr) if t > tr
                                               else (1 - tr / t) * bs - gs * math.log(t / tr)) + math.log(pr)
    return [ideal(T, 273.16, 611.7, 24.845, 4.986009, 22.98, 0.52),
            ideal(T, 195.4, 6060., 20.08, 5.62, 20.64, 1.43),
            (14.82 - 4705. / T) * math.log(10.) + 2. * math.log(101325.)]


def test_absent_reactant_on_a_degenerate_bound_converges(tmp_path):
    p = tmp_path / "nh4sh.yaml"
    p.write_text(YAML)
    th = ThermoY(ThermoOptions.from_yaml(str(p)))
    inv_mu = th.inv_mu
    c = torch.tensor([row for _, row in CELLS])
    T = torch.tensor([t for t, _ in CELLS])
    V = c / inv_mu
    rho = V.sum(-1)
    y = (V[:, 1:] / rho[:, None]).T.contiguous()
    U = th.compute("VT->U", [V, T])
    diag = torch.zeros(len(CELLS), 1)
    th.forward(rho, U, y, False, diag)
    assert (diag >= 0).all(), "saturation adjustment failed: diag = %s" % diag.flatten().tolist()

    V_out = th.compute("DY->V", [rho, y])
    T_out = th.compute("VU->T", [V_out, U])
    c_out = V_out * inv_mu
    # species: dry H2O NH3 H2S H2O(l) H2O(l,p) NH3(s) NH3(s,p) NH4SH(s) NH4SH(s,p)
    for n in range(len(CELLS)):
        t = float(T_out[n]); co = c_out[n]
        rt = RGAS * t
        lnS = [math.log(max(float(co[1]), 1e-300) * rt) - _svp(t)[0],
               math.log(max(float(co[2]), 1e-300) * rt) - _svp(t)[1],
               math.log(max(float(co[2]), 1e-300) * rt) + math.log(max(float(co[3]), 1e-300) * rt) - _svp(t)[2]]
        cloud = [float(co[4]), float(co[6]), float(co[8])]
        for j in range(3):
            assert lnS[j] < 1e-4, (n, j, lnS[j])
            assert cloud[j] <= 0. or lnS[j] > -1e-4, (n, j, lnS[j], cloud[j])
