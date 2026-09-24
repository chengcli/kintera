"""Evaporation with two gaseous products.

Diffusion from a particle whose surface holds the two products at equilibrium,
(c1 + x)(c2 + x) = K, with equal diffusivities, gives the rate kappa * Cp * x:
x is the reaction extent that would bring the ambient air to equilibrium at
fixed temperature. The previous law used kappa * Cp * (K - c1 c2), which has one
concentration unit too many.

kintera's species registry is global, so this file registers its own species and
the Kinetics test loads its card in a separate interpreter.
"""

import json
import math
import subprocess
import sys
import textwrap

import pytest
import torch

import kintera as kt

torch.set_default_dtype(torch.float64)

RGAS = 8.31446  # kintera constants.h
SPECIES = ["H2O", "NH3", "H2S", "H2O(l,p)", "NH4SH(s,p)"]
I_H2O, I_NH3, I_H2S, I_LP, I_SP = range(5)
DIFF_C, VM, DIAM = 2.0e-5, 43.7e-6, 1.0e-4
KAPPA = 12.0 * DIFF_C * VM / DIAM**2
T0, P0 = 230.0, 4.0e5


@pytest.fixture(autouse=True)
def _register_species():
    kt.set_species_names(SPECIES)
    kt.set_species_weights([18e-3, 17e-3, 34e-3, 18e-3, 51e-3])


def _module(equations, formulas):
    op = kt.EvaporationOptions()
    op.reactions([kt.Reaction(eq) for eq in equations])
    op.logsvp(formulas)
    n = len(equations)
    op.diff_c([DIFF_C] * n).diff_T([0.0] * n).diff_P([0.0] * n)
    op.vm([VM] * n).diameter([DIAM] * n)
    op.minT([0.0] * n).maxT([1.0e4] * n)
    return kt.Evaporation(op)


def _both():
    return _module(["H2O(l,p) => H2O", "NH4SH(s,p) => NH3 + H2S"],
                   ["h2o_ideal", "nh3_h2s_lewis"])


def _stoich(nrxn):
    s = torch.zeros(len(SPECIES), nrxn)
    s[I_H2O, 0], s[I_LP, 0] = 1.0, -1.0
    if nrxn > 1:
        s[I_NH3, 1], s[I_H2S, 1], s[I_SP, 1] = 1.0, 1.0, -1.0
    return s


def _K(T):
    return math.exp((14.82 - 4705.0 / T) * math.log(10.0)
                    + 2.0 * math.log(101325.0) - 2.0 * math.log(RGAS * T))


def _conc(c_nh3, c_h2s, cp=1.0e-3, c_h2o=1.0e-3, cl=1.0e-3):
    c = torch.zeros(1, len(SPECIES))
    c[0, I_H2O], c[0, I_NH3], c[0, I_H2S] = c_h2o, c_nh3, c_h2s
    c[0, I_LP], c[0, I_SP] = cl, cp
    return c


def _rate(mod, conc, nrxn, expanded=True, grad=False, temp=T0, dtype=torch.float64):
    T, P = torch.tensor([temp], dtype=dtype), torch.tensor([P0], dtype=dtype)
    conc = conc.to(dtype)
    if expanded:
        c = conc.unsqueeze(-1).expand(1, len(SPECIES), nrxn).clone()
    else:
        c = conc.clone()
    if grad:
        c.requires_grad_(True)
    k = mod.forward(T, P, c, {"stoich": _stoich(nrxn).to(dtype)})
    return k, c


def _x_ref(c1, c2, K):
    c1, c2 = max(c1, 0.0), max(c2, 0.0)
    den = (c1 + c2) + math.sqrt((c1 - c2) ** 2 + 4.0 * K)
    return max(2.0 * (K - c1 * c2) / den, 0.0)


STATES = [
    (0.0669, 0.00209),    # NH3-rich, subsaturated
    (0.00669, 0.000209),  # far below saturation
    (0.05, 0.05),         # equal abundances
    (0.0, 0.0),           # empty air
    (0.0805, 0.0805),     # just below saturation (S = 0.9995)
    (0.1, 0.1),           # supersaturated: no evaporation
]


@pytest.mark.parametrize("c1,c2", STATES)
def test_two_product_rate_is_the_diffusion_extent(c1, c2):
    K = _K(T0)
    k, _ = _rate(_both(), _conc(c1, c2), 2)
    x = k[0, 1].item() / KAPPA
    assert x == pytest.approx(_x_ref(c1, c2, K), rel=1e-10, abs=1e-300)
    if c1 * c2 < K:
        assert (c1 + x) * (c2 + x) / K == pytest.approx(1.0, rel=1e-10)


@pytest.mark.parametrize("expanded", [True, False])
def test_single_product_reaction_is_untouched_by_a_two_product_neighbour(expanded):
    conc = _conc(0.0669, 0.00209, c_h2o=2.0e-3)
    k_both, c_both = _rate(_both(), conc, 2, expanded=expanded, grad=True)
    k_both[0, 0].backward()
    one = _module(["H2O(l,p) => H2O"], ["h2o_ideal"])
    k_one, c_one = _rate(one, conc, 1, expanded=expanded, grad=True)
    k_one[0, 0].backward()
    assert k_both[0, 0].item() == k_one[0, 0].item()
    g_both = c_both.grad[0, :, 0] if expanded else c_both.grad[0]
    g_one = c_one.grad[0, :, 0] if expanded else c_one.grad[0]
    assert torch.equal(g_both, g_one)


def test_expanded_and_unexpanded_concentrations_agree():
    conc = _conc(0.0669, 0.00209)
    k_exp, _ = _rate(_both(), conc, 2, expanded=True)
    k_raw, _ = _rate(_both(), conc, 2, expanded=False)
    assert torch.equal(k_exp, k_raw)


def test_rate_derivative_matches_finite_difference():
    conc = _conc(0.0669, 0.00209)
    k, c = _rate(_both(), conc, 2, grad=True)
    k[0, 1].backward()
    for s in (I_NH3, I_H2S):
        h = 1.0e-7
        cp_, cm_ = conc.clone(), conc.clone()
        cp_[0, s] += h
        cm_[0, s] -= h
        fd = (_rate(_both(), cp_, 2)[0][0, 1] - _rate(_both(), cm_, 2)[0][0, 1]) / (2 * h)
        assert c.grad[0, s, 1].item() == pytest.approx(fd.item(), rel=1e-6)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("c", [0.0, 1.0e-3])
def test_underflowed_saturation_gives_finite_rate_and_gradient(dtype, c):
    # at 10 K the Lewis K underflows to exactly zero
    mod = _both()
    mod.to(dtype)
    k, conc = _rate(mod, _conc(c, c), 2, grad=True, temp=10.0, dtype=dtype)
    k[0, 1].backward()
    assert k[0, 1].item() == 0.0
    assert torch.isfinite(conc.grad).all()


def test_negative_products_evaporate_as_into_empty_air():
    k, _ = _rate(_both(), _conc(-1.0e-3, -1.0e-3), 2)
    assert math.isfinite(k[0, 1].item())
    assert k[0, 1].item() / KAPPA == pytest.approx(math.sqrt(_K(T0)), rel=1e-12)


@pytest.mark.parametrize("equation", [
    "NH4SH(s,p) => 2 NH3",
    "2 NH4SH(s,p) => NH3 + H2S",
    "NH4SH(s,p) => NH3 + H2S + H2O",
    "NH4SH(s,p) + H2O => NH3 + H2S",
    "NH4SH(s,p) <=> NH3 + H2S",
])
def test_other_stoichiometries_are_rejected(equation):
    with pytest.raises(RuntimeError, match="one reactant and one or two products"):
        _module([equation], ["nh3_h2s_lewis"])


@pytest.mark.parametrize("equations,formulas", [
    (["H2O(l,p) => H2O", "NH4SH(s,p) => NH3 + H2O(l,p)"], ["h2o_ideal", "nh3_h2s_lewis"]),
    (["NH4SH(s,p) => NH3 + H2O(l,p)", "H2O(l,p) => H2O"], ["nh3_h2s_lewis", "h2o_ideal"]),
])
def test_condensate_product_is_rejected(equations, formulas):
    with pytest.raises(RuntimeError, match=r"has the condensate 'H2O\(l,p\)' as a product"):
        _module(equations, formulas)


CARD = """
reference-state: {Tref: 300., Pref: 1.e5}
species:
  - {name: dry, composition: {H: 2}, cv_R: 2.5}
  - {name: H2O, composition: {H: 2, O: 1}, cv_R: 2.5, u0_R: 0.}
  - {name: "H2O(l,p)", composition: {H: 2, O: 1}, cv_R: 9.0, u0_R: -3430.}
  - {name: NH3, composition: {N: 1, H: 3}, cv_R: 2.5, u0_R: 0.}
  - {name: H2S, composition: {H: 2, S: 1}, cv_R: 2.5, u0_R: 0.}
  - {name: "NH4SH(s,p)", composition: {N: 1, H: 5, S: 1}, cv_R: 9.0, u0_R: -11000.}
reactions:
  - equation: H2O(l,p) => H2O
    type: evaporation
    rate-constant: {formula: h2o_ideal, diff_c: 2.0e-5, diff_T: 0., diff_P: 0.,
                    vm: 18.e-6, diameter: 1.0e-4}
  - equation: NH4SH(s,p) => NH3 + H2S
    type: evaporation
    rate-constant: {formula: nh3_h2s_lewis, diff_c: 2.0e-5, diff_T: 0., diff_P: 0.,
                    vm: 43.7e-6, diameter: 1.0e-4}
"""

SCRIPT = """
import json, sys, torch
import kintera as kt
torch.set_default_dtype(torch.float64)
c1, c2, cp, dt = (float(a) for a in sys.argv[2:6])
kin = kt.Kinetics(kt.KineticsOptions.from_yaml(sys.argv[1]))
stoich = kin.buffer("stoich")
j = [r for r in range(stoich.shape[1]) if (stoich[:, r] != 0).sum() == 3][0]
col = stoich[:, j]
prods = [i for i in range(stoich.shape[0]) if col[i] > 0]
reac = [i for i in range(stoich.shape[0]) if col[i] < 0][0]
conc = torch.zeros(1, stoich.shape[0])
conc[0, prods[0]], conc[0, prods[1]], conc[0, reac] = c1, c2, cp
one = [r for r in range(stoich.shape[1]) if (stoich[:, r] != 0).sum() == 2][0]
conc[0, (stoich[:, one] < 0).nonzero()[0, 0]] = 1.0e-3  # H2O(l,p)
conc[0, (stoich[:, one] > 0).nonzero()[0, 0]] = 1.0e-3  # H2O
T, P = torch.tensor([230.0]), torch.tensor([4.0e5])
rate, rc_ddC, rc_ddT = kin.forward(T, P, conc)
jac = kin.jacobian(T, conc, torch.ones(1), rate, rc_ddC)
delta = kt.evolve_implicit(rate, stoich, jac, dt)
new = conc[0] + delta[0]
print(json.dumps({"rate": rate[0, j].item(), "prod_after": (new[prods[0]] * new[prods[1]]).item(),
                  "cp_after": new[reac].item(), "one_rate": rate[0, one].item().hex(),
                  "one_rc_ddC": [v.hex() for v in rc_ddC[0, :, one].tolist()]}))
"""


def _kinetics(tmp_path, c1, c2, cp, dt):
    card = tmp_path / "nh4sh.yaml"
    card.write_text(textwrap.dedent(CARD))
    res = subprocess.run([sys.executable, "-c", SCRIPT, str(card), str(c1), str(c2), str(cp),
                          str(dt)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_kinetics_rate_is_the_diffusion_extent(tmp_path):
    c1, c2, cp = 0.0669, 0.00209, 1.0e-3
    got = _kinetics(tmp_path, c1, c2, cp, 1.0)
    assert got["rate"] == pytest.approx(KAPPA * cp * _x_ref(c1, c2, _K(T0)), rel=1e-10)


def test_kinetics_implicit_step_does_not_pass_equilibrium(tmp_path):
    # precipitation far above C1 + C2 + x: the regime where the old law overshoots
    got = _kinetics(tmp_path, 0.00669, 0.000209, 1.0, 1.0e6)
    assert got["prod_after"] <= _K(T0) * (1.0 + 1.0e-9)
    assert got["cp_after"] >= 0.0
