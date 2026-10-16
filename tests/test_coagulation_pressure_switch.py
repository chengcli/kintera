"""Pressure-switched rate constant (p-threshold, factor) on coagulation.

Kessler autoconversion dx_c/dt = -k1 x_c, dx_p/dt = +k1 x_c with k1 raised by
a factor where p < p_threshold (strict). Autoconversion conserves moles and
gas, so a closed box keeps P fixed and x_c(t) = x_c0 exp(-k t). The box below
integrates kintera's rate with RK4 against that closed form.
"""

import math

import pytest
import torch

import kintera as kt

torch.set_default_dtype(torch.float64)

R = 8.31446  # kintera constants::Rgas
K1 = 1e-4
PTH = 1e4
FACTORS = {"CH4": 1e2, "H2S": 1e4}

CARD = """
reference-state: {{Tref: 0., Pref: 1.e5}}
species:
  - {{name: dry, composition: {{H: 1.696, He: 0.152}}, cv_R: 1.82}}
  - {{name: CH4, composition: {{C: 1, H: 4}}, cv_R: 3.0}}
  - {{name: H2S, composition: {{H: 2, S: 1}}, cv_R: 3.0}}
  - {{name: "CH4(s)", composition: {{C: 1, H: 4}}, cv_R: 4.0}}
  - {{name: "H2S(s)", composition: {{H: 2, S: 1}}, cv_R: 5.0}}
  - {{name: "CH4(s,p)", composition: {{C: 1, H: 4}}, cv_R: 4.0}}
  - {{name: "H2S(s,p)", composition: {{H: 2, S: 1}}, cv_R: 5.0}}
reactions:
  - equation: CH4(s) => CH4(s,p)
    type: coagulation
    rate-constant: {{A: 1.e-4, b: 0, Ea_R: 0., {ch4}}}
  - equation: H2S(s) => H2S(s,p)
    type: coagulation
    rate-constant: {{A: 1.e-4, b: 0, Ea_R: 0., {h2s}}}
"""

SWITCH = {"ch4": "p-threshold: 1.e4, factor: 1.e2", "h2s": "p-threshold: 1.e4, factor: 1.e4"}


def load(tmp_path, **kw):
    path = tmp_path / "switch.yaml"
    path.write_text(CARD.format(**{**SWITCH, **kw}))
    op = kt.KineticsOptions.from_yaml(str(path))
    return op, kt.Kinetics(op)


@pytest.mark.parametrize("P", [math.nextafter(PTH, 0.), PTH, math.nextafter(PTH, math.inf),
                               PTH * (1. - 1e-9), PTH * (1. + 1e-9)])
def test_box_matches_switched_closed_form(tmp_path, P):
    op, kin = load(tmp_path)
    names = op.species()
    stoich = dict(kin.named_buffers())["stoich"]
    T = 80.
    n = P / (R * T)
    c = torch.zeros(len(names))
    for sp in FACTORS:
        c[names.index(f"{sp}(s)")] = 1e-3 * n
    c0 = c.clone()
    temp, pres = torch.tensor([T]), torch.tensor([P])

    def rhs(c):
        return stoich @ kin.forward(temp, pres, c.unsqueeze(0))[0][0]

    kmax = K1 * (max(FACTORS.values()) if P < PTH else 1.)
    t_end, nstep = 2. / kmax, 400
    h = t_end / nstep
    for i in range(1, nstep + 1):
        k1 = rhs(c)
        k2 = rhs(c + .5 * h * k1)
        k3 = rhs(c + .5 * h * k2)
        k4 = rhs(c + h * k3)
        c = c + h / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        for sp, fac in FACTORS.items():
            k = K1 * (fac if P < PTH else 1.)
            cc0 = c0[names.index(f"{sp}(s)")].item()
            ref = cc0 * math.exp(-k * i * h)
            assert c[names.index(f"{sp}(s)")].item() == pytest.approx(ref, rel=1e-10)
            assert c[names.index(f"{sp}(s,p)")].item() == pytest.approx(cc0 - ref, rel=1e-10)


def test_rate_constant_switches_strictly_below(tmp_path):
    op, kin = load(tmp_path)
    c = torch.ones(1, len(op.species()))
    for P, expect in [(9999.999, [1e-2, 1.]), (1e4, [1e-4, 1e-4]), (2e4, [1e-4, 1e-4])]:
        kf = kin.forward_with_rate_constants(torch.tensor([80.]), torch.tensor([P]), c)[3]
        assert kf[0].tolist() == pytest.approx(expect, rel=1e-15)


def test_default_has_no_switch(tmp_path):
    op, _ = load(tmp_path, ch4="b: 0", h2s="b: 0")
    assert op.coagulation().p_threshold() == []
    assert op.coagulation().p_factor() == []


def test_mixed_switch_fills_neutral_entries(tmp_path):
    op, kin = load(tmp_path, h2s="b: 0")
    assert op.coagulation().p_threshold() == [1e4, 0.]
    assert op.coagulation().p_factor() == [1e2, 1.]
    c = torch.ones(1, len(op.species()))
    kf = kin.forward_with_rate_constants(torch.tensor([80.]), torch.tensor([5e3]), c)[3]
    assert kf[0].tolist() == pytest.approx([1e-2, 1e-4], rel=1e-15)


@pytest.mark.parametrize("ch4,match", [
    ("p-threshold: 1.e4", "given together"),
    ("factor: 1.e2", "given together"),
    ("p-threshold: 0., factor: 1.e2", "finite and > 0"),
    ("p-threshold: -1., factor: 1.e2", "finite and > 0"),
    ("p-threshold: 1.e4, factor: -1.", "finite and >= 0"),
    ("p-threshold: .inf, factor: 1.e2", "finite and > 0"),
])
def test_bad_switch_inputs_raise(tmp_path, ch4, match):
    with pytest.raises(Exception, match=match):
        load(tmp_path, ch4=ch4)
