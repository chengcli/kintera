"""Kessler-form evaporation (form: kessler, k3).

Kessler's evaporation of precipitation, in mole fractions x = c / n of all
species (gas and condensed, n = total molar concentration of the cell):

    dx_v/dt = k3 x_p (x_sat - x_v),   only when x_v < x_sat,

with x_sat = svp/p * x_gas, x_gas = n_gas / n. In a closed cell of fixed
volume and temperature, n is conserved (one condensed mole becomes one vapour
mole) and p = n_gas R T, so x_sat = svp / (R T n) is constant and
x_v + x_p = b is constant. The equation is then a Riccati equation with the
closed-form solution used below as the reference:

    r = (a - x_v) / (b - x_v) = r0 exp(k3 (a - b) t),   a = x_sat,
    x_v = (a - b r) / (1 - r),   x_p = (b - a) / (1 - r).

kintera evaluates kappa c_p (c_sat - c_v)+ with kappa = k3 / n,
n = P/(R T) + the condensed concentrations: the same equation times n.
The box below integrates kintera's rate with RK4, updating P = n_gas R T.
"""

import math

import pytest
import torch

import kintera as kt

torch.set_default_dtype(torch.float64)

R = 8.31446  # kintera constants::Rgas
K3 = 0.03
T3, P3, BETA, GAMMA = 90.67, 11690., 12.87, 0.

CARD = """
reference-state: {{Tref: 0., Pref: 1.e5}}
species:
  - {{name: dry, composition: {{H: 1.696, He: 0.152}}, cv_R: 1.82}}
  - {{name: CH4, composition: {{C: 1, H: 4}}, cv_R: 3.0}}
  - {{name: "CH4(s)", composition: {{C: 1, H: 4}}, cv_R: 4.0, u0_R: -1166.9229}}
  - {{name: "CH4(s,p)", composition: {{C: 1, H: 4}}, cv_R: 4.0, u0_R: -1166.9229}}
reactions:
  - equation: CH4(s) => CH4(s,p)
    type: coagulation
    rate-constant: {{A: 0., b: 0, Ea_R: 0.}}
  - equation: CH4(s,p) => CH4
    type: evaporation
    rate-constant: {{formula: ideal, T3: 90.67, P3: 11690., beta: 12.87,
                    gamma: 0., {extra}}}
"""

KESSLER = "form: kessler, k3: 0.03"


def load(tmp_path, extra=KESSLER):
    path = tmp_path / "kessler.yaml"
    path.write_text(CARD.format(extra=extra))
    op = kt.KineticsOptions.from_yaml(str(path))
    return op, kt.Kinetics(op)


def svp(T):
    t = T / T3
    return P3 * math.exp((1. - 1. / t) * BETA - GAMMA * math.log(t))


def reference(T, n, xv, xp, t):
    """Closed-form x_v, x_p of Kessler's equation at time t."""
    a = svp(T) / (R * T * n)
    b = xv + xp
    if xp == 0. or xv >= a:
        return xv, xp
    r = (a - xv) / (b - xv) * math.exp(K3 * (a - b) * t)
    return (a - b * r) / (1. - r), (b - a) / (1. - r)


def box(kin, op, T, n, xv, xc, xp, t_end, nstep):
    """RK4 on kintera's rate in a closed cell; P = (c_dry + c_v) R T."""
    names = op.species()
    c = torch.zeros(len(names))
    c[names.index("CH4")] = xv * n
    c[names.index("CH4(s)")] = xc * n
    c[names.index("CH4(s,p)")] = xp * n
    cdry = (1. - xv - xc - xp) * n
    iv = names.index("CH4")
    stoich = dict(kin.named_buffers())["stoich"]
    temp = torch.tensor([T])

    def rhs(c):
        pres = (cdry + c[iv]) * R * temp
        rate = kin.forward(temp, pres, c.unsqueeze(0))[0][0]
        return stoich @ rate

    h = t_end / nstep
    out = [c / n]
    for _ in range(nstep):
        k1 = rhs(c)
        k2 = rhs(c + .5 * h * k1)
        k3 = rhs(c + .5 * h * k2)
        k4 = rhs(c + h * k3)
        c = c + h / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        out.append(c / n)
    return torch.stack(out), names


# (T [K], p0 [Pa], x_v, x_c, x_p): subsaturated b < a and b > a,
# supersaturated (rate exactly 0) and no precipitation (rate exactly 0)
STATES = [
    (80., 1.2e5, 0.010, 0.002, 0.005),
    (100., 2.0e5, 0.030, 0.000, 0.020),
    (80., 1.2e5, 0.015, 0.001, 0.010),
    (70., 1.0e5, 0.030, 0.002, 0.010),
    (90., 1.5e5, 0.010, 0.003, 0.000),
]


@pytest.mark.parametrize("T,p0,xv,xc,xp", STATES)
def test_box_matches_kessler_closed_form(tmp_path, T, p0, xv, xc, xp):
    op, kin = load(tmp_path)
    n = p0 / (R * T) / (1. - xc - xp)
    a = svp(T) / (R * T * n)
    scale = max(xp, abs(a - xv - xp), 1e-3)
    t_end = 2. / (K3 * scale)
    nstep = 400
    traj, names = box(kin, op, T, n, xv, xc, xp, t_end, nstep)
    for i, x in enumerate(traj):
        rv, rp = reference(T, n, xv, xp, i * t_end / nstep)
        assert x[names.index("CH4")].item() == pytest.approx(rv, rel=1e-10)
        assert x[names.index("CH4(s,p)")].item() == pytest.approx(rp, rel=1e-10, abs=1e-300)
        assert x[names.index("CH4(s)")].item() == xc
    if xp == 0. or xv >= a:
        assert torch.equal(traj[-1], traj[0])


def test_rate_is_k3_cp_deficit_over_n(tmp_path):
    op, kin = load(tmp_path)
    names = op.species()
    T, P = 85., 1.3e5
    c = torch.zeros(1, len(names))
    c[0, names.index("CH4")] = 3.
    c[0, names.index("CH4(s)")] = 0.4
    c[0, names.index("CH4(s,p)")] = 0.7
    rate = kin.forward(torch.tensor([T]), torch.tensor([P]), c)[0][0, 1].item()
    n = P / (R * T) + 0.4 + 0.7
    expected = K3 / n * 0.7 * (svp(T) / (R * T) - 3.)
    assert rate == pytest.approx(expected, rel=1e-13)


def test_jacobian_includes_the_total_concentration(tmp_path):
    op, kin = load(tmp_path)
    names = op.species()
    T, P = torch.tensor([85.]), torch.tensor([1.3e5])
    c = torch.zeros(1, len(names))
    c[0, names.index("CH4")] = 3.
    c[0, names.index("CH4(s)")] = 0.4
    c[0, names.index("CH4(s,p)")] = 0.7
    rate, rc_ddC, rc_ddT, kf = kin.forward_with_rate_constants(T, P, c)
    jac = kin.jacobian(T, c, torch.ones(1), rate, rc_ddC, rc_ddT, kf)
    for i in range(len(names)):
        step = 1e-6 * c[0, i].item()
        plus, minus = c.clone(), c.clone()
        plus[0, i] += step
        minus[0, i] -= step
        fd = (kin.forward(T, P, plus)[0] - kin.forward(T, P, minus)[0]) / (2 * step)
        assert jac[0, 1, i].item() == pytest.approx(fd[0, 1].item(), rel=1e-7)


def test_default_form_is_diffusion(tmp_path):
    op, _ = load(tmp_path, "diff_c: 2.e-5, vm: 30.8e-6, diameter: 1.e-5")
    assert op.evaporation().form() == ["diffusion"]
    assert op.evaporation().k3() == [0.]


@pytest.mark.parametrize("extra,match", [
    ("form: kesler, k3: 0.03", "Unknown evaporation form"),
    ("form: kessler", "'k3' required"),
    ("k3: 0.03", "requires form: kessler"),
    ("form: kessler, k3: -1.", "finite and >= 0"),
    ("form: kessler, k3: .nan", "finite and >= 0"),
    ("form: kessler, k3: 0.03, diff_c: 2.e-5", "'diff_c' is not used"),
    ("form: kessler, k3: 0.03, diameter: 1.e-5", "'diameter' is not used"),
])
def test_bad_kessler_inputs_raise(tmp_path, extra, match):
    with pytest.raises(Exception, match=match):
        load(tmp_path, extra)


def test_direct_call_requires_gas_mask(tmp_path):
    op, _ = load(tmp_path)
    ev = kt.Evaporation(op.evaporation())
    stoich = torch.tensor([[1.], [0.], [-1.]])
    c = torch.tensor([[1., 0., 1.]])
    with pytest.raises(RuntimeError, match="gas_mask"):
        ev.forward(torch.tensor([80.]), torch.tensor([1e5]), c, {"stoich": stoich})


def test_kessler_rejects_two_products():
    kt.set_species_names(["A", "B", "C(s)"])
    kt.set_species_weights([.016, .002, .016])
    op = kt.EvaporationOptions().reactions([kt.Reaction("C(s) => A + B")])
    op.logsvp(["h2o_ideal"]).diff_c([2e-5]).diff_T([0.]).diff_P([0.])
    op.vm([1e-5]).diameter([1e-5]).minT([0.]).maxT([1e4])
    op.form(["kessler"]).k3([K3])
    with pytest.raises(RuntimeError, match="one gas product"):
        kt.Evaporation(op)
