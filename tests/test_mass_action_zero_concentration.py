"""#147: an absent reactant gives an exact zero mass-action rate.

``KineticsImpl::forward`` used to floor concentrations at 1e-300 (1e-30 in
float32) before the mass-action product, so a reaction whose reactant is
absent still ran at ``k_f * 1e-300``. ``evolve_implicit`` then removed that
amount and the species ended below zero: about ``-A * 1e-300 * dt`` for a
first-order reaction. The floor was also how ``jacobian`` recovered ``k_f``
at zero concentration, as ``(k_f * 1e-300) / 1e-300``.

Now the rate uses the real, non-negative concentration, and ``jacobian``
forms d(rate)/dC_j = k nu_j C_j^(nu_j - 1) prod_{i != j} C_i^nu_i from the
rate constants (``rate_constants`` from ``forward_with_rate_constants``, or
``rate / prod C^nu`` where that product is positive) without dividing by a
concentration, so the derivative is exact at zero as well.
"""

import pytest
import torch

import kintera as kt

torch.set_default_dtype(torch.float64)

DEVICES = ["cpu"] + ["cuda"] * torch.cuda.is_available()
R = 8.31446  # kintera constants::Rgas
A_COAG = 1e-3
T_CLOUD, P_CLOUD = 90., 1e5

# The issue's chain without the thermo nucleation step: coagulation of the
# cloud into precipitation, then Kessler evaporation of the precipitation.
COAG = """
reference-state: {Tref: 0., Pref: 1.e5}
species:
  - {name: dry, composition: {H: 1.696, He: 0.152}, cv_R: 1.82}
  - {name: CH4, composition: {C: 1, H: 4}, cv_R: 3.0}
  - {name: "CH4(s)", composition: {C: 1, H: 4}, cv_R: 4.0, u0_R: -1166.9229}
  - {name: "CH4(s,p)", composition: {C: 1, H: 4}, cv_R: 4.0, u0_R: -1166.9229}
reactions:
  - equation: CH4(s) => CH4(s,p)
    type: coagulation
    rate-constant: {A: 1.e-3, b: 0, Ea_R: 0.}
"""

CHAIN = COAG + """  - equation: CH4(s,p) => CH4
    type: evaporation
    rate-constant: {formula: ideal, T3: 90.67, P3: 11690., beta: 12.87,
                    gamma: 0., form: kessler, k3: 0.03}
"""

# Second order in one reactant and first order in two, to exercise nu_j > 1
# and the product over the other reactants.
ORDERS = """
reference-state: {Tref: 300., Pref: 1.e5}
species:
  - {name: M, composition: {N: 2}, cv_R: 2.5}
  - {name: A, composition: {H: 1}, cv_R: 1.5}
  - {name: B, composition: {H: 2}, cv_R: 2.5}
  - {name: C, composition: {H: 3}, cv_R: 3.0}
reactions:
  - {equation: 2 A => B, type: arrhenius, rate-constant: {A: 1.e-12, b: 0., Ea_R: 0.}}
  - {equation: A + B => C, type: arrhenius, rate-constant: {A: 3.e-11, b: 0., Ea_R: 0.}}
"""

# A reversible reaction: the reverse row's rate constant is k_f / Kc.
OZONE = """reference-state: {Tref: 300., Pref: 1.e5}
species:
  - {name: N2, composition: {N: 2}, cv_R: 2.5}
  - {name: O2, composition: {O: 2}, cv_R: 2.5}
  - {name: O, composition: {O: 1}, cv_R: 1.5}
  - {name: O3, composition: {O: 3}, cv_R: 3.0}
reactions:
  - {equation: O + O2 <=> O3, type: arrhenius, rate-constant: {A: 1.7e-14, b: -2.4, Ea_R: 0.}}
"""


def load(tmp_path, card, dev="cpu"):
    path = tmp_path / "card.yaml"
    path.write_text(card)
    op = kt.KineticsOptions.from_yaml(str(path))
    kin = kt.Kinetics(op)
    kin.to(torch.device(dev))
    return op, kin


def conc(op, cells, dev="cpu", dtype=torch.float64):
    names = op.species()
    return torch.tensor([[cell.get(n, 0.) for n in names] for cell in cells],
                        device=dev, dtype=dtype)


def state(ncell, dev, dtype=torch.float64, T=T_CLOUD, P=P_CLOUD):
    return (torch.full((ncell,), T, device=dev, dtype=dtype),
            torch.full((ncell,), P, device=dev, dtype=dtype),
            torch.ones(ncell, device=dev, dtype=dtype))


def cloud_cells(op, dev, dtype=torch.float64):
    n = P_CLOUD / (R * T_CLOUD)
    # cell 0: no condensate (the issue); cell 1: cloud present
    return conc(op, [{"dry": n, "CH4": 1e-3 * n},
                     {"dry": n, "CH4": 1e-3 * n, "CH4(s)": 1e-4 * n}], dev, dtype)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_absent_reactant_rate_is_exactly_zero(tmp_path, dev, dtype):
    op, kin = load(tmp_path, CHAIN, dev)
    kin.to(dtype)
    c = cloud_cells(op, dev, dtype)
    T, P, _ = state(2, dev, dtype)
    rate = kin.forward(T, P, c)[0]
    assert rate[0].tolist() == [0., 0.]
    assert rate[1, 0].item() > 0.


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dt", [542., 271.])
def test_absent_condensate_stays_zero_after_implicit_step(tmp_path, dev, dt):
    op, kin = load(tmp_path, CHAIN, dev)
    names = op.species()
    s, p = names.index("CH4(s)"), names.index("CH4(s,p)")
    c = cloud_cells(op, dev)
    T, P, cvol = state(2, dev)
    rate, dc, _, kf = kin.forward_with_rate_constants(T, P, c)
    jac = kin.jacobian(T, c, cvol, rate, dc, None, kf)
    c1 = c + kt.evolve_implicit(rate, kin.buffer("stoich"), jac, dt)
    # was about -A 1e-300 dt and -1e-300 mol/m^3
    assert c1[0, s].item() == 0. and c1[0, p].item() == 0.
    assert c1[0].tolist() == c[0].tolist()
    # the cloudy cell still coagulates: backward Euler of dc/dt = -A c
    assert c1[1, s].item() == pytest.approx(c[1, s].item() / (1. + A_COAG * dt), rel=1e-12)
    assert c1[1, p].item() > 0.


@pytest.mark.parametrize("dev", DEVICES)
def test_plain_forward_and_jacobian_keep_zero(tmp_path, dev):
    # The forward() / jacobian() pair without rate constants, on the card the
    # pair supports (no evaporation at zero solid).
    op, kin = load(tmp_path, COAG, dev)
    s = op.species().index("CH4(s)")
    c = cloud_cells(op, dev)
    T, P, cvol = state(2, dev)
    rate, dc, _ = kin.forward(T, P, c)
    jac = kin.jacobian(T, c, cvol, rate, dc)
    c1 = c + kt.evolve_implicit(rate, kin.buffer("stoich"), jac, 542.)
    assert c1[0].tolist() == c[0].tolist()
    assert c1[1, s].item() == pytest.approx(c[1, s].item() / (1. + A_COAG * 542.), rel=1e-12)


@pytest.mark.parametrize("dev", DEVICES)
def test_jacobian_at_zero_concentration_from_rate_constants(tmp_path, dev):
    op, kin = load(tmp_path, ORDERS, dev)
    names = op.species()
    a, b = names.index("A"), names.index("B")
    T, P, cvol = state(1, dev, T=300.)
    c = conc(op, [{"M": 40., "B": .5}], dev)  # A absent
    rate, dc, _, kf = kin.forward_with_rate_constants(T, P, c)
    assert rate.tolist() == [[0., 0.]]
    jac = kin.jacobian(T, c, cvol, rate, dc, None, kf)
    # 2 A => B: d(k A^2)/dA = 2 k A = 0 at A = 0
    assert jac[0, 0, a].item() == 0.
    # A + B => C: d(k A B)/dA = k B, d(k A B)/dB = k A = 0
    assert jac[0, 1, a].item() == pytest.approx(kf[0, 1].item() * .5, rel=1e-14)
    assert jac[0, 1, b].item() == 0.
    # one-sided finite difference from the right
    h = 1e-6
    ch = c.clone()
    ch[0, a] = h
    fd = (kin.forward(T, P, ch)[0] - rate)[0] / h
    assert fd[1].item() == pytest.approx(jac[0, 1, a].item(), rel=1e-12)
    assert fd[0].item() == pytest.approx(kf[0, 0].item() * h, rel=1e-12)


@pytest.mark.parametrize("dev", DEVICES)
def test_jacobian_with_and_without_rate_constants(tmp_path, dev):
    op, kin = load(tmp_path, ORDERS, dev)
    names = op.species()
    a, b = names.index("A"), names.index("B")
    T, P, cvol = state(2, dev, T=300.)
    c = conc(op, [{"M": 40., "A": .3, "B": .5}, {"M": 40., "B": .5}], dev)
    rate, dc, _, kf = kin.forward_with_rate_constants(T, P, c)
    with_k = kin.jacobian(T, c, cvol, rate, dc, None, kf)
    without = kin.jacobian(T, c, cvol, rate, dc)
    # closed form where every reactant is present
    k0, k1 = kf[0, 0].item(), kf[0, 1].item()
    assert with_k[0, 0, a].item() == pytest.approx(2. * k0 * .3, rel=1e-13)
    assert with_k[0, 1, a].item() == pytest.approx(k1 * .5, rel=1e-13)
    assert with_k[0, 1, b].item() == pytest.approx(k1 * .3, rel=1e-13)
    torch.testing.assert_close(without[0], with_k[0], rtol=1e-12, atol=0.)
    # without the rate constants a zero rate carries no k: that entry is 0
    assert with_k[1, 1, a].item() == pytest.approx(k1 * .5, rel=1e-14)
    assert without[1, 1, a].item() == 0.
    assert without[1, 1, b].item() == 0.


@pytest.mark.parametrize("dev", DEVICES)
def test_reverse_reaction_at_zero_product(tmp_path, dev):
    op, kin = load(tmp_path, OZONE, dev)
    names = op.species()
    o, o3 = names.index("O"), names.index("O3")
    T, P, cvol = state(2, dev, T=300.)
    c = conc(op, [{"N2": 1., "O2": .2, "O": 1e-8},
                  {"N2": 1., "O2": .2, "O": 1e-8, "O3": 1e-6}], dev)  # O3 absent in cell 0
    rate, dc, _, kf = kin.forward_with_rate_constants(T, P, c)
    assert rate[0, 1].item() == 0.  # O3 => O + O2
    assert rate[0, 0].item() > 0.
    with_k = kin.jacobian(T, c, cvol, rate, dc, None, kf)
    without = kin.jacobian(T, c, cvol, rate, dc)
    # k_r = k_f / Kc, read off the reverse rate where O3 is present
    k_r = rate[1, 1].item() / 1e-6
    assert with_k[0, 1, o3].item() == pytest.approx(k_r, rel=1e-10)
    assert with_k[1, 1, o3].item() == pytest.approx(k_r, rel=1e-10)
    assert without[1, 1, o3].item() == pytest.approx(k_r, rel=1e-10)
    assert without[0, 1, o3].item() == 0.
    # the forward row is first order in O: k_f [O2] in both cells
    for cell in range(2):
        assert with_k[cell, 0, o].item() == pytest.approx(kf[cell, 0].item() * .2, rel=1e-13)
    torch.testing.assert_close(without[1], with_k[1], rtol=1e-12, atol=0.)


@pytest.mark.parametrize("dev", DEVICES)
def test_negative_concentration_is_treated_as_absent(tmp_path, dev):
    op, kin = load(tmp_path, COAG, dev)
    s = op.species().index("CH4(s)")
    c = cloud_cells(op, dev)[:1]
    c[0, s] = -1e-9
    T, P, cvol = state(1, dev)
    rate, dc, _, kf = kin.forward_with_rate_constants(T, P, c)
    assert rate.tolist() == [[0.]]
    jac = kin.jacobian(T, c, cvol, rate, dc, None, kf)
    assert jac[0, 0, s].item() == 0.
    c1 = c + kt.evolve_implicit(rate, kin.buffer("stoich"), jac, 542.)
    assert c1.tolist() == c.tolist()


# Fractional orders: d(k C^nu)/dC = nu k C^(nu - 1) is unbounded at C = 0 for
# nu < 1, so that factor is taken at the 1e-300 floor, the value main gave.
FRACTIONAL = """reference-state: {Tref: 300., Pref: 1.e5}
species:
  - {name: A, composition: {H: 1}, cv_R: 1.5}
  - {name: B, composition: {H: 2}, cv_R: 2.5}
reactions:
  - {equation: 0.5 A => B, type: arrhenius, rate-constant: {A: 1.e-3, b: 0., Ea_R: 0.}}
  - {equation: 1.5 A => B, type: arrhenius, rate-constant: {A: 2.e-3, b: 0., Ea_R: 0.}}
"""


@pytest.mark.parametrize("dev", DEVICES)
def test_fractional_orders_stay_finite(tmp_path, dev):
    op, kin = load(tmp_path, FRACTIONAL, dev)
    a = op.species().index("A")
    T, P, cvol = state(2, dev, T=300.)
    c = conc(op, [{}, {"A": 1e-3}], dev)
    rate, dc, _, kf = kin.forward_with_rate_constants(T, P, c)
    assert rate[0].tolist() == [0., 0.]
    with_k = kin.jacobian(T, c, cvol, rate, dc, None, kf)
    without = kin.jacobian(T, c, cvol, rate, dc)
    assert torch.isfinite(with_k).all() and torch.isfinite(without).all()
    k0, k1 = kf[0, 0].item(), kf[0, 1].item()
    # order 0.5 at zero: the floored value; order 1.5 at zero: exactly 0
    assert with_k[0, 0, a].item() == pytest.approx(.5 * k0 * 1e-300 ** -.5, rel=1e-12)
    assert with_k[0, 1, a].item() == 0.
    assert without[0, :, a].tolist() == [0., 0.]
    # closed form where A is present, with and without rate constants
    assert with_k[1, 0, a].item() == pytest.approx(.5 * k0 * 1e-3 ** -.5, rel=1e-12)
    assert with_k[1, 1, a].item() == pytest.approx(1.5 * k1 * 1e-3 ** .5, rel=1e-12)
    torch.testing.assert_close(without[1], with_k[1], rtol=1e-12, atol=0.)


# Backward through the Jacobian. pow(0, e) and its backward are finite for
# e = 0 and e >= 1, so an integer order keeps its gradient at zero (the
# Jacobian is a polynomial there). For 0 < e < 1 the backward is infinite and
# 0 * inf is NaN, so a fractional order takes a constant at zero instead.
BACKWARD = """reference-state: {Tref: 300., Pref: 1.e5}
species:
  - {name: A, composition: {H: 1}, cv_R: 1.5}
  - {name: B, composition: {H: 2}, cv_R: 2.5}
  - {name: D, composition: {H: 3}, cv_R: 2.5}
reactions:
  - {equation: 0.5 A => B, type: arrhenius, rate-constant: {A: 1.e-3, b: 0., Ea_R: 0.}}
  - {equation: 1.5 A => B, type: arrhenius, rate-constant: {A: 2.e-3, b: 0., Ea_R: 0.}}
  - {equation: 2 A => B, type: arrhenius, rate-constant: {A: 3.e-3, b: 0., Ea_R: 0.}}
  - {equation: A + D => B, type: arrhenius, rate-constant: {A: 4.e-3, b: 0., Ea_R: 0.}}
"""


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("use_k", [True, False])
def test_jacobian_backward_is_finite_at_zero(tmp_path, dev, use_k):
    op, kin = load(tmp_path, BACKWARD, dev)
    a = op.species().index("A")
    T, P, cvol = state(2, dev, T=300.)
    c = conc(op, [{"D": 1e-3}, {"A": 1e-3, "D": 1e-3}], dev).requires_grad_(True)
    rate, dc, _, kf = kin.forward_with_rate_constants(T, P, c.detach())
    jac = kin.jacobian(T, c, cvol, rate, dc, None, kf if use_k else None)
    assert torch.isfinite(jac).all()
    jac.sum().backward()
    assert torch.isfinite(c.grad).all()
    # d/dA of the Jacobian sum at A = 0: the fractional orders contribute 0,
    # 2 A => B contributes 2 k, A + D => B contributes k (from d(k A)/dA)
    if use_k:
        expected = 2. * kf[0, 2].item() + kf[0, 3].item()
        assert c.grad[0, a].item() == pytest.approx(expected, rel=1e-12)
    else:
        # without rate constants a zero rate carries no k, so nothing to vary
        assert c.grad[0, a].item() == 0.


# The two integer-order cases from the review of #151, with k = 1
INTEGER = """reference-state: {Tref: 300., Pref: 1.e5}
species:
  - {name: A, composition: {H: 1}, cv_R: 1.5}
  - {name: B, composition: {H: 2}, cv_R: 2.5}
  - {name: C, composition: {H: 3}, cv_R: 2.5}
reactions:
  - {equation: A + A + B => C, type: arrhenius, rate-constant: {A: 1.e-3, b: 0., Ea_R: 0.}}
  - {equation: A + B => C, type: arrhenius, rate-constant: {A: 1.e-3, b: 0., Ea_R: 0.}}
"""


@pytest.mark.parametrize("dev", DEVICES)
def test_integer_order_gradients_at_zero(tmp_path, dev):
    op, kin = load(tmp_path, INTEGER, dev)
    T, P, cvol = state(1, dev, T=300.)
    c = conc(op, [{"B": 2.}], dev)
    extra = {"kf_override": torch.tensor([[1., 1.]], device=dev)}
    rate, dc, _, kf = kin.forward_with_rate_constants(T, P, c, extra)
    # 2 A + B: J = [2 k A B, k A^2, 0], gradient of its sum = [2 k B + 2 k A, 2 k A, 0]
    # A + B:   J = [k B, k A, 0],       gradient of its sum = [k, k, 0]
    for row, expected in [(0, [4., 0., 0.]), (1, [1., 1., 0.])]:
        cc = c.clone().requires_grad_(True)
        jac = kin.jacobian(T, cc, cvol, rate, dc, None, kf)
        jac[0, row].sum().backward()
        assert cc.grad[0].tolist() == pytest.approx(expected, rel=1e-12, abs=0.)


# A large rate constant with a small concentration product: k nu_j C_j^(nu_j - 1)
# alone overflows while the derivative is representable, so k is multiplied last.
OVERFLOW = """reference-state: {Tref: 300., Pref: 1.e5}
species:
  - {name: A, composition: {H: 1}, cv_R: 1.5}
  - {name: B, composition: {H: 2}, cv_R: 2.5}
  - {name: D, composition: {H: 3}, cv_R: 2.5}
reactions:
  - {equation: 2 A + B => D, type: arrhenius, rate-constant: {A: 1.e-3, b: 0., Ea_R: 0.}}
"""


@pytest.mark.parametrize("dev", DEVICES)
def test_large_rate_constant_does_not_overflow(tmp_path, dev):
    op, kin = load(tmp_path, OVERFLOW, dev)
    names = op.species()
    a, b = names.index("A"), names.index("B")
    T, P, cvol = state(1, dev, T=300.)
    c = conc(op, [{"A": 1., "B": 5e-9}], dev)
    extra = {"kf_override": torch.tensor([[1e308]], device=dev)}  # k 2 [A] = 2e308 overflows
    rate, dc, _, kf = kin.forward_with_rate_constants(T, P, c, extra)
    assert rate[0, 0].item() == pytest.approx(5e299, rel=1e-12)  # k [A]^2 [B]
    with_k = kin.jacobian(T, c, cvol, rate, dc, None, kf)
    without = kin.jacobian(T, c, cvol, rate, dc)
    for jac in (with_k, without):
        assert jac[0, 0, a].item() == pytest.approx(1e300, rel=1e-12)  # 2 k [A] [B]
        assert jac[0, 0, b].item() == pytest.approx(1e308, rel=1e-12)  # k [A]^2
