"""General diffusion evaporation: surface equilibrium and bulk stoichiometry."""

import math

import pytest
import torch
import yaml

import kintera as kt

torch.set_default_dtype(torch.float64)

R = 8.31446
KAPPA = 12 * 2e-5 * 43.7e-6 / 1e-8
NAMES = ["Na", "H2S", "H2", "Na2S(s)"]
NU = [2., 1., -1., -1.]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))]
CARD = """
reference-state: {Tref: 300., Pref: 1.e5}
species:
  - {name: Na, phase: gas, composition: {Na: 1}, cv_R: 1.5}
  - {name: H2S, phase: gas, composition: {H: 2, S: 1}, cv_R: 2.5}
  - {name: H2, phase: gas, composition: {H: 2}, cv_R: 2.5}
  - {name: Na2S(s), phase: solid, composition: {Na: 2, S: 1}, cv_R: 9.}
reactions:
  - equation: Na2S(s) + H2 => 2 Na + H2S
    type: evaporation
    rate-constant: {formula: na_h2s_visscher, diff_c: 2.e-5,
                    diff_T: 0., diff_P: 0., vm: 43.7e-6, diameter: 1.e-4}
"""


def reference(c, temperature=1200., nu=(2., 1., -1.)):
    """Independent scalar bisection, including depletion and inactive limits."""
    target = (22.48 - 27778. / temperature) * math.log(10) - sum(nu) * math.log(R * temperature)

    def residual(x):
        values = [a + b * x for a, b in zip(c, nu)]
        if any(v <= 0 and b < 0 for v, b in zip(values, nu)):
            return math.inf
        if any(v <= 0 and b > 0 for v, b in zip(values, nu)):
            return -math.inf
        return sum(b * math.log(v) for b, v in zip(nu, values)) - target

    if any(a <= 0 and b < 0 for a, b in zip(c, nu)) or residual(0) >= 0:
        return 0.
    lo = 0.
    hi = min((a / -b for a, b in zip(c, nu) if b < 0), default=1.)
    while residual(hi) < 0:
        hi *= 2
    for _ in range(160):
        mid = (lo + hi) / 2
        if residual(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def module(equations=None):
    kt.set_species_names(NAMES)
    kt.set_species_weights([.023, .034, .002, .078])
    equations = equations or ["Na2S(s) + H2 => 2 Na + H2S"]
    n = len(equations)
    op = kt.EvaporationOptions().reactions([kt.Reaction(eq) for eq in equations])
    op.logsvp(["na_h2s_visscher"] * n)
    op.diff_c([2e-5] * n).diff_T([0.] * n).diff_P([0.] * n)
    op.vm([43.7e-6] * n).diameter([1e-4] * n)
    op.minT([0.] * n).maxT([1e4] * n)
    return kt.Evaporation(op)


def direct(mod, c, t=None, stoich=None, mask=True):
    t = torch.full(c.shape[:-1], 1200., dtype=c.dtype, device=c.device) if t is None else t
    s = torch.tensor(NU, dtype=c.dtype, device=c.device).unsqueeze(-1) if stoich is None else stoich
    other = {"stoich": s}
    if mask:
        other["gas_mask"] = torch.tensor([True, True, True, False], device=c.device)
    return mod.forward(t, torch.full_like(t, 1e5), c, other)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("state", [[0., 0., 1., .01], [1e-5, 2e-5, .8, .01], [1., 1., 1., .01], [0., 0., 0., .01], [0., 0., 1e-12, .01]])
def test_general_surface_extent(device, dtype, state):
    mod = module()
    mod.to(torch.device(device))
    mod.to(dtype)
    c = torch.tensor([state], dtype=dtype, device=device, requires_grad=True)
    result = direct(mod, c)
    assert result.item() / KAPPA == pytest.approx(reference(state[:3]), rel=3e-5 if dtype == torch.float32 else 2e-10, abs=1e-24)
    result.sum().backward()
    assert torch.isfinite(c.grad).all()


def test_direct_temperature_and_concentration_gradients():
    mod = module()
    c = torch.tensor([[1e-5, 2e-5, .8, .01]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([1200.], dtype=torch.float64, requires_grad=True)
    direct(mod, c, t).sum().backward()
    for i, step in enumerate([1e-8, 1e-8, 1e-5, 1e-5]):
        plus, minus = c.detach().clone(), c.detach().clone()
        plus[0, i] += step
        minus[0, i] -= step
        fd = ((direct(mod, plus) - direct(mod, minus)) / (2 * step)).item()
        assert c.grad[0, i].item() == pytest.approx(fd, rel=2e-6, abs=1e-12)
    fd_t = ((direct(mod, c.detach(), t.detach() + .001) - direct(mod, c.detach(), t.detach() - .001)) / .002).item()
    assert t.grad.item() == pytest.approx(fd_t, rel=2e-6)


def test_multiple_general_columns_and_expanded_concentrations():
    mod = module(["Na2S(s) + H2 => 2 Na + H2S", "Na2S(s) => 3 Na + 2 H2S"])
    c = torch.tensor([[1e-5, 2e-5, .8, .01], [0., 0., .2, .03]], dtype=torch.float64)
    s = torch.tensor([NU, [3., 2., 0., -1.]], dtype=torch.float64).T
    raw = direct(mod, c, stoich=s)
    expanded = mod.forward(torch.full((2,), 1200.), torch.full((2,), 1e5), c.unsqueeze(-1).expand(2, 4, 2), {"stoich": s, "gas_mask": torch.tensor([True, True, True, False])})
    torch.testing.assert_close(raw, expanded)
    for i in range(2):
        assert raw[i, 1].item() / KAPPA == pytest.approx(reference(c[i, :3].tolist(), nu=(3., 2., 0.)), rel=2e-10)


def test_mixed_direct_call_requires_gas_mask():
    with pytest.raises(RuntimeError, match="gas_mask"):
        direct(module(), torch.ones(1, 4), mask=False)


def load(tmp_path, card=CARD):
    path = tmp_path / "general.yaml"
    path.write_text(card)
    op = kt.KineticsOptions.from_yaml(str(path))
    return op, kt.Kinetics(op)


def concentrations(op, solid=.01, h2=.8):
    values = dict(zip(NAMES, [1e-5, 2e-5, h2, solid]))
    return torch.tensor([[values[name] for name in op.species()]], dtype=torch.float64)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("solid,h2", [(.01, .8), (0., .8), (.01, 0.)])
def test_kinetics_rate_and_jacobian(tmp_path, solid, h2, device):
    op, kin = load(tmp_path)
    kin.to(torch.device(device))
    c = concentrations(op, solid, h2).to(device)
    t, p = torch.tensor([1200.], device=device), torch.tensor([1e5], device=device)
    rate, dc, _, kf = kin.forward_with_rate_constants(t, p, c)
    jac = kin.jacobian(t, c, torch.ones_like(t), rate, dc, rate_constants=kf).clone()
    assert rate.item() == pytest.approx(KAPPA * solid * reference([1e-5, 2e-5, h2]), rel=2e-10, abs=1e-300)
    assert torch.isfinite(jac).all()
    # Inactive missing-H2 branch intentionally uses zero derivatives.
    if h2 == 0:
        assert torch.count_nonzero(jac) == 0
        return
    for i, name in enumerate(op.species()):
        step = 1e-8 if name in ("Na", "H2S") else 1e-6
        plus, minus = c.clone(), c.clone()
        plus[0, i] += step
        if c[0, i] == 0:
            fd = (kin.forward(t, p, plus)[0] - rate) / step
        else:
            minus[0, i] -= step
            fd = (kin.forward(t, p, plus)[0] - kin.forward(t, p, minus)[0]) / (2 * step)
        assert jac[0, 0, i].item() == pytest.approx(fd.item(), rel=2e-6, abs=1e-12)


@pytest.mark.parametrize("change,match", [
    ("missing", "explicit phase"), ("solid_product", "products must be gases"),
    ("gas_only", "one condensed reactant"), ("two_solids", "coefficient 1|one condensed reactant"),
])
def test_invalid_phase_declarations(tmp_path, change, match):
    data = yaml.safe_load(CARD)
    if change == "missing":
        del data["species"][2]["phase"]
    elif change == "solid_product":
        data["species"][0]["phase"] = "solid"
    elif change == "gas_only":
        data["species"][3]["phase"] = "gas"
    else:
        data["species"][2]["phase"] = "solid"
    with pytest.raises(RuntimeError, match=match):
        load(tmp_path, yaml.safe_dump(data))


def test_parsed_options_survive_registry_reload(tmp_path):
    op, kin = load(tmp_path)
    c, t, p = concentrations(op), torch.tensor([1200.]), torch.tensor([1e5])
    expected = kin.forward(t, p, c)[0]
    kt.ThermoOptions.from_yaml("tests/earth.yaml")
    recreated = kt.Kinetics(op)
    torch.testing.assert_close(recreated.forward(t, p, c)[0], expected)


def test_implicit_evaporation_conserves_atoms(tmp_path):
    op, kin = load(tmp_path)
    c = concentrations(op)
    t, p = torch.tensor([1200.]), torch.tensor([1e5])
    elements = {"Na": [1., 0., 0.], "H2S": [0., 2., 1.], "H2": [0., 2., 0.], "Na2S(s)": [2., 0., 1.]}
    atoms = torch.tensor([elements[name] for name in op.species()], dtype=c.dtype)
    initial = c @ atoms
    solid_id = op.species().index("Na2S(s)")
    for _ in range(20):
        rate, dc, _ = kin.forward(t, p, c)
        jac = kin.jacobian(t, c, torch.ones(1), rate, dc)
        c = c + kt.evolve_implicit(rate, kin.buffer("stoich"), jac, 1.)
        assert (c >= 0).all()
        torch.testing.assert_close(c @ atoms, initial, rtol=1e-12, atol=1e-15)
    assert c[0, solid_id] < .01


def test_mixed_arrhenius_and_evaporation_keep_distinct_orders(tmp_path):
    data = yaml.safe_load(CARD)
    data["species"].append({"name": "Na2", "phase": "gas", "composition": {"Na": 2}, "cv_R": 2.5})
    data["reactions"].append({"equation": "2 Na => Na2", "type": "arrhenius", "rate-constant": {"A": .3, "b": 0., "Ea_R": 0.}})
    op, kin = load(tmp_path, yaml.safe_dump(data))
    values = dict(zip(NAMES + ["Na2"], [1e-5, 2e-5, .8, .01, 0.]))
    c = torch.tensor([[values[name] for name in op.species()]])
    t, p = torch.tensor([1200.]), torch.tensor([1e5])
    rate, dc, _ = kin.forward(t, p, c)
    jac = kin.jacobian(t, c, torch.ones(1), rate, dc)
    stoich = kin.buffer("stoich")
    sodium = op.species().index("Na")
    ordinary = (stoich[sodium] < 0).nonzero().item()
    evaporation = 1 - ordinary
    # YAML bimolecular A uses cm^3 molecule^-1 s^-1.
    conversion = 6.02214076e23 * 1e-6
    assert rate[0, ordinary].item() == pytest.approx(.3e-10 * conversion)
    assert jac[0, ordinary, sodium].item() == pytest.approx(.6e-5 * conversion)
    assert rate[0, evaporation].item() == pytest.approx(KAPPA * .01 * reference([1e-5, 2e-5, .8]), rel=2e-10)


def test_saturated_gas_is_inactive():
    mod = module()
    x = reference([0., 0., 1.])
    c = torch.tensor([[2 * x * (1 + 1e-12), x, 1 - x, .01]], requires_grad=True)
    rate = direct(mod, c)
    rate.sum().backward()
    assert rate.item() == 0.
    assert torch.count_nonzero(c.grad) == 0


@pytest.mark.parametrize("bad_mask", [torch.ones(4), torch.ones(3, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool)])
def test_invalid_direct_gas_masks(bad_mask):
    mod = module()
    with pytest.raises(RuntimeError, match="gas_mask"):
        mod.forward(torch.tensor([1200.]), torch.tensor([1e5]), torch.ones(1, 4),
                    {"stoich": torch.tensor(NU).unsqueeze(-1), "gas_mask": bad_mask})


def test_implicit_solution_converges_with_timestep(tmp_path):
    op, kin = load(tmp_path)
    t, p = torch.tensor([1200.]), torch.tensor([1e5])

    def integrate(dt):
        c = concentrations(op)
        for _ in range(round(20 / dt)):
            rate, dc, _ = kin.forward(t, p, c)
            jac = kin.jacobian(t, c, torch.ones(1), rate, dc)
            c = c + kt.evolve_implicit(rate, kin.buffer("stoich"), jac, dt)
        return c

    coarse, half, fine = integrate(4.), integrate(2.), integrate(.125)
    assert torch.linalg.norm(half - fine) < .65 * torch.linalg.norm(coarse - fine)


@pytest.mark.parametrize("dtype,na,h2s", [
    (torch.float64, 1e100, 0.), (torch.float64, 0., 1e200),
    (torch.float32, 1e12, 0.), (torch.float32, 0., 1e24),
])
def test_trace_extent_with_one_large_product(dtype, na, h2s):
    # One product is absent; the other makes the root far smaller than the
    # empty-air upper bound. The dominant-balance roots are accurate to far
    # below floating-point precision for these concentration ratios.
    mod = module()
    mod.to(dtype)
    c = torch.tensor([[na, h2s, 1., .01]], dtype=dtype, requires_grad=True)
    result = direct(mod, c)
    kc = math.exp((22.48 - 27778. / 1200.) * math.log(10) - 2 * math.log(R * 1200.))
    expected = kc / na**2 if na else math.sqrt(kc / h2s) / 2
    assert result.item() / KAPPA == pytest.approx(expected, rel=3e-5 if dtype == torch.float32 else 2e-10, abs=0.)
    result.sum().backward()
    assert torch.isfinite(c.grad).all()


@pytest.mark.parametrize("device", DEVICES)
def test_fixed_energy_kinetics_jacobian_includes_temperature_feedback(tmp_path, device):
    op, _ = load(tmp_path)
    op.evolve_temperature(True)
    kin = kt.Kinetics(op)
    kin.to(torch.device(device))
    c = concentrations(op).to(device)
    t, p = torch.tensor([1200.], device=device), torch.tensor([1e5], device=device)
    cv_by_name = {"Na": 1.5, "H2S": 2.5, "H2": 2.5, "Na2S(s)": 9.}
    cv = R * torch.tensor([cv_by_name[name] for name in op.species()], device=device)
    cvol = (c * cv).sum(-1)
    energy = cvol * (t - 300.)
    rate, dc, dt = kin.forward(t, p, c)
    assert dt is not None
    jac = kin.jacobian(t, c, cvol, rate, dc, dt).clone()
    for i, name in enumerate(op.species()):
        step = 1e-8 if name in ("Na", "H2S") else 1e-6
        plus, minus = c.clone(), c.clone()
        plus[0, i] += step
        minus[0, i] -= step
        temp_plus = 300. + energy / (plus * cv).sum(-1)
        temp_minus = 300. + energy / (minus * cv).sum(-1)
        fd = (kin.forward(temp_plus, p, plus)[0] - kin.forward(temp_minus, p, minus)[0]) / (2 * step)
        assert jac[0, 0, i].item() == pytest.approx(fd.item(), rel=3e-6, abs=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("general", [False, True])
@pytest.mark.parametrize("wrt", ["conc", "dc", "dt"])
@pytest.mark.parametrize("solid,explicit", [(0., True), (.01, True), (.01, False)])
def test_jacobian_backward_preserves_saved_tensors(tmp_path, device, dtype, general, wrt, solid, explicit):
    card = CARD if general else CARD.replace("Na2S(s) + H2 => 2 Na + H2S", "Na2S(s) => Na + H2S")
    op, kin = load(tmp_path, card)
    kin.to(torch.device(device))
    kin.to(dtype)
    c = concentrations(op, solid=solid).to(device=device, dtype=dtype)
    t, p = torch.full((1,), 1200., device=device, dtype=dtype), torch.full((1,), 1e5, device=device, dtype=dtype)
    rate, dc, _, kf = kin.forward_with_rate_constants(t, p, c)
    c.requires_grad_(wrt == "conc")
    dc.requires_grad_(wrt == "dc")
    dt = torch.ones_like(rate, requires_grad=True) if wrt == "dt" else None
    jac = kin.jacobian(t, c, torch.ones_like(t), rate, dc, dt, rate_constants=kf if explicit else None)
    jac.sum().backward()
    solid_id = op.species().index("Na2S(s)")
    if wrt == "dc":
        torch.testing.assert_close(dc.grad, c[..., solid_id, None, None].expand_as(dc))
    elif wrt == "dt":
        cv_by_name = {"Na": 1.5, "H2S": 2.5, "H2": 2.5, "Na2S(s)": 9.}
        total_cv = sum(cv_by_name[name] for name in op.species())
        expected = torch.full_like(dt, -solid * total_cv * R * (1200. - 300.))
        torch.testing.assert_close(dt.grad, expected)
    else:
        expected = torch.zeros_like(c)
        expected[..., solid_id] = dc.sum((-1, -2))
        if not explicit:
            expected[..., solid_id] -= rate.sum(-1) / solid**2
        torch.testing.assert_close(c.grad, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("solid", [0., .01])
@pytest.mark.parametrize("general", [False, True])
def test_jacobian_retained_forward_survives_interleaved_batches(tmp_path, device, dtype, solid, general):
    card = CARD if general else CARD.replace("Na2S(s) + H2 => 2 Na + H2S", "Na2S(s) => Na + H2S")
    op, kin = load(tmp_path, card)
    kin.to(torch.device(device))
    kin.to(dtype)
    c = concentrations(op, solid=solid).to(device=device, dtype=dtype).expand(3, -1).clone()
    t, p = torch.full((3,), 1200., device=device, dtype=dtype), torch.full((3,), 1e5, device=device, dtype=dtype)
    rate, dc, _, kf = kin.forward_with_rate_constants(t, p, c)
    before = kin.jacobian(t, c, torch.ones_like(t), rate, dc, rate_constants=kf)
    kin.forward(t[:1] + 100., p[:1] * 2., c[:1])
    after = kin.jacobian(t, c, torch.ones_like(t), rate, dc, rate_constants=kf)
    torch.testing.assert_close(after, before)
    if solid:
        torch.testing.assert_close(kin.jacobian(t, c, torch.ones_like(t), rate, dc), before)
    else:
        assert (after[..., op.species().index("Na2S(s)")] > 0).all()
        with pytest.raises(RuntimeError, match="rate_constants"):
            kin.jacobian(t, c, torch.ones_like(t), rate, dc)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_solid_jacobian_preserves_pressure_and_override(tmp_path, device, dtype):
    op, kin = load(tmp_path, CARD.replace("diff_P: 0.", "diff_P: -1."))
    kin.to(torch.device(device))
    kin.to(dtype)
    c = concentrations(op, solid=0.).to(device=device, dtype=dtype).expand(2, -1).clone()
    t = torch.full((2,), 1200., device=device, dtype=dtype)
    p = torch.tensor([1e5, 2e5], device=device, dtype=dtype)
    rate, dc, _, kf = kin.forward_with_rate_constants(t, p, c)
    jac = kin.jacobian(t, c, torch.ones_like(t), rate, dc, rate_constants=kf)
    solid_id = op.species().index("Na2S(s)")
    torch.testing.assert_close(jac[..., solid_id], kf)
    torch.testing.assert_close(kf[0], 2 * kf[1])
    override = torch.full_like(kf, .123)
    rate, dc, _, kf = kin.forward_with_rate_constants(t, p, c, {"kf_override": override})
    jac = kin.jacobian(t, c, torch.ones_like(t), rate, dc, rate_constants=kf)
    torch.testing.assert_close(jac[..., solid_id], override)
    assert torch.count_nonzero(rate) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("general,negative_species", [(False, "Na"), (False, "Na2S(s)"), (True, "Na"), (True, "H2"), (True, "Na2S(s)")])
def test_negative_concentrations_clamp_rate_and_derivative(tmp_path, device, dtype, negative_species, general):
    card = CARD if general else CARD.replace("Na2S(s) + H2 => 2 Na + H2S", "Na2S(s) => Na + H2S")
    data = yaml.safe_load(card)
    op, kin = load(tmp_path, yaml.safe_dump(data))
    kin.to(torch.device(device))
    kin.to(dtype)
    values = dict(zip(NAMES, [1e-5, 2e-5, .8, .01]))
    values[negative_species] = -.01
    c = torch.tensor([[values[name] for name in op.species()]], dtype=dtype, device=device)
    t, p = torch.full((1,), 1200., device=device, dtype=dtype), torch.full((1,), 1e5, device=device, dtype=dtype)
    rate, dc, _, kf = kin.forward_with_rate_constants(t, p, c)
    jac = kin.jacobian(t, c, torch.ones_like(t), rate, dc, rate_constants=kf)
    expected = kin.forward(t, p, c.clamp_min(0))[0]
    torch.testing.assert_close(rate, expected)
    assert torch.isfinite(jac).all()
    assert torch.count_nonzero(jac[..., op.species().index(negative_species)]) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_three_gas_products_direct_extent(device, dtype):
    mod = module(["Na2S(s) => Na + H2S + H2"])
    mod.to(torch.device(device))
    mod.to(dtype)
    c = torch.tensor([[0., 0., 0., .01]], device=device, dtype=dtype, requires_grad=True)
    s = torch.tensor([[1.], [1.], [1.], [-1.]], device=device, dtype=dtype)
    rate = direct(mod, c, stoich=s)
    expected = reference([0., 0., 0.], nu=(1., 1., 1.))
    assert rate.item() / KAPPA == pytest.approx(expected, rel=3e-5 if dtype == torch.float32 else 2e-10)
    rate.sum().backward()
    assert torch.isfinite(c.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_cuda_rate_and_jacobian_agree(tmp_path, dtype):
    outputs = []
    for device in ["cpu", "cuda"]:
        op, kin = load(tmp_path)
        kin.to(torch.device(device))
        kin.to(dtype)
        c = concentrations(op).to(device=device, dtype=dtype)
        t, p = torch.full((1,), 1200., device=device, dtype=dtype), torch.full((1,), 1e5, device=device, dtype=dtype)
        rate, dc, _, kf = kin.forward_with_rate_constants(t, p, c)
        jac = kin.jacobian(t, c, torch.ones_like(t), rate, dc, rate_constants=kf)
        outputs.append((rate.cpu(), jac.cpu()))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=3e-5 if dtype == torch.float32 else 2e-10, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_negative_unrelated_gas_has_no_effect(device, dtype):
    mod = module(["Na2S(s) => 2 Na + H2S"])
    mod.to(torch.device(device))
    mod.to(dtype)
    c = torch.tensor([[1e-5, 2e-5, -.01, .01]], dtype=dtype, device=device, requires_grad=True)
    s = torch.tensor([[2.], [1.], [0.], [-1.]], dtype=dtype, device=device)
    rate = direct(mod, c, stoich=s)
    torch.testing.assert_close(rate, direct(mod, c.detach().clamp_min(0), stoich=s))
    rate.sum().backward()
    assert c.grad[0, 2] == 0
    assert torch.isfinite(c.grad).all()
