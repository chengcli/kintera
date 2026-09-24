"""YAML multi-range Arrhenius rates vs the Python options API (test_multirange_arrhenius.py)."""

import math

import pytest
import torch

import kintera as kt

torch.set_default_dtype(torch.float64)

SPECIES = ("H", "H2", "O", "O2", "N", "N2")
HEAD = "reference-state: {Tref: 300., Pref: 1.e5}\nspecies:\n"
HEAD += "".join(f"  - {{name: {s}, composition: {{{s[0]}: {len(s)}}}, cv_R: 2.5}}\n" for s in SPECIES)
HEAD += "reactions:\n"
A0, B0, E0, T0 = [1.0e3, 2.0e3, 5.0e2], [0.5, -1.0, 0.0], [100.0, 300.0, -50.0], [200.0, 500.0, 1.0e30]
A1, T1 = [3.0, 0.2], [400.0, 1.0e30]  # unimolecular (A in s^-1); b/Ea_R_ranges omitted -> 0
BANDS = [(A0, B0, E0, T0), (A1, [0.0, 0.0], [0.0, 0.0], T1)]
RXN = "- {{equation: {}, type: arrhenius, rate-constant: {{{}}}}}\n"
RANGED = RXN.format("H2 => 2 H", f"A_ranges: {A0}, b_ranges: {B0}, Ea_R_ranges: {E0}, T_ranges: {T0}")
RANGED += RXN.format("O2 => 2 O", f"A_ranges: {A1}, T_ranges: {T1}")
LEGACY = RXN.format("N2 => 2 N", "A: 7.0, b: 0.3, Ea_R: 40.0")
TGRID = torch.tensor([50.0, 150.0, 199.9, 200.0, 350.0, 400.0, 499.9, 500.0, 900.0, 3000.0])


def _load(tmp_path, rxns):
    (tmp_path / "rxns.yaml").write_text(HEAD + rxns)
    op = kt.KineticsOptions.from_yaml(str(tmp_path / "rxns.yaml")).arrhenius()
    return op, kt.Arrhenius(op)


def _rate(module, temp):
    return module.forward(temp, torch.ones_like(temp), torch.zeros(temp.shape + (1,)), {})


def test_yaml_ranges_equal_python_api(tmp_path):
    op, module = _load(tmp_path, RANGED + LEGACY)
    api = kt.ArrheniusOptions().A_ranges([A0, A1, [7.0]]).T_ranges([T0, T1, [1.0e30]])
    api.b_ranges([B0, [0.0, 0.0], [0.3]]).Ea_R_ranges([E0, [0.0, 0.0], [40.0]])
    assert op.A_ranges() == api.A_ranges() and op.T_ranges() == api.T_ranges()
    assert torch.equal(_rate(module, TGRID), _rate(kt.Arrhenius(api), TGRID))


@pytest.mark.parametrize("rxn, r", [(0, 0), (0, 1), (1, 0)])
def test_band_boundaries(tmp_path, rxn, r):
    """Just below T_r the rate uses band r; at T_r exactly it uses band r+1."""
    A, b, Ea, T = BANDS[rxn]
    below = math.nextafter(T[r], 0.0)
    got = _rate(_load(tmp_path, RANGED)[1], torch.tensor([below, T[r]]))[:, rxn]
    k = lambda t, i: A[i] * (t / 300.0) ** b[i] * math.exp(-Ea[i] / t)
    expected = torch.tensor([k(below, r), k(T[r], r + 1)])
    torch.testing.assert_close(got, expected, rtol=1e-13, atol=0.0)


def test_unranged_yaml_keeps_single_range_path(tmp_path):
    op, module = _load(tmp_path, LEGACY)
    assert op.A_ranges() == []
    api = kt.ArrheniusOptions().A([7.0]).b([0.3]).Ea_R([40.0])
    assert torch.equal(_rate(module, TGRID), _rate(kt.Arrhenius(api), TGRID))


def test_ranged_A_converted_like_A(tmp_path):
    # bimolecular: A_ranges and A both go cm^3 molecule^-1 s^-1 -> m^3 mol^-1 s^-1
    ranged = "A_ranges: [2.0e-11, 9.0e-12], b_ranges: [0.5, 0.5], Ea_R_ranges: [100., 100.], T_ranges: [250., 1.e30]"
    rxns = RXN.format("2 H => H2", ranged) + RXN.format("2 N => N2", "A: 2.0e-11, b: 0.5, Ea_R: 100.")
    op, module = _load(tmp_path, rxns)
    assert op.A_ranges()[0][0] == op.A()[1] == pytest.approx(2.0e-11 * 6.02214076e23 * 1.0e-6, rel=1e-14)
    rate = _rate(module, torch.tensor([100.0, 249.0]))
    assert torch.equal(rate[:, 0], rate[:, 1])


BAD = [("T_ranges: [1.e30]", "equal length"), ("b_ranges: [0.5, 0.5]", "must also define"),
       ("T_ranges: [200., 1.e30], b_ranges: [0.5]", "must match"),
       ("T_ranges: [200., 1.e30], Ea_R_ranges: [1., 2., 3.]", "must match")]


@pytest.mark.parametrize("extra, msg", BAD)
def test_inconsistent_ranges_rejected(tmp_path, extra, msg):
    with pytest.raises(RuntimeError, match=msg):
        _load(tmp_path, RXN.format("H2 => 2 H", f"A_ranges: [1.0, 2.0], {extra}"))


@pytest.mark.parametrize("bounds", [[500.0, 200.0, 1.0e30], [200.0, 200.0, 1.0e30]])
def test_non_increasing_T_ranges_rejected(tmp_path, bounds):
    with pytest.raises(RuntimeError, match="strictly increasing at reaction 1"):
        _load(tmp_path, LEGACY + RXN.format("H2 => 2 H", f"A_ranges: {A0}, T_ranges: {bounds}"))
    api = kt.ArrheniusOptions().A_ranges([A0]).b_ranges([B0]).Ea_R_ranges([E0]).T_ranges([bounds])
    with pytest.raises(RuntimeError, match="strictly increasing at reaction 0"):
        kt.Arrhenius(api)
