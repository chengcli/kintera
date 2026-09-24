"""Independent oracle for `use-h2-dissociation`: H2 <=> 2H on one lumped H/He species.

The reference is rebuilt here from data/nasa9.dat with the textbook NASA-9 forms and a
partial-pressure equilibrium solved by bisection, and compared with the EOS: the H fraction and
the particle factor (both through P = cz c R T), U(T) - U(300 K), cv against dU/dT, and the
entropy: T ds = du + P dv, S against the oracle's mixture entropy, and ThermoX's adiabat.
Every card here names species 0 `H2` (the species registry is process-global), so the
use-nasa9-cp / use-h2-cp cases also exercise their overlap with the lumped species.
"""
import pathlib
import re
import subprocess
import sys

import numpy as np
import pytest
import torch
import kintera
from kintera import ThermoOptions, ThermoX, ThermoY

torch.set_default_dtype(torch.float64)
ROOT = pathlib.Path(__file__).resolve().parents[1]
R, P0 = 8.31446, 1.0e5  # kintera's Rgas [J/(mol K)]; NASA-9 standard pressure [Pa]
NH, NHE = 1.6667, 0.16667
TS = np.array([2500.0, 3500.0, 4500.0])
CS = [1.0, 1.0e3]  # mol/m^3 of the lumped species


def nasa9(name):
    """(low, high) rows of 10 numbers: a1..a7, 0, b1, b2."""
    lines = (ROOT / "data" / "nasa9.dat").read_text().splitlines()
    i = [l.strip() for l in lines].index(name)
    num = [float(x) for x in re.findall(r"-?\d\.\d+E[+-]\d+", " ".join(lines[i + 1:i + 5]))]
    return np.array(num[:10]), np.array(num[10:20])


def h_s(name, T):
    """H/(RT) and S/R (McBride et al. 2002, NASA-9 form); range switch at 1000 K."""
    lo, hi = nasa9(name)
    a = np.where((T >= 1000.0)[:, None], hi, lo)
    t = [T**k for k in range(-2, 5)]
    h = (-a[:, 0] * t[0] + a[:, 1] * np.log(T) / T + a[:, 2] + a[:, 3] * T / 2
         + a[:, 4] * t[4] / 3 + a[:, 5] * t[5] / 4 + a[:, 6] * t[6] / 5 + a[:, 8] / T)
    s = (-a[:, 0] * t[0] / 2 - a[:, 1] * t[1] + a[:, 2] * np.log(T) + a[:, 3] * T
         + a[:, 4] * t[4] / 2 + a[:, 5] * t[5] / 3 + a[:, 6] * t[6] / 4 + a[:, 9])
    return h, s


def oracle(T, c):
    """[H] from Kp = p_H^2 / (p_H2 P0) by bisection; returns (x_H, cz, U [J/mol])."""
    (hH, sH), (hH2, sH2), (hHe, _) = h_s("H", T), h_s("H2", T), h_s("He", T)
    lnKp = -(2 * (hH - sH) - (hH2 - sH2))
    lo, hi = np.zeros_like(T), np.full_like(T, NH * c)
    for _ in range(200):
        x = 0.5 * (lo + hi)
        f = 2 * np.log(x * R * T) - np.log((NH * c - x) / 2 * R * T) - np.log(P0) - lnKp
        lo, hi = np.where(f < 0, x, lo), np.where(f < 0, hi, x)
    n = [x, (NH * c - x) / 2, np.full_like(T, NHE * c)]  # H, H2, He
    U = sum(ni * (hi_ - 1.0) * R * T for ni, hi_ in zip(n, [hH, hH2, hHe])) / c
    return x / sum(n), sum(n) / c, U


def s_oracle(T, c):
    """S [J/(mol K)] per mole of lumped species: ideal H2/H/He mixture, s_i - R ln(p_i/P0)."""
    xH, cz, _ = oracle(T, c)
    n = [xH * cz * c, (NH * c - xH * cz * c) / 2, np.full_like(T, NHE * c)]
    return sum(ni * R * (h_s(sp, T)[1] - np.log(ni * R * T / P0))
               for ni, sp in zip(n, ["H", "H2", "He"])) / c


def identity_residual(card, T, rho, y):
    """max |T ds - du - P dv| / |du + P dv| per unit mass, by centred differences in T and in rho."""
    th, h = ThermoY(ThermoOptions.from_yaml(str(card))), 1e-5
    def sup(T, rho):
        V = th.compute("DY->V", (rho, y))
        P = th.compute("VT->P", (V, T))
        return th.compute("PVT->S", (P, V, T)) / rho, th.compute("VT->U", (V, T)) / rho, P
    (s1, u1, _), (s2, u2, _) = sup(T * (1 + h), rho), sup(T * (1 - h), rho)
    rT = (T * (s1 - s2) - (u1 - u2)) / (u1 - u2)
    (s1, u1, _), (s2, u2, _), (_, _, P) = sup(T, rho * (1 + h)), sup(T, rho * (1 - h)), sup(T, rho)
    dv = 1 / (rho * (1 + h)) - 1 / (rho * (1 - h))
    rv = (T * (s1 - s2) - (u1 - u2) - P * dv) / (u1 - u2 + P * dv)
    return float(torch.cat([rT, rv]).abs().max())


def thermo(tmp_path, extra):
    card = tmp_path / "h2diss.yaml"
    card.write_text("reference-state: {Tref: 300.0, Pref: 1.0e5, use-h2-dissociation: true%s}\n"
                    "species:\n- {name: H2, composition: {H: %r, He: %r}, cv_R: 2.5}\n"
                    % (extra, NH, NHE))
    return ThermoY(ThermoOptions.from_yaml(str(card)))


@pytest.mark.parametrize("extra", ["", ", fused-h2diss: true", ", use-nasa9-cp: true",
                                   ", use-h2-cp: true"])
@pytest.mark.parametrize("c", CS)
def test_matches_nasa9_oracle(tmp_path, extra, c):
    th = thermo(tmp_path, extra)
    T = torch.tensor([*TS, 300.0])
    V = th.compute("DY->V", (torch.full((4,), c * kintera.species_weights()[0]), torch.zeros(0, 4)))
    cz = (th.compute("VT->P", (V, T)) / (c * R * T)).numpy()[:3]
    U = th.compute("VT->U", (V, T)).numpy() / c
    xH_ref, cz_ref, U_ref = oracle(TS, c)
    _, _, U300 = oracle(np.array([300.0]), c)
    xH = 2 * (cz - NH / 2 - NHE) / cz
    print("c=%g T=%s xH=%s cz=%s" % (c, TS, xH, cz))
    np.testing.assert_allclose(cz, cz_ref, rtol=1e-10)
    np.testing.assert_allclose(xH, xH_ref, rtol=1e-6)
    np.testing.assert_allclose(U[:3] - U[3], U_ref - U300, rtol=1e-9)
    dT = 1e-5 * TS  # cv against a centred difference of the oracle's U(T)
    dUdT = (oracle(TS + dT, c)[2] - oracle(TS - dT, c)[2]) / (2 * dT)
    cv = th.compute("VT->cv", (V, T)).numpy()[:3] / c
    np.testing.assert_allclose(cv, dUdT, rtol=1e-7)


def test_entropy_is_consistent(tmp_path):
    """T ds = du + P dv, S equals the oracle mixture entropy, and ThermoX's adiabat keeps the oracle S."""
    th, T = thermo(tmp_path, ""), torch.tensor(np.repeat(np.linspace(2500.0, 4500.0, 5), 2))
    c = torch.tensor(np.tile(CS, 5))
    rho = c * kintera.species_weights()[0]
    assert identity_residual(tmp_path / "h2diss.yaml", T, rho, torch.zeros(0, 10)) < 1e-6
    V = th.compute("DY->V", (rho, torch.zeros(0, 10)))
    S = th.compute("PVT->S", (th.compute("VT->P", (V, T)), V, T)).numpy() / c.numpy()
    np.testing.assert_allclose(S, s_oracle(T.numpy(), c.numpy()), rtol=1e-9)
    thx = ThermoX(ThermoOptions.from_yaml(str(tmp_path / "h2diss.yaml")))
    for T0, P0, dlnp in [(3000.0, 1e4, -0.1), (3500.0, 1e5, 0.1), (4000.0, 1e6, -0.1)]:
        Tt, Pt, X = torch.tensor([T0]), torch.tensor([P0]), torch.ones(1, 1)
        c0 = thx.compute("TPX->V", (Tt, Pt, X))[0, 0].item()
        thx.extrapolate_dlnp(Tt, Pt, X, dlnp)
        c1 = thx.compute("TPX->V", (Tt, Pt, X))[0, 0].item()
        dS = s_oracle(Tt.numpy(), np.array([c1])) - s_oracle(np.array([T0]), np.array([c0]))
        print("adiabat T0=%g dlnp=%g: T1=%.4f grad_ad=%.5f oracle dS=%.2e" % (T0, dlnp, Tt.item(), np.log(Tt.item() / T0) / dlnp, dS[0]))
        assert abs(dS[0]) < 1e-4, dS


def test_entropy_mixture_identity(tmp_path):
    """Lumped gas plus a second vapour: gas partial pressures are c_j R T (fresh interpreter)."""
    card = tmp_path / "mix.yaml"
    card.write_text("reference-state: {Tref: 300.0, Pref: 1.0e5, use-h2-dissociation: true}\nspecies:\n"
                    "- {name: H2, composition: {H: %r, He: %r}, cv_R: 2.5}\n"
                    "- {name: H2O, composition: {H: 2, O: 1}, cv_R: 3.0}\n"
                    "- {name: H2O(l), composition: {H: 2, O: 1}, cv_R: 9.0, u0_R: -3430.}\n"
                    "reactions:\n- {equation: H2O => H2O(l), type: nucleation, rate-constant: {formula: "
                    "h2o_ideal, T3: 273.16, P3: 611.7, beta: 24.845, delta: 4.986}}\n" % (NH, NHE))
    code = ("import sys, torch; sys.path.insert(0, %r); import test_h2diss_oracle as t; "
            "print(t.identity_residual(sys.argv[1], torch.tensor([2500., 3500., 4500.]), "
            "torch.tensor([0.003, 0.3, 3.0]), torch.tensor([[0.1] * 3, [0.0] * 3])))" % str(ROOT / "tests"))
    out = subprocess.run([sys.executable, "-c", code, str(card)], capture_output=True, text=True, check=True)
    assert float(out.stdout.split()[-1]) < 1e-6, out.stdout


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_helium_entropy_is_finite(tmp_path, dtype):
    """`He: 0` is legal: its n = 0 term adds 0 to S, not 0 * log(0) = NaN (float32 underflow)."""
    card = tmp_path / "he0.yaml"
    card.write_text("reference-state: {Tref: 300.0, Pref: 1.0e5, use-h2-dissociation: true}\n"
                    "species:\n- {name: H2, composition: {H: 2.0, He: 0}, cv_R: 2.5}\n")
    th = ThermoY(ThermoOptions.from_yaml(str(card)))
    th.to(dtype)
    T = torch.tensor(TS, dtype=dtype)
    rho = torch.full((3,), CS[0] * kintera.species_weights()[0], dtype=dtype)
    V = th.compute("DY->V", (rho, torch.zeros(0, 3, dtype=dtype)))
    S = th.compute("PVT->S", (th.compute("VT->P", (V, T)), V, T))
    assert torch.isfinite(S).all(), S
    if dtype == torch.float64:
        assert identity_residual(card, T, rho, torch.zeros(0, 3)) < 1e-6

def test_bad_inputs_rejected(tmp_path):
    for species, why in [("{name: H2, composition: {H: 1.6, He: -0.1}}", "He >= 0"),
                         ("{name: H2, composition: {H: 1.6, C: 0.1}}", "only H and He"),
                         ("{name: dry, composition: {H: 1.6}}", "not registry species 0")]:
        card = tmp_path / "bad.yaml"
        card.write_text("reference-state: {use-h2-dissociation: true}\nspecies:\n- %s\n" % species)
        with pytest.raises(RuntimeError, match=why):
            ThermoOptions.from_yaml(str(card))
    card.write_text("reference-state: {fused-h2diss: true}\nspecies:\n- {name: H2, composition: {H: 2}}\n")
    with pytest.raises(RuntimeError, match="needs use-h2-dissociation"):
        ThermoOptions.from_yaml(str(card))


FLAG_OFF = r"""
import sys, torch
from kintera import ThermoOptions, ThermoY
torch.set_default_dtype(torch.float64)
op = ThermoOptions.from_yaml(sys.argv[1])
th, ny = ThermoY(op), len(op.species()) - 1
rho, T = torch.tensor([0.1, 1.0, 10.0]), torch.tensor([150.0, 800.0, 3000.0])
V = th.compute("DY->V", (rho, torch.full((ny, 3), 1e-3)))
P, U, cv = (th.compute(k, (V, T)) for k in ("VT->P", "VT->U", "VT->cv"))
out = [P, U, cv, th.compute("VU->T", (V, U)), th.compute("PV->T", (P, V)), th.compute("PVT->S", (P, V, T))]
print(" ".join(float(x).hex() for t in out for x in t.flatten()))
"""


def test_flag_off_is_bit_identical(tmp_path):
    """jupiter.yaml as shipped vs with `use-h2-dissociation: false` (fresh interpreters)."""
    text = (ROOT / "tests" / "jupiter.yaml").read_text()
    off = tmp_path / "off.yaml"
    off.write_text(text.replace("reference-state:\n", "reference-state:\n  use-h2-dissociation: false\n", 1))
    run = lambda p: subprocess.run([sys.executable, "-c", FLAG_OFF, str(p)], capture_output=True,
                                   text=True, check=True).stdout
    assert run(ROOT / "tests" / "jupiter.yaml") == run(off)
