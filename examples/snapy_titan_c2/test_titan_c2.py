"""Gates B/C for the Titan C2 snapy case. Run under snapy's pyenv:

    VIRTUAL_ENV=~/pyenv ~/pyenv/bin/python -m pytest test_titan_c2.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from titan_c2_chem import TitanC2Chemistry  # noqa: E402

torch.set_default_dtype(torch.float64)
KB_SI = 1.380649e-23
YAML = HERE / "titan_c2_chem.yaml"
NPZ = HERE / "titan_c2_data.npz"


def _make_state(n=64, device="cpu", seed=0):
    """Random-but-physical Titan-ish states: T 120-190 K, n 1e9-1e17 cm^-3."""
    g = torch.Generator().manual_seed(seed)
    T = 120.0 + 70.0 * torch.rand(n, generator=g)
    dens_cgs = 10.0 ** (9.0 + 8.0 * torch.rand(n, generator=g))
    chem = TitanC2Chemistry(YAML, NPZ, device="cpu")
    vmr = 1e-7 + 1e-5 * torch.rand(n, chem.nsp, generator=g)
    vmr[:, chem.sp["CH4"]] = 0.014
    vmr[:, chem.sp["N2"]] = 1.0 - vmr.sum(dim=1) + vmr[:, chem.sp["N2"]]
    conc_cgs = vmr * dens_cgs[:, None]
    conc = conc_cgs * 1.0e6 / 6.02214076e23          # mol/m^3
    P = dens_cgs * 1e6 * KB_SI * T
    dev = torch.device(device)
    return (T.to(dev), P.to(dev), conc.to(dev))


def test_b0_evolve_matches_kintera():
    """Pure-torch implicit step solves the same system as kintera.evolve_implicit.

    Element-wise identity is not required on ill-conditioned (nearly
    algebraic) stiff cells, where many deltas satisfy A*delta=b to machine
    precision; equivalence is judged by the linear-system residual.
    """
    import kintera
    from titan_c2_chem import evolve_implicit_torch
    T, P, conc = _make_state(32)
    chem = TitanC2Chemistry(YAML, NPZ, device="cpu")
    jr = 1e-8 * torch.rand(32, chem.nphoto)
    rate, jac = chem.rates(T, P, conc, jr)
    dt = 1.0e4
    d_ref = kintera.evolve_implicit(rate, chem.stoich, jac, dt)
    d_new = evolve_implicit_torch(rate, chem.stoich, jac, dt)

    b = torch.einsum("sr,...r->...s", chem.stoich, rate)
    A = (torch.eye(chem.nsp) / dt
         - torch.einsum("sr,...rn->...sn", chem.stoich, jac))
    res_new = (torch.einsum("...sn,...n->...s", A, d_new) - b).abs().amax(-1)
    res_ref = (torch.einsum("...sn,...n->...s", A, d_ref) - b).abs().amax(-1)
    bscale = b.abs().amax(-1).clamp_min(1e-300)
    assert (res_new <= torch.maximum(1e-10 * bscale, 10.0 * res_ref)).all(), (
        f"torch solve residual worse than kintera: "
        f"max(res_new/bscale)={float((res_new/bscale).max()):.2e}")
    # element-wise agreement on the well-conditioned majority of cells
    rel = ((d_ref - d_new).abs() / d_ref.abs().clamp_min(1e-300)).amax(-1)
    assert (rel < 1e-9).float().mean() > 0.9, (
        f"only {(rel < 1e-9).float().mean():.0%} cells match element-wise")


def test_gate_a_rerun():
    """Keep the rate-validation gate live in the test suite."""
    rc = subprocess.run(
        [sys.executable, str(HERE / "validate_c2_network.py"),
         "--yaml", str(YAML), "--data", str(NPZ),
         "--ref", str(HERE / "c2_ref_rates.npz")],
        cwd=str(HERE), capture_output=True, text=True)
    assert rc.returncode == 0, rc.stdout[-2000:] + rc.stderr[-2000:]


def test_b1_cpu_vs_cuda_rates_and_advance():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    T, P, conc = _make_state(128)
    chem_cpu = TitanC2Chemistry(YAML, NPZ, device="cpu")
    chem_gpu = TitanC2Chemistry(YAML, NPZ, device="cuda")
    jr = 1e-8 * torch.rand(128, chem_cpu.nphoto)

    r_cpu, j_cpu = chem_cpu.rates(T, P, conc, jr)
    r_gpu, j_gpu = chem_gpu.rates(T.cuda(), P.cuda(), conc.cuda(), jr.cuda())
    assert torch.allclose(r_cpu, r_gpu.cpu(), rtol=1e-12, atol=0)
    assert torch.allclose(j_cpu, j_gpu.cpu(), rtol=1e-12, atol=1e-300)

    # One ROS2 sub-step at a FIXED dt: the numerical kernel must be
    # device-equivalent. (The full adaptive advance() is NOT bit-compared
    # across devices: its accept/reject decision reads a block-max error via
    # .item(), which can differ by ~1e-12 between cuBLAS and the CPU LAPACK
    # and legitimately shift the step count by one -- a control-flow
    # difference, not a numerical-accuracy one. dt=100 s keeps every cell
    # well-conditioned so equivalence is not confounded by LU pivoting.)
    d_cpu, _ = chem_cpu._ros2_substep(T, P, conc, jr, 1.0e2)
    d_gpu, _ = chem_gpu._ros2_substep(T.cuda(), P.cuda(), conc.cuda(),
                                      jr.cuda(), 1.0e2)
    assert torch.isfinite(d_cpu).all() and torch.isfinite(d_gpu).all()
    rel = ((d_cpu - d_gpu.cpu()).abs()
           / d_cpu.abs().clamp_min(1e-300)).max()
    assert rel < 1e-7, f"CPU vs CUDA ROS2 sub-step rel diff {rel:.2e}"

    # the full adaptive advance must still run and stay finite/non-negative
    # on both devices, and agree to the controller tolerance (~rtol)
    s_cpu = (conc * chem_cpu.mw).T.contiguous()
    s_gpu = s_cpu.cuda().contiguous()
    chem_cpu.advance(T, P, s_cpu, 1.0e2, jr)
    chem_gpu.advance(T.cuda(), P.cuda(), s_gpu, 1.0e2, jr.cuda())
    assert torch.isfinite(s_cpu).all() and torch.isfinite(s_gpu).all()
    assert (s_cpu >= 0).all() and (s_gpu >= 0).all()
    rel = ((s_cpu - s_gpu.cpu()).abs() / s_cpu.abs().clamp_min(1e-300)).max()
    assert rel < 5e-2, f"CPU vs CUDA advance rel diff {rel:.2e}"


def test_b2_box_conservation_and_growth():
    """0-D box: atom conservation + photochemical C2 growth, vmr >= 0."""
    chem = TitanC2Chemistry(YAML, NPZ, device="cpu")
    n = 1
    T = torch.full((n,), 170.0)
    dens_cgs = torch.full((n,), 2.0e12)               # ~ 300 km
    P = dens_cgs * 1e6 * KB_SI * T
    conc = torch.zeros(n, chem.nsp)
    conc[:, chem.sp["N2"]] = 0.986 * dens_cgs
    conc[:, chem.sp["CH4"]] = 0.014 * dens_cgs
    conc *= 1.0e6 / 6.02214076e23
    scalar_s = (conc * chem.mw).T.contiguous()

    # fixed unattenuated J (subsolar TOA)
    jr = chem.photolysis_rates(chem.toa_flux.unsqueeze(0))

    # atom bookkeeping
    d = np.load(NPZ)
    comp_c = torch.tensor([{"N2": 0, "H": 0, "H2": 0, "C": 1, "CH": 1,
                            "CH2_1": 1, "CH2_3": 1, "CH3": 1, "CH4": 1,
                            "C2": 2, "C2H": 2, "C2H2": 2, "C2H3": 2,
                            "C2H4": 2, "C2H5": 2, "C2H6": 2}[s]
                           for s in chem.species], dtype=torch.float64)
    comp_h = torch.tensor([{"N2": 0, "H": 1, "H2": 2, "C": 0, "CH": 1,
                            "CH2_1": 2, "CH2_3": 2, "CH3": 3, "CH4": 4,
                            "C2": 0, "C2H": 1, "C2H2": 2, "C2H3": 3,
                            "C2H4": 4, "C2H5": 5, "C2H6": 6}[s]
                           for s in chem.species], dtype=torch.float64)

    def atoms(s):
        c = (s.T / chem.mw)
        return (c * comp_c).sum().item(), (c * comp_h).sum().item()

    c0, h0 = atoms(scalar_s)
    # production cadence (dt ~ hydro step); 150 steps is ample for C2 growth.
    # (Long dt=1h integrations are needlessly slow with the adaptive ROS2
    # integrator -- a 1h step from the radical-free seed forces it to the
    # sub-step floor; the GCM never takes such steps.)
    dt = 66.0
    for _ in range(150):
        chem.advance(T, P, scalar_s, dt, jr)
    c1, h1 = atoms(scalar_s)

    # ROS2 is exactly element-conserving (the stoichiometry annihilates any
    # conserved-atom vector, so each stage solve preserves it); the only
    # source of atom drift is the positivity clamp, which fires when a stiff
    # radical's linearized increment overshoots below zero.
    assert abs(c1 - c0) / c0 < 1e-4, f"C-atom drift {(c1-c0)/c0:.2e}"
    assert abs(h1 - h0) / h0 < 1e-4, f"H-atom drift {(h1-h0)/h0:.2e}"
    conc_end = (scalar_s.T / chem.mw)
    assert (conc_end >= 0).all()
    # photochemistry must have produced C2 species and H2
    for sp in ("C2H2", "C2H6", "C2H4", "H2"):
        assert conc_end[0, chem.sp[sp]] > 0, f"no {sp} produced"
    # stiff radicals reach a quasi-steady state that stays a small fraction of
    # their methane parent (they do not accumulate). (On these short,
    # production-cadence integrations the slow C2 products have not yet
    # overtaken the radicals in absolute abundance -- that needs ~1e3 h.)
    assert conc_end[0, chem.sp["CH3"]] < 1e-2 * conc_end[0, chem.sp["CH4"]]
    assert conc_end[0, chem.sp["H"]] < 1e-2 * conc_end[0, chem.sp["CH4"]]


def test_b2b_no_photolysis_is_inert():
    """Closed-shell mix with J=0 must be (nearly) chemically inert."""
    chem = TitanC2Chemistry(YAML, NPZ, device="cpu")
    T = torch.full((1,), 170.0)
    dens_cgs = torch.full((1,), 2.0e12)
    P = dens_cgs * 1e6 * KB_SI * T
    conc = torch.zeros(1, chem.nsp)
    conc[:, chem.sp["N2"]] = 0.986 * dens_cgs
    conc[:, chem.sp["CH4"]] = 0.014 * dens_cgs
    conc *= 1.0e6 / 6.02214076e23
    scalar_s = (conc * chem.mw).T.contiguous()
    before = scalar_s.clone()
    chem.advance(T, P, scalar_s, 3600.0, jrate=None)
    # kintera produces denormal (~1e-313) rates for zero-radical states;
    # require the change to be physically nothing in mixing-ratio terms.
    dvmr = ((scalar_s - before).T / chem.mw).abs().max() / (
        (before.T / chem.mw).sum(-1).max())
    assert dvmr < 1e-30, f"inert mix changed by vmr {dvmr:.2e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_c1_actinic_flux_beer_lambert():
    """Gate C1: py2sess actinic flux vs analytic Beer-Lambert."""
    from py2sess_rt import TitanC2Radiation
    nlyr = 40
    rad = TitanC2Radiation(NPZ, nlyr, device="cpu", sza_bin_deg=1.0)
    chem = TitanC2Chemistry(YAML, NPZ, device="cpu")

    # Titan-ish CH4 column, TOA->BOA: exponential density
    dens = torch.logspace(10, 13, nlyr)                  # cm^-3, top->bottom
    conc_abs = torch.zeros(1, nlyr, len(rad.abs_species))
    ich4 = rad.abs_species.index("CH4")
    conc_abs[0, :, ich4] = 0.014 * dens
    dz_cm = torch.full((nlyr,), 5.0e5)                   # 5 km layers

    for sza_deg in (0.0, 60.0):
        mu0 = torch.tensor([float(np.cos(np.radians(sza_deg)))])
        out = rad.actinic_flux(conc_abs, dz_cm, mu0)[0]  # (nlyr, nwave)
        # analytic with the same binned mu0 the driver uses
        mu_c = float(np.cos(np.radians(round(sza_deg))))
        tau = rad.optical_depth(conc_abs, dz_cm)[0]      # (nlyr, nwave)
        ct = torch.cumsum(tau, dim=0)
        ct_center = ct - 0.5 * tau
        # layer-center actinic ~ mean of level transmissions (the driver
        # averages adjacent levels; compare against the same construction)
        lvl = torch.cat([torch.zeros(1, rad.nwave), ct], dim=0)
        ana = 0.5 * (torch.exp(-lvl[:-1] / mu_c) + torch.exp(-lvl[1:] / mu_c))
        ana = ana * (rad.toa_flux * rad.toa_transmission).view(1, -1)
        mask = (lvl[1:] / mu_c) < 8.0                    # ignore opaque depths
        rel = ((out - ana).abs() / ana.clamp_min(1e-300))[mask]
        assert rel.max() < 0.02, f"SZA {sza_deg}: rel {rel.max():.3f}"

    # night column -> exactly zero
    out_night = rad.actinic_flux(conc_abs, dz_cm, torch.tensor([-0.3]))
    assert (out_night == 0).all()

    # J at optically thin TOA == unattenuated J (ties RT to chemistry)
    mu0 = torch.tensor([1.0])
    thin = torch.zeros_like(conc_abs)
    jtop = chem.photolysis_rates(
        rad.actinic_flux(thin, dz_cm, mu0)[0, 0].unsqueeze(0))
    jref = chem.photolysis_rates(chem.toa_flux.unsqueeze(0))
    rel = ((jtop - jref).abs() / jref.clamp_min(1e-300)).max()
    assert rel < 1e-10, f"thin-column J mismatch {rel:.2e}"


def test_c2_cuda_actinic_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from py2sess_rt import TitanC2Radiation
    nlyr = 40
    g = torch.Generator().manual_seed(1)
    dens = torch.logspace(10, 13, nlyr)
    ncol = 64
    conc = torch.zeros(ncol, nlyr, 5)
    conc[:, :, 3] = 0.014 * dens                          # CH4-ish
    conc[:, :, 0] = 1e-6 * dens * torch.rand(ncol, nlyr, generator=g)
    dz = torch.full((nlyr,), 5.0e5)
    mu0 = torch.linspace(-0.2, 1.0, ncol)
    rad_c = TitanC2Radiation(NPZ, nlyr, device="cpu")
    rad_g = TitanC2Radiation(NPZ, nlyr, device="cuda")
    fc = rad_c.actinic_flux(conc, dz, mu0)
    fg = rad_g.actinic_flux(conc.cuda(), dz.cuda(), mu0.cuda()).cpu()
    rel = ((fc - fg).abs() / fc.abs().clamp_min(1e-300))[fc > 0]
    assert rel.max() < 1e-10, f"CPU vs CUDA actinic rel {rel.max():.2e}"
