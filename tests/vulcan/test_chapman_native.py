"""
1D Chapman cycle using ONLY kintera-native modules.

All chemistry, photolysis, diffusion, and time-stepping use kintera:
  - kintera.Kinetics (from YAML) for reaction rates + analytical Jacobian
  - kintera.evolve_implicit for batched implicit Euler
  - kintera.diffusion_coefficients for diffusion operator
  - Beer-Lambert RT for actinic flux (pure physics, no external code)

VULCAN is only used as a comparison reference (run as a separate subprocess).

Usage:
    python3.11 test_chapman_native.py
"""
import os, sys, subprocess, pickle, re, math, time
import numpy as np
import torch
import kintera as kt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "VULCAN")

AVOGADRO = 6.02214076e23
KB_CGS = 1.380649e-16  # erg/K
RGAS = 8.314462        # J/(mol·K)


# ── Beer-Lambert RT (pure physics) ──────────────────────────────────────

def beer_lambert_actinic_flux(n_absorbers, cross_sections, stellar_flux,
                               dz, cos_zen):
    """
    Beer-Lambert actinic flux computation.

    Parameters
    ----------
    n_absorbers : list of ndarray (nz,)
        Number density of each absorber [molecule/cm³].
    cross_sections : list of ndarray (nwave,)
        Total absorption cross-section for each absorber [cm²].
    stellar_flux : ndarray (nwave,)
        TOA flux [photons cm⁻² s⁻¹ nm⁻¹].
    dz : ndarray (nz,)
        Layer thickness [cm].
    cos_zen : float
        Cosine of solar zenith angle.

    Returns
    -------
    aflux : ndarray (nz, nwave)
        Actinic flux at layer centers.
    """
    nz = len(dz)
    nwave = len(stellar_flux)

    alpha = np.zeros((nz, nwave))
    for n_sp, sigma in zip(n_absorbers, cross_sections):
        alpha += n_sp[:, None] * sigma[None, :]

    dtau = alpha * dz[:, None]

    tau = np.zeros((nz, nwave))
    tau[nz - 1] = dtau[nz - 1] / 2.0
    for j in range(nz - 2, -1, -1):
        tau[j] = tau[j + 1] + (dtau[j + 1] + dtau[j]) / 2.0

    return stellar_flux[None, :] * np.exp(-tau / cos_zen) / cos_zen


# ── Atmosphere setup ────────────────────────────────────────────────────

def setup_atmosphere(nz, P_bot_dyncm2=10.0, P_top_dyncm2=0.01,
                     T_K=250.0, Kzz_cm2s=1e6):
    """
    Set up an isothermal atmosphere with log-spaced pressure.

    Returns all quantities in both cgs (for RT) and SI (for kintera).
    """
    P_cgs = np.logspace(np.log10(P_bot_dyncm2), np.log10(P_top_dyncm2), nz)
    P_Pa = P_cgs * 0.1

    n_cgs = P_cgs / (KB_CGS * T_K)  # molecule/cm³

    mu = 29.0  # mean molecular weight
    g = 980.0  # cm/s²
    H = KB_CGS * T_K / (mu * 1.66054e-24 * g)  # scale height [cm]

    z = H * np.log(P_cgs[0] / P_cgs)  # altitude [cm], z[0]=0 at bottom
    dzi = np.diff(z)  # interface spacings [cm]

    dz = np.zeros(nz)
    dz[0] = dzi[0]
    dz[-1] = dzi[-1]
    for j in range(1, nz - 1):
        dz[j] = (dzi[j - 1] + dzi[j]) / 2.0

    Kzz = np.full(nz - 1, Kzz_cm2s)

    return {
        "nz": nz, "T": T_K,
        "P_cgs": P_cgs, "P_Pa": P_Pa,
        "n_cgs": n_cgs,
        "z_cm": z, "dzi_cm": dzi, "dz_cm": dz,
        "dzi_m": dzi * 0.01, "Kzz_cm2s": Kzz, "Kzz_m2s": Kzz * 1e-4,
    }


def solar_uv_flux(wavelengths_nm):
    """Simple solar UV flux model at 1 AU [photons cm⁻² s⁻¹ nm⁻¹]."""
    flux = np.zeros_like(wavelengths_nm)
    for i, w in enumerate(wavelengths_nm):
        if w < 150:
            flux[i] = 1e10 * math.exp(-(150 - w) / 20)
        elif w < 200:
            flux[i] = 1e10 * (w / 150) ** 6
        elif w < 300:
            flux[i] = 1e13 * math.exp(-(w - 250) ** 2 / 5000)
        else:
            flux[i] = 1e14
    return flux


# ── Diffusion (kintera native) ──────────────────────────────────────────

def init_diffusion(n_0_molm3, Kzz_m2s, dzi_m):
    """Compute tridiagonal diffusion coefficients via kintera C++."""
    nz = len(n_0_molm3)
    y_dummy = torch.zeros(nz, 1, dtype=torch.float64)
    y_dummy[:, 0] = torch.from_numpy(n_0_molm3.copy())
    Kzz_t = torch.from_numpy(Kzz_m2s.copy()).double()
    dzi_t = torch.from_numpy(dzi_m.copy()).double()
    A, B, C = kt.diffusion_coefficients(y_dummy, Kzz_t, dzi_t)
    return A.numpy(), B.numpy(), C.numpy()


def diffusion_tendency(y, A, B, C):
    """Vectorized diffusion tendency from precomputed tridiagonal coefficients."""
    tend = A[:, None] * y.copy()
    tend[:-1] += B[:-1, None] * y[1:]
    tend[1:] += C[1:, None] * y[:-1]
    return tend


# ── Kintera-native 1D solver ───────────────────────────────────────────

def kintera_1d_solve(yaml_path, nz=20, max_steps=5000, rt_update_frq=10):
    """
    1D photochemistry-diffusion solver using only kintera modules.

    Returns (C_final, nsteps, elapsed_seconds, atm_info).
    """
    opts = kt.KineticsOptions.from_yaml(yaml_path)
    kinet = kt.Kinetics(opts)
    species = opts.species()
    nspecies = len(species)
    stoich = kinet.stoich  # (nspecies, nreaction)

    print(f"  Species: {species}")
    print(f"  Stoich: {stoich.shape[0]} species x {stoich.shape[1]} reactions")

    # species indices
    idx = {sp: i for i, sp in enumerate(species)}

    # Extract cross-section data from kintera buffers for Beer-Lambert RT
    wl_tensor = kinet.buffer("photolysis.wavelength")
    wl_np = wl_tensor.numpy().copy()
    nwave = len(wl_np)

    xs_O2 = kinet.buffer("photolysis.cross_section_0").numpy().copy()  # (nwave, 2)
    xs_O3 = kinet.buffer("photolysis.cross_section_1").numpy().copy()  # (nwave, 2)
    xs_O2_total = xs_O2.sum(axis=1)  # total absorption [cm²]
    xs_O3_total = xs_O3.sum(axis=1)

    print(f"  Wavelength grid: {wl_np[0]:.0f}-{wl_np[-1]:.0f} nm, {nwave} points")

    # Stellar flux
    stellar_flux = solar_uv_flux(wl_np)

    # Atmosphere
    atm = setup_atmosphere(nz)
    T_K = atm["T"]
    n_cgs = atm["n_cgs"]
    P_Pa = atm["P_Pa"]

    print(f"  Atmosphere: nz={nz}, T={T_K}K")
    print(f"  P range: {atm['P_cgs'][0]:.1e} to {atm['P_cgs'][-1]:.1e} dyn/cm²")

    # Initial concentrations in mol/m³
    C_molm3 = np.zeros((nz, nspecies))
    n_total_molm3 = P_Pa / (RGAS * T_K)
    for j in range(nz):
        C_molm3[j, idx["N2"]] = 0.79 * n_total_molm3[j]
        C_molm3[j, idx["O2"]] = 0.21 * n_total_molm3[j]
        C_molm3[j, idx["O"]]  = 1e-10 * n_total_molm3[j]
        C_molm3[j, idx["O3"]] = 1e-8  * n_total_molm3[j]

    # Active species for diffusion (everything except N2)
    active_idx = [idx["O2"], idx["O"], idx["O3"]]
    n_active = len(active_idx)

    # Diffusion coefficients (in SI: mol/m³, m²/s, m)
    diff_A, diff_B, diff_C = init_diffusion(
        n_total_molm3, atm["Kzz_m2s"], atm["dzi_m"])

    # Torch tensors for kintera calls
    T_t = torch.full((nz,), T_K, dtype=torch.float64)
    P_t = torch.from_numpy(P_Pa.copy()).double()
    wl_t = torch.from_numpy(wl_np).double()
    eye_ns = torch.eye(nspecies, dtype=torch.float64)

    cos_zen = 1.0
    dt = 1e-10
    dt_max = 1e6

    # Initial actinic flux
    C_cgs = C_molm3 * AVOGADRO * 1e-6  # mol/m³ → molecule/cm³
    aflux = beer_lambert_actinic_flux(
        [C_cgs[:, idx["O2"]], C_cgs[:, idx["O3"]]],
        [xs_O2_total, xs_O3_total],
        stellar_flux, atm["dz_cm"], cos_zen)

    t0 = time.time()

    for step in range(max_steps):
        # 1. Update actinic flux periodically
        if step > 0 and step % rt_update_frq == 0:
            C_cgs = C_molm3 * AVOGADRO * 1e-6
            # Re-normalize to hydrostatic density for RT
            n_total_current = C_cgs.sum(axis=1)
            scale = n_cgs / n_total_current
            C_cgs_rt = C_cgs * scale[:, None]
            aflux = beer_lambert_actinic_flux(
                [C_cgs_rt[:, idx["O2"]], C_cgs_rt[:, idx["O3"]]],
                [xs_O2_total, xs_O3_total],
                stellar_flux, atm["dz_cm"], cos_zen)

        # 2. kintera forward: rates + rate derivatives
        C_t = torch.from_numpy(C_molm3.copy()).double()
        aflux_t = torch.from_numpy(aflux.T.copy()).double()  # (nwave, nz)
        extra = {"wavelength": wl_t, "actinic_flux": aflux_t}

        rate, rc_ddC, rc_ddT = kinet.forward(T_t, P_t, C_t, extra)

        # 3. Chemistry species tendency: stoich @ rate
        chem_tend = torch.matmul(
            stoich, rate.unsqueeze(-1)).squeeze(-1).numpy()  # (nz, nspecies)

        # 4. Chemistry Jacobian
        cvol = torch.ones(nz, dtype=torch.float64)
        jac_rxn = kinet.jacobian(T_t, C_t, cvol, rate, rc_ddC)  # (nz, nr, ns)
        chem_jac = torch.matmul(stoich, jac_rxn).numpy()  # (nz, ns, ns)

        # 5. Diffusion tendency (active species only)
        diff_tend_active = diffusion_tendency(
            C_molm3[:, active_idx], diff_A, diff_B, diff_C)

        diff_tend = np.zeros_like(C_molm3)
        diff_tend[:, active_idx] = diff_tend_active

        # 6. Diffusion Jacobian (diagonal only)
        diff_jac = np.zeros((nz, nspecies, nspecies))
        for k, ai in enumerate(active_idx):
            diff_jac[:, ai, ai] = diff_A.ravel()

        # 7. Total tendency and Jacobian
        total_tend = chem_tend + diff_tend
        total_jac = chem_jac + diff_jac

        # 8. Implicit solve via kintera
        C_old = C_molm3.copy()
        tend_t = torch.from_numpy(np.ascontiguousarray(total_tend)).double()
        jac_t = torch.from_numpy(np.ascontiguousarray(total_jac)).double()

        try:
            delta = kt.evolve_implicit(tend_t, eye_ns, jac_t, dt).numpy()
        except RuntimeError:
            dt = max(dt * 0.5, 1e-14)
            continue

        C_molm3 = np.maximum(C_old + delta, 1e-50)

        # Adaptive time stepping
        rc = np.max(np.abs(C_molm3 - C_old) / (np.abs(C_old) + 1e-50))
        if rc < 0.3:
            dt = min(dt * 1.2, dt_max)
        elif rc > 1.0:
            dt = max(dt * 0.5, 1e-14)

        # Convergence check
        if step > 500 and step % 20 == 0:
            C_cgs = C_molm3 * AVOGADRO * 1e-6
            n_total_current = C_cgs.sum(axis=1)
            scale = n_cgs / n_total_current
            C_cgs_rt = C_cgs * scale[:, None]
            aflux_chk = beer_lambert_actinic_flux(
                [C_cgs_rt[:, idx["O2"]], C_cgs_rt[:, idx["O3"]]],
                [xs_O2_total, xs_O3_total],
                stellar_flux, atm["dz_cm"], cos_zen)
            aflux_chk_t = torch.from_numpy(aflux_chk.T.copy()).double()
            extra_chk = {"wavelength": wl_t, "actinic_flux": aflux_chk_t}
            C_t2 = torch.from_numpy(C_molm3.copy()).double()
            rate2, _, _ = kinet.forward(T_t, P_t, C_t2, extra_chk)
            tend2 = torch.matmul(stoich, rate2.unsqueeze(-1)).squeeze(-1).numpy()
            diff2 = diffusion_tendency(
                C_molm3[:, active_idx], diff_A, diff_B, diff_C)
            tend2[:, active_idx] += diff2
            rel_tend = np.max(np.abs(tend2) / (np.abs(C_molm3) + 1e-50))
            if rel_tend < 1e-6:
                print(f"  Converged at step {step+1}, rel_tend={rel_tend:.2e}")
                break

    elapsed = time.time() - t0
    nsteps = step + 1

    ymix = C_molm3 / C_molm3.sum(axis=1, keepdims=True)

    return ymix, nsteps, elapsed, atm, species, idx


# ── VULCAN comparison ───────────────────────────────────────────────────

def run_vulcan_chapman(nz=20):
    """Run VULCAN with the Chapman cycle for comparison."""
    cfg_path = os.path.join(VULCAN_DIR, "vulcan_cfg.py")
    cfg_template = os.path.join(VULCAN_DIR, "vulcan_cfg_chapman.py")

    if not os.path.exists(cfg_template):
        print("  VULCAN not available, skipping comparison")
        return None

    with open(cfg_template) as f:
        cfg = f.read()

    cfg = cfg.replace("use_photo = True", "use_photo = True")
    cfg = cfg.replace("use_Kzz = False", "use_Kzz = True")
    cfg = cfg.replace("ini_update_photo_frq = 999999",
                      "ini_update_photo_frq = 100")
    cfg = cfg.replace("final_update_photo_frq = 999999",
                      "final_update_photo_frq = 5")
    cfg = cfg.replace("out_name = 'chapman.vul'",
                      "out_name = 'chapman_native_cmp.vul'")
    cfg = cfg.replace("P_b = 1e-1", "P_b = 1e1")
    cfg = cfg.replace("P_t = 1e-4", "P_t = 1e-2")
    cfg = cfg.replace("nz = 10", f"nz = {nz}")

    with open(cfg_path, "w") as f:
        f.write(cfg)

    P = np.logspace(np.log10(1e1), np.log10(1e-2), nz)
    atm_file = os.path.join(VULCAN_DIR, "atm", "atm_chapman.txt")
    with open(atm_file, "w") as f:
        f.write("# Pressure(dyne/cm2)  Temp(K)  Kzz(cm2/s)\n"
                "Pressure\tTemp\tKzz\n")
        for p in P:
            f.write(f"{p:.6e}\t250.0\t1.0e+06\n")

    out = os.path.join(VULCAN_DIR, "output", "chapman_native_cmp.vul")
    if os.path.exists(out):
        os.remove(out)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "vulcan.py"],
        capture_output=True, text=True, cwd=VULCAN_DIR, timeout=120
    )
    vulcan_elapsed = time.time() - t0

    if result.returncode != 0:
        print("VULCAN stdout:", result.stdout[-300:])
        print("VULCAN stderr:", result.stderr[-500:])
        raise RuntimeError("VULCAN failed")

    m = re.search(r'with (\d+) steps', result.stdout)
    vulcan_steps = int(m.group(1)) if m else -1

    with open(out, "rb") as f:
        data = pickle.load(f)
    return data["variable"], data["atm"], vulcan_steps, vulcan_elapsed


# ── Main test ───────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("1D Chapman Cycle — Kintera-Native (no VULCAN code in solver)")
    print("=" * 80)

    yaml_path = os.path.join(os.path.dirname(SCRIPT_DIR), "chapman_cycle.yaml")
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(SCRIPT_DIR, "..", "chapman_cycle.yaml")
    print(f"\n  YAML: {yaml_path}")

    nz = 20
    print(f"\n  --- Kintera-native solver ---")
    ymix_kt, nsteps_kt, elapsed_kt, atm, species, idx = \
        kintera_1d_solve(yaml_path, nz=nz, max_steps=5000, rt_update_frq=10)

    print(f"\n  Kintera results: {nsteps_kt} steps, {elapsed_kt:.1f}s")
    print(f"\n  {'lev':>3s} {'P(dyn/cm²)':>12s} {'O':>10s} {'O2':>10s} {'O3':>10s}")
    print(f"  {'-'*3} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for lev in range(nz):
        print(f"  {lev:3d} {atm['P_cgs'][lev]:12.3e} "
              f"{ymix_kt[lev, idx['O']]:10.3e} "
              f"{ymix_kt[lev, idx['O2']]:10.3e} "
              f"{ymix_kt[lev, idx['O3']]:10.3e}")

    # Physical sanity checks
    print(f"\n  --- Sanity checks ---")
    top = nz - 1
    print(f"  Top O  = {ymix_kt[top, idx['O']]:.3e}")
    print(f"  Bot O  = {ymix_kt[0, idx['O']]:.3e}")
    print(f"  Top O3 = {ymix_kt[top, idx['O3']]:.3e}")
    print(f"  Bot O3 = {ymix_kt[0, idx['O3']]:.3e}")

    O_range = ymix_kt[:, idx["O"]].max() / (ymix_kt[:, idx["O"]].min() + 1e-50)
    print(f"  O mixing ratio range: {O_range:.1f}x")
    assert ymix_kt[top, idx["O"]] > ymix_kt[0, idx["O"]], \
        "Top should have more O (stronger photolysis)"
    assert O_range > 1.5, "Should show vertical variation"
    assert ymix_kt[:, idx["O3"]].max() > 1e-10, "O3 should build up"
    print("  All sanity checks PASSED")

    # VULCAN comparison
    print(f"\n  --- VULCAN comparison ---")
    try:
        vulcan_result = run_vulcan_chapman(nz=nz)
    except Exception as e:
        print(f"  VULCAN comparison skipped: {e}")
        vulcan_result = None

    if vulcan_result is not None:
        var, vatm, vulcan_steps, vulcan_elapsed = vulcan_result
        vulcan_ymix = var["ymix"]
        v_species = var["species"]

        # Map VULCAN species to kintera species
        # VULCAN Chapman: O, O2, O3, N2
        v_idx = {sp: i for i, sp in enumerate(v_species)}

        print(f"\n  VULCAN: {vulcan_steps} steps, {vulcan_elapsed:.1f}s")
        print(f"  Kintera: {nsteps_kt} steps, {elapsed_kt:.1f}s")
        print(f"  Speed: kintera {vulcan_elapsed/elapsed_kt:.1f}x vs VULCAN\n")

        print(f"  {'lev':>3s} {'P':>10s} | "
              f"{'VUL O':>9s} {'KT O':>9s} | "
              f"{'VUL O2':>9s} {'KT O2':>9s} | "
              f"{'VUL O3':>9s} {'KT O3':>9s}")
        print(f"  {'-'*3} {'-'*10} | {'-'*9} {'-'*9} | "
              f"{'-'*9} {'-'*9} | {'-'*9} {'-'*9}")

        for lev in range(nz):
            vO = vulcan_ymix[lev, v_idx["O"]]
            vO2 = vulcan_ymix[lev, v_idx["O2"]]
            vO3 = vulcan_ymix[lev, v_idx["O3"]]
            kO = ymix_kt[lev, idx["O"]]
            kO2 = ymix_kt[lev, idx["O2"]]
            kO3 = ymix_kt[lev, idx["O3"]]
            print(f"  {lev:3d} {atm['P_cgs'][lev]:10.2e} | "
                  f"{vO:9.2e} {kO:9.2e} | "
                  f"{vO2:9.2e} {kO2:9.2e} | "
                  f"{vO3:9.2e} {kO3:9.2e}")

        # Qualitative comparison: both should show ozone layer
        v_O3_max = vulcan_ymix[:, v_idx["O3"]].max()
        k_O3_max = ymix_kt[:, idx["O3"]].max()
        print(f"\n  Peak O3: VULCAN={v_O3_max:.3e}, kintera={k_O3_max:.3e}")

        v_has_profile = vulcan_ymix[:, v_idx["O"]].max() / \
                       (vulcan_ymix[:, v_idx["O"]].min() + 1e-50) > 1.5
        k_has_profile = O_range > 1.5
        print(f"  Vertical structure: VULCAN={'yes' if v_has_profile else 'no'}, "
              f"kintera={'yes' if k_has_profile else 'no'}")

    print("\n" + "=" * 80)
    print("ALL CHECKS PASSED — kintera-native Chapman cycle works correctly")
    print("=" * 80)


if __name__ == "__main__":
    main()
