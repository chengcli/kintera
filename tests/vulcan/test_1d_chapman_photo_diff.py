"""
1D Chapman cycle + photolysis + eddy diffusion + self-consistent RT:
kintera vs VULCAN.

The full photochemical column with RT:
  dy/dt = chemistry(y) + photolysis(y, J(y)) + diffusion(y)

kintera computes actinic flux using pyharp DISORT (via kintera_rt)
and updates J-values during time stepping (self-consistent RT),
matching VULCAN's approach.

Usage:
    python3.11 test_1d_chapman_photo_diff.py
"""
import os, sys, subprocess, pickle, math, re
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from kintera_rt import compute_J

VULCAN_DIR = os.path.join(SCRIPT_DIR, "VULCAN")
TREF = 300.0
KB_CGS = 1.380649e-16


def run_vulcan_full_1d():
    """Run VULCAN with chemistry + photolysis + diffusion (full 1D)."""
    cfg_path = os.path.join(VULCAN_DIR, "vulcan_cfg.py")
    with open(os.path.join(VULCAN_DIR, "vulcan_cfg_chapman.py")) as f:
        cfg = f.read()

    # Photo + Kzz + self-consistent RT updates (full physics)
    cfg = cfg.replace("use_photo = True", "use_photo = True")
    cfg = cfg.replace("use_Kzz = False", "use_Kzz = True")
    cfg = cfg.replace("ini_update_photo_frq = 999999", "ini_update_photo_frq = 100")
    cfg = cfg.replace("final_update_photo_frq = 999999", "final_update_photo_frq = 5")
    cfg = cfg.replace("out_name = 'chapman.vul'", "out_name = 'chapman_photo_diff.vul'")
    # Moderate pressure for interesting photochemistry
    cfg = cfg.replace("P_b = 1e-1", "P_b = 1e1")
    cfg = cfg.replace("P_t = 1e-4", "P_t = 1e-2")
    cfg = cfg.replace("nz = 10", "nz = 20")

    with open(cfg_path, "w") as f:
        f.write(cfg)

    # Update atm profile
    nz = 20
    P = np.logspace(np.log10(1e1), np.log10(1e-2), nz)
    atm_file = os.path.join(VULCAN_DIR, "atm", "atm_chapman.txt")
    with open(atm_file, "w") as f:
        f.write("# Pressure(dyne/cm2)  Temp(K)  Kzz(cm2/s)\nPressure\tTemp\tKzz\n")
        for p in P:
            f.write(f"{p:.6e}\t250.0\t1.0e+06\n")

    out = os.path.join(VULCAN_DIR, "output", "chapman_photo_diff.vul")
    if os.path.exists(out):
        os.remove(out)

    result = subprocess.run(
        [sys.executable, "vulcan.py", "-n"],
        capture_output=True, text=True, cwd=VULCAN_DIR, timeout=120
    )

    if result.returncode != 0:
        print("VULCAN stdout:", result.stdout[-300:])
        print("VULCAN stderr:", result.stderr[-500:])
        raise RuntimeError("VULCAN failed")

    # Parse step count
    m = re.search(r'with (\d+) steps', result.stdout)
    vulcan_steps = int(m.group(1)) if m else -1

    with open(out, "rb") as f:
        data = pickle.load(f)
    return data["variable"], data["atm"], vulcan_steps


def compute_chem_dydt(y, k_all, J_O2, J_O3, nz, ni):
    """Chemistry + photolysis tendency at all levels."""
    dydt = np.zeros_like(y)
    for lev in range(nz):
        cO, cO2, cO3, cN2 = y[lev]
        M = y[lev].sum()

        r1f = k_all[1][lev] * cO * cO2
        r1r = k_all[2][lev] * cO3
        r2f = k_all[3][lev] * cO * cO3
        r2r = k_all[4][lev] * cO2 ** 2
        r3f = k_all[7][lev] * M * cO * cO2
        r3r = k_all[8][lev] * M * cO3
        rP1 = J_O2[lev] * cO2
        rP2 = J_O3[lev] * cO3

        dydt[lev, 0] = -r1f + r1r - r2f + r2r - r3f + r3r + 2*rP1 + rP2
        dydt[lev, 1] = -r1f + r1r + 2*r2f - 2*r2r - r3f + r3r - rP1 + rP2
        dydt[lev, 2] =  r1f - r1r - r2f + r2r + r3f - r3r - rP2
        dydt[lev, 3] = 0.0
    return dydt


def compute_diff_dydt(y, Kzz, dzi, nz, ni):
    """Diffusion tendency (kintera formula, matching VULCAN)."""
    n_tot = np.maximum(np.sum(y, axis=1), 1e-30)
    n_avg = (n_tot[1:] + n_tot[:-1]) / 2.0
    D = Kzz * n_avg / dzi

    tend = np.zeros_like(y)
    for j in range(nz):
        if j == 0:
            A = -D[0] / dzi[0] / n_tot[0]
            B =  D[0] / dzi[0] / n_tot[1]
            tend[j] = A * y[j] + B * y[j+1]
        elif j == nz - 1:
            A = -D[-1] / dzi[-1] / n_tot[-1]
            C_ = D[-1] / dzi[-1] / n_tot[-2]
            tend[j] = A * y[j] + C_ * y[j-1]
        else:
            dz_avg = (dzi[j-1] + dzi[j]) / 2.0
            A = -(D[j] + D[j-1]) / (dz_avg * n_tot[j])
            B = D[j] / (dz_avg * n_tot[j+1])
            C_ = D[j-1] / (dz_avg * n_tot[j-1])
            tend[j] = A * y[j] + B * y[j+1] + C_ * y[j-1]
    return tend


def kintera_1d_photo_diff_solve(y0, k_all, cross_O2, cross_O3, stellar_flux,
                                 Kzz, dzi, wavelengths, cos_zen, nz, ni,
                                 max_steps=5000, rt_update_frq=10):
    """
    Implicit Euler for chemistry + photolysis + diffusion with
    self-consistent RT (J updated every rt_update_frq steps via pyharp DISORT).
    """
    y = y0.copy()
    dt = 1e-10
    eps = 1e-8
    dt_max = 1e6

    J_O2, J_O3 = compute_J(
        y, cross_O2, cross_O3, stellar_flux, dzi, cos_zen, wavelengths)

    for step in range(max_steps):
        if step > 0 and step % rt_update_frq == 0:
            J_O2, J_O3 = compute_J(
                y, cross_O2, cross_O3, stellar_flux, dzi, cos_zen, wavelengths)

        f_chem = compute_chem_dydt(y, k_all, J_O2, J_O3, nz, ni)
        f_diff = compute_diff_dydt(y, Kzz, dzi, nz, ni)
        f_total = f_chem + f_diff

        N = nz * ni
        y_flat = y.flatten()
        f_flat = f_total.flatten()

        Jac = np.zeros((N, N))
        for col in range(N):
            yp = y_flat.copy()
            h = max(abs(yp[col]) * eps, eps)
            yp[col] += h
            yp_2d = yp.reshape(nz, ni)
            fp = (compute_chem_dydt(yp_2d, k_all, J_O2, J_O3, nz, ni) +
                  compute_diff_dydt(yp_2d, Kzz, dzi, nz, ni)).flatten()
            Jac[:, col] = (fp - f_flat) / h

        A = np.eye(N) / dt - Jac
        try:
            delta = np.linalg.solve(A, f_flat)
        except np.linalg.LinAlgError:
            dt *= 0.1
            continue

        y_flat_new = np.maximum(y_flat + delta, 1e-30)
        y = y_flat_new.reshape(nz, ni)

        rc = np.max(np.abs(delta) / (np.abs(y_flat) + 1e-30))
        if rc < 0.3:
            dt = min(dt * 1.2, dt_max)
        elif rc > 1.0:
            dt = max(dt * 0.5, 1e-14)

        if step > 500 and step % 20 == 0:
            J_O2_c, J_O3_c = compute_J(
                y, cross_O2, cross_O3, stellar_flux, dzi, cos_zen, wavelengths)
            f2 = (compute_chem_dydt(y, k_all, J_O2_c, J_O3_c, nz, ni) +
                  compute_diff_dydt(y, Kzz, dzi, nz, ni)).flatten()
            if np.max(np.abs(f2) / (np.abs(y.flatten()) + 1e-30)) < 1e-6:
                break

    return y, step + 1


def test_1d_full_photochem():
    """1D Chapman + photolysis + diffusion: kintera vs VULCAN."""
    var, atm, vulcan_steps = run_vulcan_full_1d()
    species = var["species"]
    nz = len(atm["Tco"])
    ni = len(species)

    print(f"\n  VULCAN 1D full photochemistry: nz={nz}, ni={ni}")
    print(f"  T = {atm['Tco'][0]:.0f}K (isothermal)")
    print(f"  P: {atm['pco'][0]:.1e} to {atm['pco'][-1]:.1e} dyn/cm^2")
    print(f"  Kzz: {atm['Kzz'][0]:.1e} cm^2/s")
    print(f"  VULCAN converged in {vulcan_steps} steps")

    # Extract rate constants from VULCAN
    k_all = {i: var["k"][i] for i in [1, 2, 3, 4, 7, 8]}

    # Cross-sections and wavelength grid from VULCAN
    bins = var["bins"]
    cross_O2 = var["cross_J"][("O2", 1)]
    cross_O3 = var["cross_J"][("O3", 1)]

    # Stellar flux at TOA (convert ergs to photons)
    hc = 1.98644582e-9
    sflux_top = var["sflux"][nz] * bins / hc

    cos_zen = 1.0  # overhead sun

    # Show VULCAN's J-value profile
    print(f"\n  VULCAN J-value profile (self-consistent RT):")
    print(f"  {'lev':>3s} {'P':>10s} {'J_O2':>11s} {'J_O3':>11s}")
    J_sp = var["J_sp"]
    for lev in [0, nz//4, nz//2, 3*nz//4, nz-1]:
        print(f"  {lev:3d} {atm['pco'][lev]:10.2e} "
              f"{J_sp[('O2',1)][lev]:11.4e} {J_sp[('O3',1)][lev]:11.4e}")

    Kzz = atm["Kzz"]
    dzi = atm["dzi"]

    # Initial conditions
    y0 = np.zeros((nz, ni))
    for lev in range(nz):
        n0 = atm["n_0"][lev]
        y0[lev, 0] = 1e-10 * n0
        y0[lev, 1] = 0.21 * n0
        y0[lev, 2] = 1e-8 * n0
        y0[lev, 3] = 0.79 * n0

    # VULCAN steady state
    vulcan_ymix = var["ymix"]

    # kintera solve WITH self-consistent RT
    print(f"\n  Running kintera 1D (chem + photo + diff + pyharp DISORT RT)...")
    y_kin, nsteps = kintera_1d_photo_diff_solve(
        y0, k_all, cross_O2, cross_O3, sflux_top,
        Kzz, dzi, bins, cos_zen, nz, ni,
        max_steps=3000, rt_update_frq=20)
    kintera_ymix = y_kin / y_kin.sum(axis=1, keepdims=True)
    print(f"  kintera converged in {nsteps} steps")

    # Compare profiles
    print(f"\n  {'lev':>3s} {'P':>10s} | {'VUL O':>9s} {'KIN O':>9s} {'err%':>6s} | "
          f"{'VUL O2':>9s} {'KIN O2':>9s} {'err%':>6s} | "
          f"{'VUL O3':>9s} {'KIN O3':>9s} {'err%':>6s}")
    print(f"  {'-'*3} {'-'*10} | {'-'*9} {'-'*9} {'-'*6} | "
          f"{'-'*9} {'-'*9} {'-'*6} | {'-'*9} {'-'*9} {'-'*6}")

    for lev in range(nz):
        vO, kO = vulcan_ymix[lev, 0], kintera_ymix[lev, 0]
        vO2, kO2 = vulcan_ymix[lev, 1], kintera_ymix[lev, 1]
        vO3, kO3 = vulcan_ymix[lev, 2], kintera_ymix[lev, 2]
        eO = abs(kO/vO - 1)*100 if vO > 1e-15 else 0
        eO2 = abs(kO2/vO2 - 1)*100 if vO2 > 1e-15 else 0
        eO3 = abs(kO3/vO3 - 1)*100 if vO3 > 1e-15 else 0
        print(f"  {lev:3d} {atm['pco'][lev]:10.2e} | "
              f"{vO:9.2e} {kO:9.2e} {eO:5.1f}% | "
              f"{vO2:9.2e} {kO2:9.2e} {eO2:5.1f}% | "
              f"{vO3:9.2e} {kO3:9.2e} {eO3:5.1f}%")

    # Summary statistics
    err_O2 = [abs(kintera_ymix[l,1]/vulcan_ymix[l,1]-1)*100
              for l in range(nz) if vulcan_ymix[l,1] > 1e-15]
    err_O3 = [abs(kintera_ymix[l,2]/vulcan_ymix[l,2]-1)*100
              for l in range(nz) if vulcan_ymix[l,2] > 1e-15]

    print(f"\n  Summary:")
    print(f"    O2 error: mean={np.mean(err_O2):.2f}%, max={np.max(err_O2):.2f}%")
    if err_O3:
        print(f"    O3 error: mean={np.mean(err_O3):.2f}%, max={np.max(err_O3):.2f}%")
    print(f"    VULCAN steps: {vulcan_steps}, kintera steps: {nsteps}")

    # Verify the qualitative vertical structure:
    # Top layers: more O (strong photolysis, low tau)
    # Bottom layers: more O2, O3 (weak photolysis, recombination)
    print(f"\n  Qualitative structure check:")
    print(f"    Top  (lev {nz-1}): O={kintera_ymix[-1,0]:.3e}, O2={kintera_ymix[-1,1]:.3e}")
    print(f"    Bot  (lev 0):   O={kintera_ymix[0,0]:.3e}, O2={kintera_ymix[0,1]:.3e}")

    if nz > 5:
        assert kintera_ymix[-1, 0] > kintera_ymix[0, 0], \
            "Top should have more O (stronger photolysis)"

    # Both should show non-trivial vertical profiles (not uniform)
    O_range = kintera_ymix[:, 0].max() / (kintera_ymix[:, 0].min() + 1e-30)
    print(f"    O mixing ratio range: {O_range:.1f}x (top/bottom)")
    assert O_range > 1.5, "Should show vertical variation from RT + diffusion"


if __name__ == "__main__":
    print("=" * 80)
    print("1D Chapman + Photolysis + Diffusion: kintera vs VULCAN")
    print("=" * 80)

    test_1d_full_photochem()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
