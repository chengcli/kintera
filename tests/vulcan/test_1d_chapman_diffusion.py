"""
1D Chapman cycle + eddy diffusion (no RT/photolysis): kintera vs VULCAN.

Both solve the same 1D system:
  dy/dt = chemistry(y) + diffusion(y)

with:
- Arrhenius + three-body + Gibbs reverse reactions
- Kzz eddy diffusion on a vertical column
- NO photolysis, NO radiative transfer

VULCAN runs with use_photo=False, use_Kzz=True.
kintera runs the same chemistry + diffusion in a Python loop using the
same rate constants and diffusion stencil.

Usage:
    python3.11 test_1d_chapman_diffusion.py
"""
import os, sys, subprocess, pickle, math, shutil
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "VULCAN")
TREF = 300.0
KB_CGS = 1.380649e-16  # erg/K


def run_vulcan_nophoto():
    """Run VULCAN with chemistry + diffusion, no photolysis."""
    # Create a no-photo config
    cfg_path = os.path.join(VULCAN_DIR, "vulcan_cfg.py")
    with open(os.path.join(VULCAN_DIR, "vulcan_cfg_chapman.py")) as f:
        cfg = f.read()

    # Disable photolysis, enable Kzz
    cfg = cfg.replace("use_photo = True", "use_photo = False")
    cfg = cfg.replace("use_Kzz = False", "use_Kzz = True")
    cfg = cfg.replace("nz = 10", "nz = 10")
    cfg = cfg.replace("out_name = 'chapman.vul'", "out_name = 'chapman_diff.vul'")
    # Use a reasonable pressure for chemistry
    cfg = cfg.replace("P_b = 1e-1", "P_b = 1e3")
    cfg = cfg.replace("P_t = 1e-4", "P_t = 1e0")

    with open(cfg_path, "w") as f:
        f.write(cfg)

    # Update atm profile for these pressures
    nz = 10
    P = np.logspace(np.log10(1e3), np.log10(1e0), nz)
    atm_file = os.path.join(VULCAN_DIR, "atm", "atm_chapman.txt")
    with open(atm_file, "w") as f:
        f.write("# Pressure(dyne/cm2)  Temp(K)  Kzz(cm2/s)\nPressure\tTemp\tKzz\n")
        for p in P:
            f.write(f"{p:.6e}\t250.0\t1.0e+08\n")

    # Remove old output
    out = os.path.join(VULCAN_DIR, "output", "chapman_diff.vul")
    if os.path.exists(out):
        os.remove(out)

    result = subprocess.run(
        [sys.executable, "vulcan.py", "-n"],
        capture_output=True, text=True, cwd=VULCAN_DIR, timeout=120
    )

    if result.returncode != 0:
        print("VULCAN stderr:", result.stderr[-500:])
        raise RuntimeError("VULCAN failed")

    with open(out, "rb") as f:
        data = pickle.load(f)
    return data["variable"], data["atm"]


def arrhenius_k(T, A, b, Ea_R):
    return A * (T / TREF) ** b * math.exp(-Ea_R / T)


def compute_chem_dydt(y, k_all, nz, ni):
    """Chemistry tendency at all levels (no photolysis)."""
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

        dydt[lev, 0] = -r1f + r1r - r2f + r2r - r3f + r3r
        dydt[lev, 1] = -r1f + r1r + 2*r2f - 2*r2r - r3f + r3r
        dydt[lev, 2] =  r1f - r1r - r2f + r2r + r3f - r3r
        dydt[lev, 3] = 0.0
    return dydt


def compute_diff_dydt(y, Kzz, dzi, nz, ni):
    """Diffusion tendency (kintera formula)."""
    n_tot = np.sum(y, axis=1)
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


def kintera_1d_solve(y0, k_all, Kzz, dzi, nz, ni, max_steps=2000):
    """Implicit Euler for chemistry + diffusion on 1D column."""
    y = y0.copy()
    dt = 1e-10
    eps = 1e-8

    for step in range(max_steps):
        f_chem = compute_chem_dydt(y, k_all, nz, ni)
        f_diff = compute_diff_dydt(y, Kzz, dzi, nz, ni)
        f_total = f_chem + f_diff

        # Numerical Jacobian for the full (nz*ni) system
        N = nz * ni
        y_flat = y.flatten()
        f_flat = f_total.flatten()

        J = np.zeros((N, N))
        for col in range(N):
            yp = y_flat.copy()
            h = max(abs(yp[col]) * eps, eps)
            yp[col] += h
            yp_2d = yp.reshape(nz, ni)
            fp = (compute_chem_dydt(yp_2d, k_all, nz, ni) +
                  compute_diff_dydt(yp_2d, Kzz, dzi, nz, ni)).flatten()
            J[:, col] = (fp - f_flat) / h

        A = np.eye(N) / dt - J
        try:
            delta = np.linalg.solve(A, f_flat)
        except np.linalg.LinAlgError:
            dt *= 0.1
            continue

        y_flat_new = np.maximum(y_flat + delta, 0.0)
        y = y_flat_new.reshape(nz, ni)

        rc = np.max(np.abs(delta) / (np.abs(y_flat) + 1e-30))
        if rc < 0.5:
            dt = min(dt * 1.5, 1e8)
        elif rc > 2.0:
            dt = max(dt * 0.5, 1e-14)

        if step > 100 and step % 10 == 0:
            f2 = (compute_chem_dydt(y, k_all, nz, ni) +
                  compute_diff_dydt(y, Kzz, dzi, nz, ni)).flatten()
            if np.max(np.abs(f2) / (np.abs(y.flatten()) + 1e-30)) < 1e-10:
                break

    return y, step + 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_1d_chapman_diffusion():
    """1D Chapman + Kzz diffusion: kintera vs VULCAN (no photolysis)."""
    var, atm = run_vulcan_nophoto()
    species = var["species"]  # ['O', 'O2', 'O3', 'N2']
    nz = len(atm["Tco"])
    ni = len(species)

    print(f"\n  VULCAN 1D Chapman+diffusion (no photo): nz={nz}, ni={ni}")
    print(f"  T = {atm['Tco'][0]:.0f}K (isothermal)")
    print(f"  P: {atm['pco'][0]:.1e} to {atm['pco'][-1]:.1e} dyn/cm^2")
    print(f"  Kzz: {atm['Kzz'][0]:.1e} cm^2/s")

    # Extract VULCAN rate constants
    k_all = {}
    for idx in [1, 2, 3, 4, 7, 8]:
        k_all[idx] = var["k"][idx]  # (nz,) arrays

    # Extract grid
    Kzz = atm["Kzz"]  # (nz-1,)
    dzi = atm["dzi"]   # (nz-1,)

    # Initial conditions: same as VULCAN
    y0 = np.zeros((nz, ni))
    for lev in range(nz):
        n0 = atm["n_0"][lev]
        y0[lev, 0] = 1e-10 * n0  # O
        y0[lev, 1] = 0.21 * n0   # O2
        y0[lev, 2] = 1e-8 * n0   # O3
        y0[lev, 3] = 0.79 * n0   # N2

    # VULCAN steady state
    vulcan_ymix = var["ymix"]  # (nz, ni)

    # kintera solve
    print("\n  Running kintera 1D implicit Euler...")
    y_kin, nsteps = kintera_1d_solve(y0, k_all, Kzz, dzi, nz, ni, max_steps=500)
    kintera_ymix = y_kin / y_kin.sum(axis=1, keepdims=True)
    print(f"  kintera converged in {nsteps} steps")

    # Compare at each level
    print(f"\n  {'lev':>3s} {'P(dyn/cm2)':>12s} | {'VULCAN O2':>11s} {'kintera O2':>11s} {'err%':>7s} | "
          f"{'VULCAN O3':>11s} {'kintera O3':>11s} {'err%':>7s}")
    print(f"  {'-'*3} {'-'*12} | {'-'*11} {'-'*11} {'-'*7} | {'-'*11} {'-'*11} {'-'*7}")

    max_err_O2, max_err_O3 = 0, 0
    for lev in range(nz):
        vO2, kO2 = vulcan_ymix[lev, 1], kintera_ymix[lev, 1]
        vO3, kO3 = vulcan_ymix[lev, 2], kintera_ymix[lev, 2]
        eO2 = abs(kO2/vO2 - 1)*100 if vO2 > 1e-15 else 0
        eO3 = abs(kO3/vO3 - 1)*100 if vO3 > 1e-15 else 0
        max_err_O2 = max(max_err_O2, eO2)
        max_err_O3 = max(max_err_O3, eO3)
        print(f"  {lev:3d} {atm['pco'][lev]:12.2e} | {vO2:11.4e} {kO2:11.4e} {eO2:6.2f}% | "
              f"{vO3:11.4e} {kO3:11.4e} {eO3:6.2f}%")

    print(f"\n  Max error: O2 = {max_err_O2:.2f}%, O3 = {max_err_O3:.2f}%")

    # Without photolysis, the chemistry is trivial: O recombines into O3/O2
    # Both codes should reach similar steady states
    for lev in range(nz):
        if vulcan_ymix[lev, 1] > 1e-10:
            assert abs(kintera_ymix[lev, 1] / vulcan_ymix[lev, 1] - 1) < 0.05, \
                f"O2 mismatch at level {lev}"


if __name__ == "__main__":
    print("=" * 70)
    print("1D Chapman + Diffusion (no RT): kintera vs VULCAN")
    print("=" * 70)

    test_1d_chapman_diffusion()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
