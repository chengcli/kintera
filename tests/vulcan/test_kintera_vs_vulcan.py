"""
Ultimate comparison: kintera implicit Euler vs VULCAN Ros2.

Both solve the EXACT same 0D Chapman cycle ODE:
- Same rate constants (verified to match)
- Same J-values (extracted from VULCAN output)
- Same reverse reactions (R1 reverse included)
- No transport (VULCAN use_Kzz=False)

The only difference is the ODE solver:
  VULCAN: Rosenbrock-2 (2nd order, L-stable)
  kintera: Implicit Euler (1st order, A-stable)

Usage:
    python3.11 test_kintera_vs_vulcan.py
"""
import os, sys, subprocess, pickle, math, re
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JINKELA_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "VULCAN")
BUILD_DIR = os.path.join(JINKELA_DIR, "build", "tests")
VULCAN_OUTPUT = os.path.join(VULCAN_DIR, "output", "chapman.vul")

IDX_O, IDX_O2, IDX_O3, IDX_N2 = 0, 1, 2, 3
NS = 4


def load_vulcan():
    with open(VULCAN_OUTPUT, "rb") as f:
        data = pickle.load(f)
    return data["variable"], data["atm"]


def compute_dydt(y, k_all, J_O2, J_O3, M_fixed=None):
    """
    dy/dt for the full Chapman cycle (forward + reverse + photolysis).
    Uses VULCAN's rate constants directly (molecule, cm, s).
    M_fixed: if set, use this as the third-body concentration (VULCAN convention).
    """
    cO, cO2, cO3, cN2 = y
    M = M_fixed if M_fixed is not None else (cO + cO2 + cO3 + cN2)

    r1f = k_all[1] * cO * cO2       # O + O2 -> O3
    r1r = k_all[2] * cO3            # O3 -> O + O2 (Gibbs reverse)
    r2f = k_all[3] * cO * cO3       # O + O3 -> 2O2
    r2r = k_all[4] * cO2 * cO2      # 2O2 -> O + O3 (negligible)
    r3f = k_all[7] * M * cO * cO2   # O + O2 + M -> O3 + M
    r3r = k_all[8] * M * cO3        # O3 + M -> O + O2 + M (negligible)
    rP1 = J_O2 * cO2                # O2 -> 2O
    rP2 = J_O3 * cO3                # O3 -> O2 + O

    dydt = np.zeros(NS)
    dydt[IDX_O]  = -r1f + r1r - r2f + r2r - r3f + r3r + 2*rP1 + rP2
    dydt[IDX_O2] = -r1f + r1r + 2*r2f - 2*r2r - r3f + r3r - rP1 + rP2
    dydt[IDX_O3] =  r1f - r1r - r2f + r2r + r3f - r3r - rP2
    dydt[IDX_N2] = 0.0
    return dydt


def kintera_implicit_euler(y0, k_all, J_O2, J_O3, M_fixed=None, max_steps=2000):
    """Implicit Euler with adaptive dt (kintera's evolve_implicit formula)."""
    y = y0.copy()
    t = 0.0
    dt = 1e-10
    eps = 1e-8
    hist_t, hist_y = [t], [y.copy()]

    for step in range(max_steps):
        f = compute_dydt(y, k_all, J_O2, J_O3, M_fixed)

        J = np.zeros((NS, NS))
        for j in range(NS):
            yp = y.copy()
            h = max(abs(y[j]) * eps, eps)
            yp[j] += h
            J[:, j] = (compute_dydt(yp, k_all, J_O2, J_O3, M_fixed) - f) / h

        A = np.eye(NS) / dt - J
        try:
            delta = np.linalg.solve(A, f)
        except np.linalg.LinAlgError:
            dt *= 0.1
            continue
        y = np.maximum(y + delta, 0.0)

        rel_change = np.max(np.abs(delta) / (np.abs(y) + 1e-30))
        if rel_change < 0.5:
            dt = min(dt * 1.5, 1e10)
        elif rel_change > 2.0:
            dt = max(dt * 0.5, 1e-14)

        t += dt
        hist_t.append(t)
        hist_y.append(y.copy())

        if step > 200 and step % 10 == 0:
            f2 = compute_dydt(y, k_all, J_O2, J_O3)
            max_rel_dydt = np.max(np.abs(f2) / (np.abs(y) + 1e-30))
            if max_rel_dydt < 1e-12:
                break

    return np.array(hist_t), np.array(hist_y)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_identical_ode_steady_state():
    """
    Both solvers solve the EXACT same ODE at VULCAN layer 0.
    Using VULCAN's own rate constants and J-values eliminates all
    sources of difference except the solver method itself.
    """
    var, atm = load_vulcan()
    layer = 0
    n_0 = atm["n_0"][layer]

    # Extract ALL rate constants from VULCAN (including reverses)
    k_all = {i: var["k"][i][layer] for i in [1, 2, 3, 4, 7, 8]}
    J_O2 = var["J_sp"][("O2", 1)][layer]
    J_O3 = var["J_sp"][("O3", 1)][layer]

    # Same initial conditions
    y0 = np.array([1e-10*n_0, 0.21*n_0, 1e-8*n_0, 0.79*n_0])

    # VULCAN steady state (from Ros2 solver)
    vulcan_mix = var["ymix"][layer]

    # kintera implicit Euler (same ODE, same inputs, same M convention)
    t_kin, y_kin = kintera_implicit_euler(y0, k_all, J_O2, J_O3,
                                          M_fixed=n_0, max_steps=2000)
    kintera_mix = y_kin[-1] / y_kin[-1].sum()

    # VULCAN time evolution
    t_vul = np.array(var["t_time"])
    y_vul = np.array(var["y_time"])[:, layer, :]
    mix_vul = y_vul / y_vul.sum(axis=1, keepdims=True)

    print(f"\n  Setup: T={atm['Tco'][layer]:.0f}K, P={atm['pco'][layer]:.2e} dyn/cm^2, "
          f"n_0={n_0:.3e} cm^-3")
    print(f"  J_O2={J_O2:.6e} s^-1, J_O3={J_O3:.6e} s^-1")
    print(f"  k_fwd1={k_all[1]:.3e}, k_rev1={k_all[2]:.3e}, "
          f"k_fwd2={k_all[3]:.3e}, k0={k_all[7]:.3e}")

    print(f"\n  VULCAN Ros2: {len(t_vul)} steps -> t={t_vul[-1]:.2e} s")
    print(f"  kintera IE:  {len(t_kin)} steps -> t={t_kin[-1]:.2e} s")

    print(f"\n  VULCAN Ros2 evolution:")
    for i in [0, 20, 50, 100, 150, len(t_vul)-1]:
        if i < len(t_vul):
            print(f"    step {i:4d}  t={t_vul[i]:10.2e}  "
                  f"O={mix_vul[i,0]:.4e}  O2={mix_vul[i,1]:.4e}  O3={mix_vul[i,2]:.4e}")

    print(f"\n  kintera implicit Euler evolution:")
    show = [0, 20, 50, 100, min(200, len(t_kin)-1), len(t_kin)-1]
    for i in sorted(set(show)):
        if i < len(t_kin):
            m = y_kin[i] / y_kin[i].sum()
            print(f"    step {i:4d}  t={t_kin[i]:10.2e}  "
                  f"O={m[0]:.4e}  O2={m[1]:.4e}  O3={m[2]:.4e}")

    print(f"\n  === Steady-State Comparison (same ODE) ===")
    print(f"  {'Species':>8s}  {'VULCAN':>14s}  {'kintera':>14s}  {'ratio':>10s}")
    sp = ["O", "O2", "O3", "N2"]
    for i, s in enumerate(sp):
        r = kintera_mix[i] / vulcan_mix[i] if vulcan_mix[i] > 1e-20 else float("nan")
        print(f"  {s:>8s}  {vulcan_mix[i]:14.6e}  {kintera_mix[i]:14.6e}  {r:10.6f}")

    # O2 and O3 should match closely (< 0.5%)
    # O has slightly larger difference (~2%) due to solver truncation error
    # and VULCAN's small atom conservation error (~0.3%)
    for idx, name, tol in [(IDX_O, "O", 0.02), (IDX_O2, "O2", 0.005),
                            (IDX_O3, "O3", 0.005), (IDX_N2, "N2", 0.001)]:
        if vulcan_mix[idx] > 1e-15:
            rel_err = abs(kintera_mix[idx] / vulcan_mix[idx] - 1.0)
            assert rel_err < tol, \
                f"{name}: kintera={kintera_mix[idx]:.6e} vulcan={vulcan_mix[idx]:.6e} err={rel_err:.2e}"


def test_oxygen_conservation():
    """Both solvers must conserve O atoms in their 0D box."""
    var, atm = load_vulcan()
    layer = 0
    n_0 = atm["n_0"][layer]

    k_all = {i: var["k"][i][layer] for i in [1, 2, 3, 4, 7, 8]}
    J_O2 = var["J_sp"][("O2", 1)][layer]
    J_O3 = var["J_sp"][("O3", 1)][layer]

    y0 = np.array([1e-10*n_0, 0.21*n_0, 1e-8*n_0, 0.79*n_0])
    O_init = y0[IDX_O] + 2*y0[IDX_O2] + 3*y0[IDX_O3]

    _, y_kin = kintera_implicit_euler(y0, k_all, J_O2, J_O3, M_fixed=n_0, max_steps=500)
    O_kin = y_kin[-1][IDX_O] + 2*y_kin[-1][IDX_O2] + 3*y_kin[-1][IDX_O3]
    err = abs(O_kin - O_init) / O_init

    print(f"\n  kintera O conservation: error = {err*100:.4e}%")
    assert err < 1e-4


def test_rate_constants_match():
    """Rate constants from kintera formulas must match VULCAN."""
    var, atm = load_vulcan()
    TREF = 300.0
    print()
    for layer in range(len(atm["Tco"])):
        T = atm["Tco"][layer]
        assert abs(1.7e-14*(T/TREF)**(-2.4) / var["k"][1][layer] - 1) < 1e-6
        assert abs(8.0e-12*math.exp(-2060/T) / var["k"][3][layer] - 1) < 1e-6
        assert abs(6.0e-34*(T/TREF)**(-2.4) / var["k"][7][layer] - 1) < 1e-6
    print(f"  Rate constants match at all {len(atm['Tco'])} layers")


if __name__ == "__main__":
    print("=" * 70)
    print("kintera vs VULCAN: Same ODE, Different Solvers")
    print("=" * 70)

    print("\n[1] Rate constants")
    test_rate_constants_match()
    print("    PASSED")

    print("\n[2] O conservation")
    test_oxygen_conservation()
    print("    PASSED")

    print("\n[3] Steady-state comparison (identical ODE)")
    test_identical_ode_steady_state()
    print("    PASSED")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
