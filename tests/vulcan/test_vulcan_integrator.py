"""
VULCAN Ros2 vs kintera Implicit Euler: Integrator Comparison.

VULCAN solves the Chapman cycle with its native Ros2 (Rosenbrock-2) solver.
kintera solves the same ODE (same rate constants, same J-values, same initial
conditions) with implicit Euler. We compare the time evolution and steady states.

Usage:
    python3.11 test_vulcan_integrator.py           # from tests/vulcan/
    python3.11 -m pytest test_vulcan_integrator.py -v
"""
import os, sys, math, pickle, subprocess
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "VULCAN")
VULCAN_OUTPUT = os.path.join(VULCAN_DIR, "output", "chapman.vul")

IDX_O, IDX_O2, IDX_O3, IDX_N2 = 0, 1, 2, 3
NS = 4

# ---------------------------------------------------------------------------
# VULCAN data loading
# ---------------------------------------------------------------------------
def ensure_vulcan_output():
    if os.path.exists(VULCAN_OUTPUT):
        return
    import shutil
    shutil.copy2(os.path.join(VULCAN_DIR, "vulcan_cfg_chapman.py"),
                 os.path.join(VULCAN_DIR, "vulcan_cfg.py"))
    subprocess.check_call([sys.executable, "make_chem_funs.py"], cwd=VULCAN_DIR)
    subprocess.check_call([sys.executable, "vulcan.py", "-n"], cwd=VULCAN_DIR)

def load_vulcan():
    ensure_vulcan_output()
    with open(VULCAN_OUTPUT, "rb") as f:
        data = pickle.load(f)
    return data["variable"], data["atm"]

# ---------------------------------------------------------------------------
# kintera-style chemistry (same formulas as C++ code)
# ---------------------------------------------------------------------------
def compute_dydt(y, k_fwd, k_rev, J_O2, J_O3):
    """
    Compute dy/dt for the Chapman cycle using mass-action kinetics.
    Uses the same reactions and reverse rates as VULCAN.

    Reactions (VULCAN indexing):
      k[1]: O + O2 -> O3           (fwd R1)
      k[2]: O3 -> O + O2           (rev R1, thermodynamic)
      k[3]: O + O3 -> 2 O2         (fwd R2)
      k[4]: 2 O2 -> O + O3         (rev R2, negligible)
      k[7]: O + O2 + M -> O3 + M   (fwd R3, three-body)
      k[8]: O3 + M -> O + O2 + M   (rev R3, negligible)
      k[9]: O2 -> 2 O              (photolysis)
      k[11]: O3 -> O2 + O          (photolysis)
    """
    cO, cO2, cO3, cN2 = y
    M = cO + cO2 + cO3 + cN2

    r1f = k_fwd[1] * cO * cO2
    r1r = k_rev[2] * cO3
    r2f = k_fwd[3] * cO * cO3
    r2r = k_rev[4] * cO2 * cO2
    r3f = k_fwd[7] * M * cO * cO2
    r3r = k_rev[8] * M * cO3
    rP1 = J_O2 * cO2
    rP2 = J_O3 * cO3

    dydt = np.zeros(NS)
    dydt[IDX_O]  = -r1f + r1r - r2f + r2r - r3f + r3r + 2*rP1 + rP2
    dydt[IDX_O2] = -r1f + r1r + 2*r2f - 2*r2r - r3f + r3r - rP1 + rP2
    dydt[IDX_O3] =  r1f - r1r - r2f + r2r + r3f - r3r - rP2
    dydt[IDX_N2] = 0.0
    return dydt


def kintera_implicit_euler(y0, k_fwd, k_rev, J_O2, J_O3, max_steps=2000):
    """
    Implicit Euler with adaptive dt -- kintera's evolve_implicit formula:
        (I/dt - J) * delta = f(y)
    where J is the numerical Jacobian of dy/dt.

    Uses aggressive dt growth (similar to VULCAN's approach) to reach
    steady state in a reasonable number of steps.
    """
    y = y0.copy()
    t = 0.0
    dt = 1e-10
    eps_jac = 1e-8
    history_t = [t]
    history_y = [y.copy()]

    for step in range(max_steps):
        f = compute_dydt(y, k_fwd, k_rev, J_O2, J_O3)

        # Numerical Jacobian
        J = np.zeros((NS, NS))
        for j in range(NS):
            yp = y.copy()
            h = max(abs(y[j]) * eps_jac, eps_jac)
            yp[j] += h
            fp = compute_dydt(yp, k_fwd, k_rev, J_O2, J_O3)
            J[:, j] = (fp - f) / h

        # Solve (I/dt - J) * delta = f
        A = np.eye(NS) / dt - J
        delta = np.linalg.solve(A, f)
        y_new = np.maximum(y + delta, 0.0)

        # Adaptive dt based on relative change (implicit Euler is A-stable)
        rel_change = np.max(np.abs(delta) / (np.abs(y) + 1e-30))
        if rel_change < 0.5:
            dt *= 1.5
        elif rel_change > 2.0:
            dt *= 0.5

        y = y_new
        t += dt
        history_t.append(t)
        history_y.append(y.copy())

        # Convergence: dy/dt is small relative to y
        if step > 100:
            f_final = compute_dydt(y, k_fwd, k_rev, J_O2, J_O3)
            max_dydt = np.max(np.abs(f_final) / (np.abs(y) + 1e-30))
            if max_dydt < 1e-10:
                break

    return np.array(history_t), np.array(history_y)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_steady_state_match():
    """
    Both VULCAN (Ros2) and kintera (implicit Euler) must converge to the
    same steady-state mixing ratios starting from the same initial conditions.
    """
    var, atm = load_vulcan()
    layer = 0
    n_0 = atm["n_0"][layer]
    species = var["species"]

    # Extract VULCAN rate constants
    k_fwd = {i: var["k"][i][layer] for i in [1, 3, 7]}
    k_rev = {i: var["k"][i][layer] for i in [2, 4, 8]}
    J_O2 = var["J_sp"][("O2", 1)][layer]
    J_O3 = var["J_sp"][("O3", 1)][layer]

    # Same initial conditions
    y0 = np.array([1e-10*n_0, 0.21*n_0, 1e-8*n_0, 0.79*n_0])

    # VULCAN steady state
    vulcan_ymix = var["ymix"][layer]

    # kintera implicit Euler
    t_kin, y_kin = kintera_implicit_euler(y0, k_fwd, k_rev, J_O2, J_O3,
                                          max_steps=1000)
    kintera_ymix = y_kin[-1] / y_kin[-1].sum()

    print(f"\n  Layer {layer}: T={atm['Tco'][layer]:.0f}K, "
          f"P={atm['pco'][layer]:.2e} dyn/cm^2")
    print(f"  VULCAN Ros2: {len(var['t_time'])} steps, "
          f"t_final={var['t_time'][-1]:.2e} s")
    print(f"  kintera impl Euler: {len(t_kin)} steps, "
          f"t_final={t_kin[-1]:.2e} s")

    print(f"\n  {'Species':>6s}  {'VULCAN Ros2':>14s}  {'kintera IE':>14s}  {'ratio':>10s}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*10}")
    sp_names = ["O", "O2", "O3", "N2"]
    for i, sp in enumerate(sp_names):
        r = kintera_ymix[i] / vulcan_ymix[i] if vulcan_ymix[i] > 1e-20 else float("nan")
        print(f"  {sp:>6s}  {vulcan_ymix[i]:14.6e}  {kintera_ymix[i]:14.6e}  {r:10.6f}")

    # Assert match within 5% for active species
    for idx, sp in [(IDX_O, "O"), (IDX_O2, "O2"), (IDX_O3, "O3")]:
        if vulcan_ymix[idx] > 1e-15:
            rel_err = abs(kintera_ymix[idx] / vulcan_ymix[idx] - 1.0)
            assert rel_err < 0.05, \
                f"{sp} mismatch: kintera={kintera_ymix[idx]:.4e} " \
                f"vulcan={vulcan_ymix[idx]:.4e} rel_err={rel_err:.2e}"

    # N2 must be conserved
    assert abs(kintera_ymix[IDX_N2] / vulcan_ymix[IDX_N2] - 1.0) < 1e-3


def test_time_evolution_comparison():
    """
    Compare the time evolution trajectories of VULCAN and kintera.
    Both should show: O rises fast, O2 drops, O3 builds up, then all plateau.
    """
    var, atm = load_vulcan()
    layer = 0
    n_0 = atm["n_0"][layer]

    k_fwd = {i: var["k"][i][layer] for i in [1, 3, 7]}
    k_rev = {i: var["k"][i][layer] for i in [2, 4, 8]}
    J_O2 = var["J_sp"][("O2", 1)][layer]
    J_O3 = var["J_sp"][("O3", 1)][layer]

    y0 = np.array([1e-10*n_0, 0.21*n_0, 1e-8*n_0, 0.79*n_0])

    # VULCAN time evolution at layer 0
    t_vul = np.array(var["t_time"])
    y_vul = np.array(var["y_time"])[:, layer, :]
    mix_vul = y_vul / y_vul.sum(axis=1, keepdims=True)

    # kintera time evolution
    t_kin, y_kin = kintera_implicit_euler(y0, k_fwd, k_rev, J_O2, J_O3,
                                          max_steps=1000)
    mix_kin = y_kin / y_kin.sum(axis=1, keepdims=True)

    print(f"\n  VULCAN Ros2 evolution (layer 0):")
    print(f"  {'step':>6s}  {'t (s)':>12s}  {'x_O':>10s}  {'x_O2':>10s}  {'x_O3':>10s}")
    for i in [0, 10, 50, 100, 150, len(t_vul)-1]:
        if i < len(t_vul):
            print(f"  {i:6d}  {t_vul[i]:12.2e}  {mix_vul[i,0]:10.4e}  "
                  f"{mix_vul[i,1]:10.4e}  {mix_vul[i,2]:10.4e}")

    print(f"\n  kintera implicit Euler evolution (layer 0):")
    print(f"  {'step':>6s}  {'t (s)':>12s}  {'x_O':>10s}  {'x_O2':>10s}  {'x_O3':>10s}")
    show_steps = [0, 10, 50, 100, 200, len(t_kin)-1]
    for i in show_steps:
        if i < len(t_kin):
            print(f"  {i:6d}  {t_kin[i]:12.2e}  {mix_kin[i,0]:10.4e}  "
                  f"{mix_kin[i,1]:10.4e}  {mix_kin[i,2]:10.4e}")

    # Both should show O increasing from ~0 to ~0.01
    assert mix_vul[-1, IDX_O] > 0.005, "VULCAN: O should increase"
    assert mix_kin[-1, IDX_O] > 0.005, "kintera: O should increase"

    # O3 should build up
    assert mix_vul[-1, IDX_O3] > 0.01, "VULCAN: O3 should build up"
    assert mix_kin[-1, IDX_O3] > 0.01, "kintera: O3 should build up"


def test_oxygen_conservation():
    """
    Oxygen conservation: kintera (0D box) should conserve exactly.
    VULCAN (1D with transport) conserves globally across all layers,
    not per-layer, so we check global VULCAN conservation.
    """
    var, atm = load_vulcan()
    layer = 0
    n_0 = atm["n_0"][layer]

    k_fwd = {i: var["k"][i][layer] for i in [1, 3, 7]}
    k_rev = {i: var["k"][i][layer] for i in [2, 4, 8]}
    J_O2 = var["J_sp"][("O2", 1)][layer]
    J_O3 = var["J_sp"][("O3", 1)][layer]

    # kintera 0D box conservation
    y0 = np.array([1e-10*n_0, 0.21*n_0, 1e-8*n_0, 0.79*n_0])
    O_init_kin = y0[IDX_O] + 2*y0[IDX_O2] + 3*y0[IDX_O3]

    _, y_kin = kintera_implicit_euler(y0, k_fwd, k_rev, J_O2, J_O3, max_steps=500)
    O_final_kin = y_kin[-1][IDX_O] + 2*y_kin[-1][IDX_O2] + 3*y_kin[-1][IDX_O3]
    err_kin = abs(O_final_kin - O_init_kin) / O_init_kin

    # VULCAN global conservation (sum over all layers)
    y_vul_init = np.array(var["y_time"])[0]   # (nz, ni)
    y_vul_final = np.array(var["y_time"])[-1]  # (nz, ni)
    O_glob_init = (y_vul_init[:,IDX_O] + 2*y_vul_init[:,IDX_O2] + 3*y_vul_init[:,IDX_O3]).sum()
    O_glob_final = (y_vul_final[:,IDX_O] + 2*y_vul_final[:,IDX_O2] + 3*y_vul_final[:,IDX_O3]).sum()
    err_vul_glob = abs(O_glob_final - O_glob_init) / O_glob_init

    print(f"\n  Oxygen conservation:")
    print(f"    kintera (0D box):     error = {err_kin*100:.4e}%")
    print(f"    VULCAN (1D, global):  error = {err_vul_glob*100:.4e}%")

    assert err_kin < 1e-6, f"kintera O conservation: {err_kin*100:.4f}%"
    assert err_vul_glob < 0.05, f"VULCAN global O conservation: {err_vul_glob*100:.4f}%"


def test_rate_constants_match():
    """Rate constants from kintera formulas must match VULCAN's stored values."""
    var, atm = load_vulcan()
    TREF = 300.0
    print()
    for layer in range(len(atm["Tco"])):
        T = atm["Tco"][layer]
        k1 = 1.7e-14 * (T/TREF)**(-2.4)
        k3 = 8.0e-12 * math.exp(-2060.0/T)
        k0 = 6.0e-34 * (T/TREF)**(-2.4)
        assert abs(k1/var["k"][1][layer] - 1) < 1e-6
        assert abs(k3/var["k"][3][layer] - 1) < 1e-6
        assert abs(k0/var["k"][7][layer] - 1) < 1e-6
    print(f"  Rate constants match at all {len(atm['Tco'])} layers (< 1e-6)")


if __name__ == "__main__":
    print("=" * 70)
    print("VULCAN Ros2 vs kintera Implicit Euler: Chapman Cycle")
    print("=" * 70)

    print("\n[1] Rate constants match")
    test_rate_constants_match()
    print("    PASSED")

    print("\n[2] Oxygen conservation")
    test_oxygen_conservation()
    print("    PASSED")

    print("\n[3] Time evolution comparison")
    test_time_evolution_comparison()
    print("    PASSED")

    print("\n[4] Steady-state match")
    test_steady_state_match()
    print("    PASSED")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
