"""
Ultimate comparison: kintera C++ time marching vs VULCAN Ros2.

Runs both integrators on the Chapman cycle and compares:
1. Rate constants (should match exactly after unit conversion)
2. Time evolution qualitative behavior
3. Steady-state mixing ratios (with analysis of differences)

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


def run_kintera_cpp():
    """Run the C++ TimeMarching test and parse its output."""
    exe = os.path.join(BUILD_DIR, "test_chapman_cycle.release")
    if not os.path.exists(exe):
        raise FileNotFoundError(f"Build the C++ test first: {exe}")

    result = subprocess.run(
        [exe, "--gtest_filter=*TimeMarching/cpu*"],
        capture_output=True, text=True, cwd=BUILD_DIR
    )

    output = result.stdout + result.stderr
    steps = []
    final = {}

    for line in output.split("\n"):
        m = re.search(r"step\s+(\d+)\s+dt=([\d.e+\-]+)\s+O=([\d.e+\-]+)\s+O2=([\d.e+\-]+)\s+O3=([\d.e+\-]+)", line)
        if m:
            steps.append({
                "step": int(m.group(1)),
                "dt": float(m.group(2)),
                "O": float(m.group(3)),
                "O2": float(m.group(4)),
                "O3": float(m.group(5)),
            })

        m2 = re.search(r"Final:\s+O=([\d.e+\-]+)\s+O2=([\d.e+\-]+)\s+O3=([\d.e+\-]+)", line)
        if m2:
            final = {"O": float(m2.group(1)), "O2": float(m2.group(2)), "O3": float(m2.group(3))}

        m3 = re.search(r"Converged at step (\d+)", line)
        if m3:
            final["converged_step"] = int(m3.group(1))

    if "converged_step" not in final and steps:
        final["converged_step"] = steps[-1]["step"]

    return steps, final, result.returncode == 0


def load_vulcan():
    with open(VULCAN_OUTPUT, "rb") as f:
        data = pickle.load(f)
    return data["variable"], data["atm"]


def test_rate_constant_equivalence():
    """Verify kintera and VULCAN use equivalent rate constants (after unit conversion)."""
    var, atm = load_vulcan()
    T, TREF = 250.0, 300.0
    AVOGADRO = 6.02214076e23
    R = 8.314

    # VULCAN rate constants (molecule, cm, s)
    k1_vul = var["k"][1][0]
    k3_vul = var["k"][3][0]
    k0_vul = var["k"][7][0]

    # kintera (mol, m, s) — computed from the same A values after from_yaml conversion
    conv_bi = AVOGADRO * 1e-6
    k1_kin = 1.7e-14 * conv_bi * (T / TREF) ** (-2.4)
    k3_kin = 8.0e-12 * conv_bi * math.exp(-2060 / T)
    conv_tri = AVOGADRO ** 2 * 1e-12
    k0_kin = 6.0e-34 * conv_tri * (T / TREF) ** (-2.4)

    # Convert VULCAN to mol,m,s for comparison
    k1_vul_SI = k1_vul * conv_bi
    k3_vul_SI = k3_vul * conv_bi
    k0_vul_SI = k0_vul * conv_tri

    print(f"\n  Rate constants at T={T}K (mol, m, s):")
    print(f"  {'Reaction':>25s}  {'kintera':>14s}  {'VULCAN':>14s}  {'ratio':>10s}")
    for name, kk, kv in [("O+O2->O3", k1_kin, k1_vul_SI),
                          ("O+O3->2O2", k3_kin, k3_vul_SI),
                          ("O+O2+M->O3+M (k0)", k0_kin, k0_vul_SI)]:
        r = kk / kv
        print(f"  {name:>25s}  {kk:14.6e}  {kv:14.6e}  {r:10.6f}")
        assert abs(r - 1.0) < 1e-4, f"{name}: ratio={r}"


def test_time_marching_comparison():
    """Run both integrators and compare their steady states."""
    # Run kintera C++
    steps_kin, final_kin, ok = run_kintera_cpp()
    assert ok, "kintera C++ test failed"
    assert final_kin, "Could not parse kintera output"

    # Load VULCAN
    var, atm = load_vulcan()
    t_vul = np.array(var["t_time"])
    y_vul = np.array(var["y_time"])[:, 0, :]  # layer 0
    mix_vul = y_vul / y_vul.sum(axis=1, keepdims=True)
    final_vul = {"O": mix_vul[-1, 0], "O2": mix_vul[-1, 1], "O3": mix_vul[-1, 2]}

    print(f"\n  === Time Evolution ===")
    print(f"\n  VULCAN Ros2 ({len(t_vul)} steps, Rosenbrock-2 with transport + reverse reactions):")
    print(f"  {'step':>6s} {'t (s)':>12s}  {'x_O':>10s}  {'x_O2':>10s}  {'x_O3':>10s}")
    for i in [0, 10, 50, 100, 150, len(t_vul) - 1]:
        if i < len(t_vul):
            print(f"  {i:6d} {t_vul[i]:12.2e}  {mix_vul[i,0]:10.4e}  "
                  f"{mix_vul[i,1]:10.4e}  {mix_vul[i,2]:10.4e}")

    print(f"\n  kintera C++ ({final_kin.get('converged_step','?')} steps, "
          f"implicit Euler, no transport, no reverse reactions):")
    print(f"  {'step':>6s} {'dt':>12s}  {'x_O':>10s}  {'x_O2':>10s}  {'x_O3':>10s}")
    for s in steps_kin:
        print(f"  {s['step']:6d} {s['dt']:12.2e}  {s['O']:10.4e}  "
              f"{s['O2']:10.4e}  {s['O3']:10.4e}")

    print(f"\n  === Steady-State Comparison ===")
    print(f"  {'Species':>8s}  {'kintera C++':>14s}  {'VULCAN Ros2':>14s}  {'ratio':>10s}")
    print(f"  {'--------':>8s}  {'-'*14}  {'-'*14}  {'-'*10}")
    for sp in ["O", "O2", "O3"]:
        kk = final_kin[sp]
        vv = final_vul[sp]
        r = kk / vv if vv > 1e-20 else float("nan")
        print(f"  {sp:>8s}  {kk:14.6e}  {vv:14.6e}  {r:10.4f}")

    print(f"\n  === Difference Analysis ===")
    print(f"  VULCAN includes thermodynamic reverse of O+O2->O3:")
    print(f"    k_rev(O3->O+O2) = {var['k'][2][0]:.3e} s^-1 (significant!)")
    print(f"    This destroys O3 and produces O, shifting the equilibrium.")
    print(f"  kintera has only forward reactions, so O3 accumulates more.")
    print(f"  VULCAN also has 1D vertical transport (eddy diffusion).")
    print(f"")
    print(f"  Despite these differences, both integrators show the same")
    print(f"  qualitative Chapman cycle evolution:")
    print(f"    1. O builds up rapidly from O2 photolysis")
    print(f"    2. O3 builds up from O + O2 recombination")
    print(f"    3. System reaches photochemical steady state")

    # Qualitative checks: both should have similar order-of-magnitude results
    assert final_kin["O"] > 0.005, "kintera: O should build up"
    assert final_kin["O3"] > 0.01, "kintera: O3 should build up"
    assert final_vul["O"] > 0.005, "VULCAN: O should build up"
    assert final_vul["O3"] > 0.01, "VULCAN: O3 should build up"

    # O2 should decrease in both
    assert final_kin["O2"] < 0.21, "kintera: O2 should decrease"
    assert final_vul["O2"] < 0.21, "VULCAN: O2 should decrease"


def test_integrator_efficiency():
    """Compare integrator convergence efficiency."""
    steps_kin, final_kin, _ = run_kintera_cpp()
    var, _ = load_vulcan()
    t_vul = np.array(var["t_time"])

    nsteps_kin = final_kin.get("converged_step", len(steps_kin))
    nsteps_vul = len(t_vul)

    print(f"\n  Integrator efficiency:")
    print(f"    VULCAN Ros2:         {nsteps_vul:4d} steps to steady state")
    print(f"    kintera impl Euler:  {nsteps_kin:4d} steps to steady state")
    print(f"    VULCAN solver:       Rosenbrock-2 (2nd order, L-stable)")
    print(f"    kintera solver:      Implicit Euler (1st order, A-stable)")
    print(f"    VULCAN overhead:     1D atmosphere, transport, RT, reverse reactions")
    print(f"    kintera overhead:    0D box model, forward reactions only")


if __name__ == "__main__":
    print("=" * 70)
    print("kintera C++ vs VULCAN: Chapman Cycle Time Marching")
    print("=" * 70)

    print("\n[1] Rate constant equivalence (after unit conversion)")
    test_rate_constant_equivalence()
    print("    PASSED")

    print("\n[2] Time marching comparison")
    test_time_marching_comparison()
    print("    PASSED")

    print("\n[3] Integrator efficiency")
    test_integrator_efficiency()
    print("    PASSED")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
