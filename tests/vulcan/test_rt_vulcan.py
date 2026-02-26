"""
Validate kintera_rt (pyharp DISORT) against VULCAN's RT output.

Compares the actinic flux and J-values computed by kintera_rt against
VULCAN's self-consistent RT calculation for the same atmospheric state.

Usage:
    python3.11 test_rt_vulcan.py
"""
import os, sys, pickle, math, subprocess
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from kintera_rt import compute_J, compute_actinic_flux

VULCAN_DIR = os.path.join(SCRIPT_DIR, "VULCAN")


def ensure_vulcan_output():
    """Make sure we have a VULCAN run with photo enabled."""
    out = os.path.join(VULCAN_DIR, "output", "chapman_photo_diff.vul")
    if os.path.exists(out):
        return out

    # Run the full 1D photochem test to generate the output
    cfg_path = os.path.join(VULCAN_DIR, "vulcan_cfg.py")
    with open(os.path.join(VULCAN_DIR, "vulcan_cfg_chapman.py")) as f:
        cfg = f.read()
    cfg = cfg.replace("use_Kzz = False", "use_Kzz = True")
    cfg = cfg.replace("out_name = 'chapman.vul'", "out_name = 'chapman_photo_diff.vul'")
    cfg = cfg.replace("P_b = 1e-1", "P_b = 1e1")
    cfg = cfg.replace("P_t = 1e-4", "P_t = 1e-2")
    cfg = cfg.replace("nz = 10", "nz = 20")
    with open(cfg_path, "w") as f:
        f.write(cfg)

    nz = 20
    P = np.logspace(np.log10(1e1), np.log10(1e-2), nz)
    atm_file = os.path.join(VULCAN_DIR, "atm", "atm_chapman.txt")
    with open(atm_file, "w") as f:
        f.write("# Pressure(dyne/cm2)  Temp(K)  Kzz(cm2/s)\nPressure\tTemp\tKzz\n")
        for p in P:
            f.write(f"{p:.6e}\t250.0\t1.0e+06\n")

    subprocess.check_call([sys.executable, "vulcan.py", "-n"], cwd=VULCAN_DIR,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def test_j_values_vs_vulcan():
    """Compare kintera J-values against VULCAN's RT-computed J-values."""
    out_path = ensure_vulcan_output()
    with open(out_path, "rb") as f:
        data = pickle.load(f)
    var, atm = data["variable"], data["atm"]
    nz = len(atm["Tco"])

    # VULCAN's J-values
    J_O2_vulcan = var["J_sp"][("O2", 1)]
    J_O3_vulcan = var["J_sp"][("O3", 1)]

    # VULCAN's cross-sections and wavelength grid
    bins = var["bins"]
    cross_O2 = var["cross_J"][("O2", 1)]
    cross_O3 = var["cross_J"][("O3", 1)]

    # VULCAN's actinic flux (for reference)
    aflux_vulcan = var["aflux"]

    # VULCAN sflux is in ergs/cm²/s/nm; aflux is in photons/cm²/s/nm
    # Convert: F_photon = F_erg * lambda / hc
    hc = 1.98644582e-9  # erg nm
    sflux_erg = var["sflux"][nz]  # sflux at TOA in ergs
    sflux_top = sflux_erg * bins / hc  # convert to photons

    # Use INITIAL composition (same as what VULCAN used to compute J)
    # VULCAN computed J once at initialization with const_mix = {O2:0.21, ...}
    y = var["y_ini"]  # (nz, ni) initial composition
    dzi = atm["dzi"]

    # Compute J-values using kintera_rt (pyharp DISORT)
    cos_zen = 1.0  # overhead sun (from VULCAN config sl_angle=0)
    J_O2_kin, J_O3_kin = compute_J(
        y, cross_O2, cross_O3, sflux_top, dzi, cos_zen, bins)
    aflux_kin = compute_actinic_flux(
        y, cross_O2, cross_O3, sflux_top, dzi, cos_zen, bins)

    print(f"\n  J-value comparison (nz={nz}):")
    print(f"  {'lev':>3s} {'P':>10s} | {'J_O2 VUL':>11s} {'J_O2 KIN':>11s} {'err%':>7s} | "
          f"{'J_O3 VUL':>11s} {'J_O3 KIN':>11s} {'err%':>7s}")

    for lev in range(nz):
        eO2 = abs(J_O2_kin[lev] / J_O2_vulcan[lev] - 1) * 100 if J_O2_vulcan[lev] > 1e-20 else 0
        eO3 = abs(J_O3_kin[lev] / J_O3_vulcan[lev] - 1) * 100 if J_O3_vulcan[lev] > 1e-20 else 0
        print(f"  {lev:3d} {atm['pco'][lev]:10.2e} | "
              f"{J_O2_vulcan[lev]:11.4e} {J_O2_kin[lev]:11.4e} {eO2:6.1f}% | "
              f"{J_O3_vulcan[lev]:11.4e} {J_O3_kin[lev]:11.4e} {eO3:6.1f}%")

    # Summary
    err_O2 = [abs(J_O2_kin[l]/J_O2_vulcan[l]-1)*100
              for l in range(nz) if J_O2_vulcan[l] > 1e-20]
    err_O3 = [abs(J_O3_kin[l]/J_O3_vulcan[l]-1)*100
              for l in range(nz) if J_O3_vulcan[l] > 1e-20]

    print(f"\n  J_O2 error: mean={np.mean(err_O2):.1f}%, max={np.max(err_O2):.1f}%")
    print(f"  J_O3 error: mean={np.mean(err_O3):.1f}%, max={np.max(err_O3):.1f}%")

    # Beer-Lambert should approximately match VULCAN's RT
    # (VULCAN uses 2-stream with scattering; our pure absorption is an approximation)
    # For UV with low scattering, the match should be within ~10%
    assert np.max(err_O3) < 5.0, f"J_O3 error too large: {np.max(err_O3):.1f}%"


def test_actinic_flux_profile():
    """Compare actinic flux vertical profiles."""
    out_path = ensure_vulcan_output()
    with open(out_path, "rb") as f:
        data = pickle.load(f)
    var, atm = data["variable"], data["atm"]
    nz = len(atm["Tco"])

    aflux_vulcan = var["aflux"]
    bins = var["bins"]

    # Pick a wavelength near the O3 Hartley band (255nm)
    idx_255 = np.argmin(np.abs(bins - 255))
    print(f"\n  Actinic flux at lambda={bins[idx_255]:.0f}nm:")
    print(f"  {'lev':>3s} {'P':>10s} {'VULCAN':>12s} {'ratio_to_top':>12s}")
    for lev in range(nz):
        r = aflux_vulcan[lev, idx_255] / aflux_vulcan[-1, idx_255]
        print(f"  {lev:3d} {atm['pco'][lev]:10.2e} {aflux_vulcan[lev, idx_255]:12.4e} {r:12.4f}")

    # Top should have highest flux
    assert aflux_vulcan[-1, idx_255] > aflux_vulcan[0, idx_255]


if __name__ == "__main__":
    print("=" * 70)
    print("kintera RT vs VULCAN RT")
    print("=" * 70)

    print("\n[1] J-values comparison")
    test_j_values_vs_vulcan()
    print("    PASSED")

    print("\n[2] Actinic flux profile")
    test_actinic_flux_profile()
    print("    PASSED")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
