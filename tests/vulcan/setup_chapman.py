"""
Generate all synthetic data files for the VULCAN Chapman cycle simulation.
Run from tests/vulcan/ to populate VULCAN/thermo/ and VULCAN/atm/.
"""
import os, math
import numpy as np

VULCAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VULCAN")

def write_cross_section(species, sigma_abs_fn, sigma_diss_fn, wl_range=(100, 350)):
    d = os.path.join(VULCAN_DIR, "thermo", "photo_cross", species)
    os.makedirs(d, exist_ok=True)
    wls = np.arange(wl_range[0], wl_range[1], 1.0)
    with open(os.path.join(d, f"{species}_cross.csv"), "w") as f:
        f.write("# wavelength (nm) photoabsorption  photodissociation (cm^2)\n")
        for wl in wls:
            a, d_ = sigma_abs_fn(wl), sigma_diss_fn(wl)
            f.write(f"{wl:.4f},    {a+d_:.5e},    {d_:.5e}\n")
    with open(os.path.join(d, f"{species}_branch.csv"), "w") as f:
        f.write(f"# Branching ratios for {species} -> (1) dissociation\n")
        f.write("# lambda , br_ratio_1\n")
        f.write(f"{wl_range[0]:.3e},    1.000\n")
        f.write(f"{wl_range[1]:.3e},    1.000\n")

def setup_all():
    # O2 cross-sections
    write_cross_section("O2",
        lambda wl: 7e-18 * math.exp(-(wl-145)**2/400),
        lambda wl: 1e-17 * math.exp(-(wl-160)**2/800) if wl < 240 else 0.0)
    # O3 cross-sections
    write_cross_section("O3",
        lambda wl: 1e-17 * math.exp(-(wl-255)**2/800),
        lambda wl: 1e-17 * math.exp(-(wl-300)**2/1200) if wl < 320 else 0.0)
    # Thresholds
    with open(os.path.join(VULCAN_DIR, "thermo", "photo_cross", "thresholds.txt"), "w") as f:
        f.write("# species  threshold (nm)\nO2         2.400000e+02\nO3         3.200000e+02\n")
    # Stellar flux (ergs/cm2/s/nm at stellar surface; config sets geometric factor = 1)
    d = os.path.join(VULCAN_DIR, "atm", "stellar_flux"); os.makedirs(d, exist_ok=True)
    hc = 1.98644582e-9
    with open(os.path.join(d, "sflux_chapman.txt"), "w") as f:
        f.write("# Wavelength(nm)  Flux(ergs/cm2/s/nm)\n")
        for wl in np.arange(100, 400, 0.5):
            if wl < 200:   flux_ph = 1e10 * math.exp(-(200-wl)/30)
            elif wl < 320: flux_ph = 1e13 * math.exp(-(wl-250)**2/5000)
            else:          flux_ph = 1e14
            f.write(f"  {wl:.1f}    {flux_ph * hc / wl:.6e}\n")
    # Atmospheric profile (very low P for optically thin)
    d2 = os.path.join(VULCAN_DIR, "atm"); os.makedirs(d2, exist_ok=True)
    P = np.logspace(np.log10(1e-1), np.log10(1e-4), 20)
    with open(os.path.join(d2, "atm_chapman.txt"), "w") as f:
        f.write("# Pressure(dyne/cm2)  Temp(K)  Kzz(cm2/s)\nPressure\tTemp\tKzz\n")
        for p in P: f.write(f"{p:.6e}\t250.0\t1.0e+05\n")
    # Empty BC files
    for name in ["BC_top.txt", "BC_bot.txt"]:
        path = os.path.join(d2, name)
        if not os.path.exists(path):
            with open(path, "w") as f: f.write("# Empty BC\n")
    print("Setup complete.")

if __name__ == "__main__":
    setup_all()
