#!/usr/bin/env python3
"""Chapman Cycle Example - Stratospheric Ozone Chemistry"""

import torch
import numpy as np
import kintera
from kintera import (
    Reaction,
    PhotolysisOptions,
    Photolysis,
    ArrheniusOptions,
    Arrhenius,
)

torch.set_default_dtype(torch.float64)


def setup_chapman_cycle():
    """Initialize species and return kinetics options."""
    species = ["N2", "O2", "O", "O3"]
    weights = [28.e-3, 32.e-3, 16.e-3, 48.e-3]
    cv_R = [2.5, 2.5, 1.5, 3.5]

    kintera.set_species_names(species)
    kintera.set_species_weights(weights)
    kintera.set_species_cref_R(cv_R)

    # Photolysis reactions
    photo_opts = PhotolysisOptions()
    photo_opts.reactions([
        Reaction("O2 => 2 O"),
        Reaction("O3 => O2 + O"),
    ])

    wave_o2 = [120., 130., 140., 150., 160., 170., 180., 190., 200., 205., 210., 220., 230., 240., 250.]
    wave_o3 = [200., 210., 220., 230., 240., 250., 255., 260., 270., 280., 290., 300., 310., 320., 330.]

    xs_o2 = [1.0e-17, 1.2e-17, 1.5e-17, 1.8e-17, 1.5e-17, 1.0e-17, 5.0e-18,
             1.5e-18, 7.0e-19, 5.0e-19, 2.0e-19, 5.0e-20, 1.0e-20, 1.0e-21, 1.0e-23]
    xs_o3 = [3.0e-19, 6.0e-19, 1.0e-18, 2.5e-18, 5.0e-18, 8.0e-18, 1.1e-17,
             1.0e-17, 7.0e-18, 4.0e-18, 2.0e-18, 8.0e-19, 3.0e-19, 5.0e-20, 5.0e-21]

    all_waves = sorted(set(wave_o2 + wave_o3))
    photo_opts.wavelength(all_waves)
    photo_opts.temperature([200., 300.])

    # Interpolate cross-sections to combined grid
    xs_combined = []
    for w in all_waves:
        if w in wave_o2:
            xs_combined.append(xs_o2[wave_o2.index(w)])
        elif w < wave_o2[0]:
            xs_combined.append(xs_o2[0])
        elif w > wave_o2[-1]:
            xs_combined.append(xs_o2[-1])
        else:
            for i in range(len(wave_o2) - 1):
                if wave_o2[i] <= w <= wave_o2[i + 1]:
                    frac = (w - wave_o2[i]) / (wave_o2[i + 1] - wave_o2[i])
                    xs_combined.append(xs_o2[i] + frac * (xs_o2[i + 1] - xs_o2[i]))
                    break
            else:
                xs_combined.append(0.0)

    for w in all_waves:
        if w in wave_o3:
            xs_combined.append(xs_o3[wave_o3.index(w)])
        elif w < wave_o3[0]:
            xs_combined.append(xs_o3[0])
        elif w > wave_o3[-1]:
            xs_combined.append(xs_o3[-1])
        else:
            for i in range(len(wave_o3) - 1):
                if wave_o3[i] <= w <= wave_o3[i + 1]:
                    frac = (w - wave_o3[i]) / (wave_o3[i + 1] - wave_o3[i])
                    xs_combined.append(xs_o3[i] + frac * (xs_o3[i + 1] - xs_o3[i]))
                    break
            else:
                xs_combined.append(0.0)

    photo_opts.cross_section(xs_combined)
    photo_opts.branches([
        [{"O2": 1.0}],
        [{"O3": 1.0}],
    ])

    # Arrhenius reactions
    arrhenius_opts = ArrheniusOptions()
    arrhenius_opts.reactions([
        Reaction("O + O2 => O3"),
        Reaction("O + O3 => 2 O2"),
    ])
    arrhenius_opts.A([1.7e-14, 8.0e-12])
    arrhenius_opts.b([-2.4, 0.0])
    arrhenius_opts.Ea_R([0.0, 2060.0])

    return photo_opts, arrhenius_opts


def run_chapman_cycle():
    """Run Chapman cycle simulation."""
    print("=" * 60)
    print("Chapman Cycle - Stratospheric Ozone Chemistry")
    print("=" * 60)

    photo_opts, arrhenius_opts = setup_chapman_cycle()
    photolysis = Photolysis(photo_opts)
    arrhenius = Arrhenius(arrhenius_opts)

    T, P = 250.0, 1000.0
    temp = torch.tensor([T])
    pres = torch.tensor([P])

    n_tot = P / (8.314 * T)
    print(f"\nTotal number density: {n_tot:.3e} mol/m^3")

    x_N2, x_O2, x_O, x_O3 = 0.78, 0.21, 1.e-10, 1.e-8
    conc = torch.tensor([[x_N2 * n_tot, x_O2 * n_tot, x_O * n_tot, x_O3 * n_tot]])
    print(f"Initial O3 mixing ratio: {x_O3 * 1e6:.3f} ppm")

    # Create actinic flux
    wavelength = torch.tensor(photo_opts.wavelength())
    flux_values = torch.zeros(len(wavelength))
    for i, w in enumerate(wavelength):
        if w < 200:
            flux_values[i] = 1.e10 * torch.exp(-(200 - w) / 30)
        elif w < 320:
            flux_values[i] = 1.e13 * torch.exp(-(w - 250)**2 / 5000)
        else:
            flux_values[i] = 1.e14

    flux_map = {"wavelength": wavelength, "actinic_flux": flux_values}

    J = photolysis.forward(temp, pres, conc, flux_map)
    k = arrhenius.forward(temp, pres, conc, {})
    print(f"\nPhotolysis rates:")
    print(f"  J(O2) = {J[0, 0].item():.3e} s^-1")
    print(f"  J(O3) = {J[0, 1].item():.3e} s^-1")
    print(f"\nArrhenius rate constants:")
    print(f"  k2 = {k[0, 0].item():.3e}")
    print(f"  k4 = {k[0, 1].item():.3e}")

    # Time evolution
    print("\n" + "-" * 60)
    print("Time Evolution to Steady State")
    print("-" * 60)

    dt, nsteps = 1.0, 10000
    history = {"time": [], "O2": [], "O": [], "O3": [], "O_total": []}
    conc_evolve = conc.clone()

    for step in range(nsteps):
        c_N2, c_O2, c_O, c_O3 = conc_evolve[0].tolist()

        J_vals = photolysis.forward(temp, pres, conc_evolve, flux_map)
        k_vals = arrhenius.forward(temp, pres, conc_evolve, {})
        J_O2, J_O3 = J_vals[0, 0].item(), J_vals[0, 1].item()
        k2, k4 = k_vals[0, 0].item(), k_vals[0, 1].item()

        R1 = J_O2 * c_O2
        R2 = k2 * c_O * c_O2
        R3 = J_O3 * c_O3
        R4 = k4 * c_O * c_O3

        conc_evolve[0, 1] += (-R1 + R2 + R3 + 2 * R4) * dt
        conc_evolve[0, 2] += (2 * R1 - R2 + R3 - R4) * dt
        conc_evolve[0, 3] += (R2 - R3 - R4) * dt
        conc_evolve = torch.clamp(conc_evolve, min=0.0)

        if step % 100 == 0:
            c_N2, c_O2, c_O, c_O3 = conc_evolve[0].tolist()
            history["time"].append(step * dt)
            history["O2"].append(c_O2)
            history["O"].append(c_O)
            history["O3"].append(c_O3)
            history["O_total"].append(2 * c_O2 + c_O + 3 * c_O3)

    c_N2, c_O2, c_O, c_O3 = conc_evolve[0].tolist()
    O3_ppm = c_O3 / n_tot * 1e6

    print(f"\nFinal concentrations after {nsteps * dt:.0f} seconds:")
    print(f"  O2: {c_O2:.3e} mol/m^3")
    print(f"  O:  {c_O:.3e} mol/m^3")
    print(f"  O3: {c_O3:.3e} mol/m^3 ({O3_ppm:.3f} ppm)")

    # Validation
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    O_initial = 2 * x_O2 * n_tot + x_O * n_tot + 3 * x_O3 * n_tot
    O_final = 2 * c_O2 + c_O + 3 * c_O3
    mass_error = abs(O_final - O_initial) / O_initial * 100

    print(f"\n1. Mass Conservation: {mass_error:.4f}% error")
    print("   " + ("PASS" if mass_error < 1.0 else "FAIL"))

    J_O2_final = photolysis.forward(temp, pres, conc_evolve, flux_map)[0, 0].item()
    J_O3_final = photolysis.forward(temp, pres, conc_evolve, flux_map)[0, 1].item()
    print(f"\n2. Photolysis Rates:")
    print(f"   J(O2) = {J_O2_final:.3e} s^-1")
    print(f"   J(O3) = {J_O3_final:.3e} s^-1")

    print(f"\n3. Ozone: {O3_ppm:.3f} ppm")
    print("   " + ("PASS" if 0.1 < O3_ppm < 50 else "WARNING"))

    print("\n" + "=" * 60)
    return history


if __name__ == "__main__":
    history = run_chapman_cycle()

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].plot(history["time"], np.array(history["O3"]) * 1e6, 'b-', lw=2)
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("O3 (μmol/m³)")
        axes[0, 0].set_title("Ozone Evolution")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].semilogy(history["time"], history["O"], 'r-', lw=2)
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("O (mol/m³)")
        axes[0, 1].set_title("Atomic Oxygen")
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(history["time"], history["O2"], 'g-', lw=2)
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("O2 (mol/m³)")
        axes[1, 0].set_title("Molecular Oxygen")
        axes[1, 0].grid(True, alpha=0.3)

        O_total = np.array(history["O_total"])
        axes[1, 1].plot(history["time"], (O_total - O_total[0]) / O_total[0] * 100, 'k-', lw=2)
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Error (%)")
        axes[1, 1].set_title("Mass Conservation")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("chapman_cycle_results.png", dpi=150)
        print("\nPlot saved to: chapman_cycle_results.png")
        plt.show()

    except ImportError:
        print("\nNote: Install matplotlib to generate plots")
