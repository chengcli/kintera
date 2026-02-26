"""
Validate kintera_rt (pyharp DISORT wrapper) against analytical Beer-Lambert.

For pure absorption (no scattering), DISORT should reproduce the
analytical Beer-Lambert attenuation exactly. This validates our
kintera_rt module.

Usage:
    python3.11 test_rt_pyharp.py
"""
import math
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kintera_rt import compute_actinic_flux


def beer_lambert_reference(n_density, cross_section, stellar_flux, dz, cos_zen):
    """
    Analytical Beer-Lambert actinic flux at layer centers.
    All arrays are numpy. n_density: (nz,), cross_section: scalar,
    stellar_flux: scalar, dz: (nz,), cos_zen: scalar.
    Returns actinic flux at layer centers (nz,).
    """
    nz = len(n_density)
    dtau = n_density * cross_section * dz

    tau_levels = np.zeros(nz + 1)
    for k in range(nz):
        layer = nz - 1 - k
        tau_levels[k + 1] = tau_levels[k] + dtau[layer]

    F_levels = stellar_flux * np.exp(-tau_levels / cos_zen)

    aflux = np.zeros(nz)
    for j in range(nz):
        k_top = nz - 1 - j
        k_bot = nz - j
        aflux[j] = (F_levels[k_top] + F_levels[k_bot]) / 2.0

    return aflux


def kintera_rt_single_species(n_density, cross_section, stellar_flux, dz, cos_zen):
    """
    Call kintera_rt.compute_actinic_flux for a single-species, single-wavelength
    scenario. Returns actinic flux at layer centers (nz,).
    """
    nz = len(n_density)
    ni = 4  # [O, O2, O3, N2]
    y = np.zeros((nz, ni))
    y[:, 1] = n_density  # put absorber in O2 slot

    cross_O2 = np.array([cross_section])
    cross_O3 = np.zeros(1)
    sflux = np.array([stellar_flux])
    wavelengths = np.array([200.0])
    dzi = np.full(nz - 1, dz[0]) if np.allclose(dz, dz[0]) else (dz[:-1] + dz[1:]) / 2.0

    aflux = compute_actinic_flux(y, cross_O2, cross_O3, sflux, dzi, cos_zen, wavelengths)
    return aflux[:, 0]


def test_uniform_atmosphere():
    """Uniform atmosphere: kintera_rt (DISORT) vs analytical Beer-Lambert."""
    nz = 20
    n_density = np.full(nz, 1e12)
    sigma = 1e-17
    F0 = 1e13
    dz = np.full(nz, 1e5)
    cos_zen = 1.0

    aflux_ref = beer_lambert_reference(n_density, sigma, F0, dz, cos_zen)
    aflux_rt = kintera_rt_single_species(n_density, sigma, F0, dz, cos_zen)

    print(f"\n  Uniform atmosphere (nz={nz}, n=1e12, sigma=1e-17, cos_zen=1.0):")
    print(f"  {'layer':>5s} {'reference':>12s} {'kintera_rt':>12s} {'ratio':>10s}")
    max_err = 0
    for j in [0, 1, 5, 10, 15, 19]:
        if aflux_ref[j] > 1e-20 and aflux_rt[j] > 1e-20:
            r = aflux_rt[j] / aflux_ref[j]
            err = abs(r - 1)
            max_err = max(max_err, err)
            print(f"  {j:5d} {aflux_ref[j]:12.4e} {aflux_rt[j]:12.4e} {r:10.8f}")
        else:
            print(f"  {j:5d} {aflux_ref[j]:12.4e} {aflux_rt[j]:12.4e}    (below threshold)")

    print(f"  Max relative error: {max_err:.2e}")
    assert max_err < 1e-10, f"DISORT vs Beer-Lambert mismatch: {max_err}"


def test_slanted_path():
    """Slanted solar beam (cos_zen = 0.5)."""
    nz = 15
    n_density = np.full(nz, 5e11)
    sigma = 2e-17
    F0 = 1e13
    dz = np.full(nz, 2e5)
    cos_zen = 0.5

    aflux_ref = beer_lambert_reference(n_density, sigma, F0, dz, cos_zen)
    aflux_rt = kintera_rt_single_species(n_density, sigma, F0, dz, cos_zen)

    print(f"\n  Slanted path (cos_zen=0.5, nz={nz}):")
    max_err = 0
    for j in range(nz):
        if aflux_ref[j] > 1e-20 and aflux_rt[j] > 1e-20:
            err = abs(aflux_rt[j] / aflux_ref[j] - 1)
            max_err = max(max_err, err)

    print(f"  Max relative error: {max_err:.2e}")
    print(f"  Top flux: ref={aflux_ref[-1]:.4e}, rt={aflux_rt[-1]:.4e}")
    print(f"  Bot flux: ref={aflux_ref[0]:.4e}, rt={aflux_rt[0]:.4e}")
    assert max_err < 0.05, f"Slanted path error too large: {max_err:.2e}"


def test_nonuniform_density():
    """Non-uniform density profile (exponential decrease with altitude)."""
    nz = 20
    n_density = np.array([1e13 * math.exp(-i * 0.3) for i in range(nz)])
    sigma = 5e-18
    F0 = 5e12
    dz = np.full(nz, 5e4)
    cos_zen = 0.7

    aflux_ref = beer_lambert_reference(n_density, sigma, F0, dz, cos_zen)
    aflux_rt = kintera_rt_single_species(n_density, sigma, F0, dz, cos_zen)

    max_err = 0
    for j in range(nz):
        if aflux_ref[j] > 1e-20 and aflux_rt[j] > 1e-20:
            err = abs(aflux_rt[j] / aflux_ref[j] - 1)
            max_err = max(max_err, err)

    print(f"\n  Non-uniform density (exponential, nz={nz}):")
    print(f"  Max relative error: {max_err:.2e}")
    assert max_err < 1e-8


def test_optically_thick():
    """Optically thick atmosphere (large tau)."""
    nz = 10
    n_density = np.full(nz, 1e14)
    sigma = 1e-16
    F0 = 1e13
    dz = np.full(nz, 1e6)
    cos_zen = 1.0

    aflux_ref = beer_lambert_reference(n_density, sigma, F0, dz, cos_zen)
    aflux_rt = kintera_rt_single_species(n_density, sigma, F0, dz, cos_zen)

    print(f"\n  Optically thick (total tau={n_density[0]*sigma*dz[0]*nz:.1f}):")
    print(f"  Top flux: ref={aflux_ref[-1]:.4e}, rt={aflux_rt[-1]:.4e}")
    print(f"  Bot flux: ref={aflux_ref[0]:.4e}, rt={aflux_rt[0]:.4e}")

    max_err = 0
    for j in range(nz):
        if aflux_ref[j] > 1e-20 and aflux_rt[j] > 1e-20:
            err = abs(aflux_rt[j] / aflux_ref[j] - 1)
            max_err = max(max_err, err)

    print(f"  Max relative error: {max_err:.2e}")
    assert max_err < 1e-8


if __name__ == "__main__":
    print("=" * 70)
    print("kintera_rt (pyharp DISORT) vs analytical Beer-Lambert")
    print("=" * 70)

    print("\n[1] Uniform atmosphere")
    test_uniform_atmosphere()
    print("    PASSED")

    print("\n[2] Slanted solar path")
    test_slanted_path()
    print("    PASSED")

    print("\n[3] Non-uniform density")
    test_nonuniform_density()
    print("    PASSED")

    print("\n[4] Optically thick")
    test_optically_thick()
    print("    PASSED")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED: kintera_rt (DISORT) matches Beer-Lambert")
    print("=" * 70)
