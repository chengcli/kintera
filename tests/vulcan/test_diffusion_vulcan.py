"""
Compare kintera's diffusion_tendency with VULCAN's diffdf_no_mol.

Both implement the same conservative finite-volume discretization:
  d/dz [ Kzz * n_tot * d(y/n_tot)/dz ]

We set up an identical Gaussian profile and compare the diffusion
tendencies at a single timestep. They should match to machine precision.

Usage:
    python3.11 test_diffusion_vulcan.py
"""
import os, sys, math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "VULCAN")

# ---------------------------------------------------------------------------
# VULCAN's diffdf_no_mol (extracted and simplified — no advection, no BC flux)
# ---------------------------------------------------------------------------
def vulcan_diffdf(y, Kzz, dzi, nz, ni):
    """
    VULCAN's diffdf_no_mol without advection and without top/bot flux.
    y: (nz, ni), Kzz: (nz-1,), dzi: (nz-1,)
    """
    ysum = np.sum(y, axis=1)  # total number density per level

    A = np.zeros(nz)
    B = np.zeros(nz)
    C = np.zeros(nz)

    # Bottom boundary (j=0): zero-flux below
    A[0] = -1.0 / dzi[0] * (Kzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[0]
    B[0] =  1.0 / dzi[0] * (Kzz[0] / dzi[0]) * (ysum[1] + ysum[0]) / 2.0 / ysum[1]
    C[0] = 0

    # Top boundary (j=nz-1): zero-flux above
    A[nz-1] = -1.0 / dzi[nz-2] * (Kzz[nz-2] / dzi[nz-2]) * (ysum[nz-1] + ysum[nz-2]) / 2.0 / ysum[nz-1]
    B[nz-1] = 0
    C[nz-1] =  1.0 / dzi[nz-2] * (Kzz[nz-2] / dzi[nz-2]) * (ysum[nz-1] + ysum[nz-2]) / 2.0 / ysum[nz-2]

    # Interior
    for j in range(1, nz-1):
        dz_ave = 0.5 * (dzi[j-1] + dzi[j])
        A[j] = -2.0 / (dzi[j-1] + dzi[j]) * (
            Kzz[j] / dzi[j] * (ysum[j+1] + ysum[j]) / 2.0
            + Kzz[j-1] / dzi[j-1] * (ysum[j] + ysum[j-1]) / 2.0
        ) / ysum[j]
        B[j] = 2.0 / (dzi[j-1] + dzi[j]) * Kzz[j] / dzi[j] * (ysum[j+1] + ysum[j]) / 2.0 / ysum[j+1]
        C[j] = 2.0 / (dzi[j-1] + dzi[j]) * Kzz[j-1] / dzi[j-1] * (ysum[j] + ysum[j-1]) / 2.0 / ysum[j-1]

    # Compute tendency
    tmp0 = A[0] * y[0] + B[0] * y[1]
    tmp_mid = np.array([
        A[j] * y[j] + B[j] * y[j+1] + C[j] * y[j-1]
        for j in range(1, nz-1)
    ])
    tmp_top = A[nz-1] * y[nz-1] + C[nz-1] * y[nz-2]

    diff = np.vstack([tmp0[np.newaxis, :], tmp_mid, tmp_top[np.newaxis, :]])
    return diff, A, B, C


# ---------------------------------------------------------------------------
# kintera's diffusion_tendency (Python reimplementation matching the C++)
# ---------------------------------------------------------------------------
def kintera_diffdf(y, Kzz, dzi, nz, ni):
    """
    kintera's diffusion_tendency — Python version matching the C++ code.
    """
    n_tot = np.sum(y, axis=1)
    n_avg = (n_tot[1:] + n_tot[:-1]) / 2.0
    D = Kzz * n_avg / dzi

    A = np.zeros(nz)
    B = np.zeros(nz)
    C = np.zeros(nz)

    # Bottom
    A[0] = -D[0] / dzi[0] / n_tot[0]
    B[0] =  D[0] / dzi[0] / n_tot[1]

    # Top
    A[nz-1] = -D[nz-2] / dzi[nz-2] / n_tot[nz-1]
    C[nz-1] =  D[nz-2] / dzi[nz-2] / n_tot[nz-2]

    # Interior
    if nz > 2:
        for j in range(1, nz-1):
            dz_avg = (dzi[j-1] + dzi[j]) / 2.0
            A[j] = -(D[j] + D[j-1]) / (dz_avg * n_tot[j])
            B[j] = D[j] / (dz_avg * n_tot[j+1])
            C[j] = D[j-1] / (dz_avg * n_tot[j-1])

    # Tendency
    tend = np.zeros_like(y)
    for j in range(nz):
        tend[j] = A[j] * y[j]
        if j < nz-1:
            tend[j] += B[j] * y[j+1]
        if j > 0:
            tend[j] += C[j] * y[j-1]

    return tend, A, B, C


# ---------------------------------------------------------------------------
# Test setup
# ---------------------------------------------------------------------------
def make_test_case(nz=20, ni=2, Kzz_val=1e5, dz_val=1e5, n_bg=1e12):
    """Create a Gaussian test profile."""
    dzi = np.full(nz-1, dz_val)
    Kzz = np.full(nz-1, Kzz_val)
    y = np.zeros((nz, ni))
    z_mid = nz / 2.0
    sigma = nz / 6.0
    for j in range(nz):
        gauss = 0.01 * n_bg * math.exp(-(j - z_mid)**2 / (2 * sigma**2))
        y[j, 0] = gauss
        y[j, 1] = n_bg - gauss
    return y, Kzz, dzi


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_coefficients_match():
    """A, B, C coefficients must match between VULCAN and kintera."""
    nz, ni = 20, 2
    y, Kzz, dzi = make_test_case(nz, ni)

    _, Av, Bv, Cv = vulcan_diffdf(y, Kzz, dzi, nz, ni)
    _, Ak, Bk, Ck = kintera_diffdf(y, Kzz, dzi, nz, ni)

    print(f"\n  Coefficient comparison (nz={nz}):")
    print(f"  {'j':>3s}  {'A_vulcan':>14s} {'A_kintera':>14s} {'ratio':>10s}  "
          f"{'B_vulcan':>14s} {'B_kintera':>14s} {'ratio':>10s}")
    for j in [0, 1, nz//2, nz-2, nz-1]:
        rA = Ak[j]/Av[j] if abs(Av[j]) > 1e-30 else float('nan')
        rB = Bk[j]/Bv[j] if abs(Bv[j]) > 1e-30 else float('nan')
        print(f"  {j:3d}  {Av[j]:14.6e} {Ak[j]:14.6e} {rA:10.6f}  "
              f"{Bv[j]:14.6e} {Bk[j]:14.6e} {rB:10.6f}")

    for j in range(nz):
        if abs(Av[j]) > 1e-30:
            assert abs(Ak[j]/Av[j] - 1) < 1e-10, f"A[{j}] mismatch"
        if abs(Bv[j]) > 1e-30:
            assert abs(Bk[j]/Bv[j] - 1) < 1e-10, f"B[{j}] mismatch"
        if abs(Cv[j]) > 1e-30:
            assert abs(Ck[j]/Cv[j] - 1) < 1e-10, f"C[{j}] mismatch"


def test_tendency_match():
    """Diffusion tendencies must match between VULCAN and kintera."""
    nz, ni = 20, 2
    y, Kzz, dzi = make_test_case(nz, ni)

    tend_v, _, _, _ = vulcan_diffdf(y, Kzz, dzi, nz, ni)
    tend_k, _, _, _ = kintera_diffdf(y, Kzz, dzi, nz, ni)

    print(f"\n  Tendency comparison (species 0, nz={nz}):")
    print(f"  {'j':>3s}  {'VULCAN':>14s} {'kintera':>14s} {'ratio':>10s}")
    for j in [0, 1, nz//4, nz//2, 3*nz//4, nz-2, nz-1]:
        r = tend_k[j,0]/tend_v[j,0] if abs(tend_v[j,0]) > 1e-30 else float('nan')
        print(f"  {j:3d}  {tend_v[j,0]:14.6e} {tend_k[j,0]:14.6e} {r:10.6f}")

    max_err = 0
    for j in range(nz):
        for s in range(ni):
            if abs(tend_v[j,s]) > 1e-30:
                err = abs(tend_k[j,s]/tend_v[j,s] - 1)
                max_err = max(max_err, err)

    print(f"  Max relative error: {max_err:.2e}")
    assert max_err < 1e-10, f"Tendency mismatch: max rel err = {max_err}"


def test_mass_conservation():
    """Both should conserve total mass."""
    nz, ni = 20, 2
    y, Kzz, dzi = make_test_case(nz, ni)

    tend_v, _, _, _ = vulcan_diffdf(y, Kzz, dzi, nz, ni)
    tend_k, _, _, _ = kintera_diffdf(y, Kzz, dzi, nz, ni)

    for label, tend in [("VULCAN", tend_v), ("kintera", tend_k)]:
        for s in range(ni):
            net = np.sum(tend[:, s])
            scale = np.mean(np.abs(y[:, s]))
            rel = abs(net) / (scale + 1e-30)
            print(f"  {label} species {s}: net tendency = {net:.3e}, rel = {rel:.3e}")
            assert rel < 1e-10, f"{label} species {s} not conserved"


def test_nonuniform_grid():
    """Test with a non-uniform grid."""
    nz, ni = 15, 3
    dzi = np.array([1e4 * (1.5**i) for i in range(nz-1)])
    Kzz = np.full(nz-1, 5e4)
    n_bg = 1e13
    y = np.zeros((nz, ni))
    for j in range(nz):
        y[j, 0] = n_bg * 0.1 * math.exp(-(j - 5)**2 / 8)
        y[j, 1] = n_bg * 0.3
        y[j, 2] = n_bg * 0.6

    tend_v, _, _, _ = vulcan_diffdf(y, Kzz, dzi, nz, ni)
    tend_k, _, _, _ = kintera_diffdf(y, Kzz, dzi, nz, ni)

    max_err = 0
    for j in range(nz):
        for s in range(ni):
            if abs(tend_v[j,s]) > 1e-30:
                err = abs(tend_k[j,s]/tend_v[j,s] - 1)
                max_err = max(max_err, err)

    print(f"\n  Non-uniform grid (nz={nz}, ni={ni}): max rel error = {max_err:.2e}")
    assert max_err < 1e-10


if __name__ == "__main__":
    print("=" * 70)
    print("kintera vs VULCAN: Diffusion Tendency Comparison")
    print("=" * 70)

    print("\n[1] Coefficient match")
    test_coefficients_match()
    print("    PASSED")

    print("\n[2] Tendency match")
    test_tendency_match()
    print("    PASSED")

    print("\n[3] Mass conservation")
    test_mass_conservation()
    print("    PASSED")

    print("\n[4] Non-uniform grid")
    test_nonuniform_grid()
    print("    PASSED")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
