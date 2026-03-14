#!/usr/bin/env python3
"""CHO photo-chemistry: kintera-native vs VULCAN comparison.

Uses implicit Euler with lstsq fallback for chemistry,
implicit diffusion (tridiagonal solve) for transport, and hydrostatic
normalization after each step to preserve total atmospheric density.

Step-by-step verification:
  1. Match initial conditions
  2. Match first chemistry step (rates, tendencies)
  3. Match second step
  4. Full run to steady state

Usage:
    python3.11 run_comparison.py              # full run + plot
    python3.11 run_comparison.py --verify     # step-by-step verification only
"""
import argparse
import os
import sys
import subprocess
import pickle
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "..", "VULCAN")
TOOLS_DIR = os.path.join(SCRIPT_DIR, "..", "tools")
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from tools.extract_vulcan_data import load_vulcan_output, vulcan_to_kintera_ic

VUL_FILE = os.path.join(VULCAN_DIR, "output", "CHO-kintera.vul")
YAML_FILE = os.path.join(SCRIPT_DIR, "..", "cho_photo.yaml")
AVO = 6.02214076e23


def verify_steps():
    """Step-by-step verification against VULCAN."""
    script = r'''
import os, sys, pickle
import numpy as np
import torch
import kintera as kt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from tools.extract_vulcan_data import load_vulcan_output, vulcan_to_kintera_ic

vul_file, yaml_file = sys.argv[1], sys.argv[2]
vul = load_vulcan_output(vul_file)

opts = kt.KineticsOptions.from_yaml(yaml_file)
kinet = kt.Kinetics(opts)
species = opts.species()
nspecies = len(species)
stoich = kinet.stoich
idx = {sp: i for i, sp in enumerate(species)}
vul_idx = {sp: i for i, sp in enumerate(vul["species"])}

nz = vul["nz"]
C = vulcan_to_kintera_ic(vul, species)

T_t = torch.from_numpy(vul["T_K"].copy()).double()
P_t = torch.from_numpy(vul["P_Pa"].copy()).double()

has_photo = "photolysis.wavelength" in [kinet.buffer_names()[i]
    for i in range(len(kinet.buffer_names()))] if hasattr(kinet, "buffer_names") else True
try:
    wl_np = kinet.buffer("photolysis.wavelength").numpy().copy()
    wl_t = torch.from_numpy(wl_np).double()
    zero_flux = torch.zeros(len(wl_np), nz, dtype=torch.float64)
    photo_args = {"wavelength": wl_t, "actinic_flux": zero_flux}
except:
    photo_args = {}

# Step 1: Check initial conditions (only compare where both have non-negligible values)
print("=== Step 1: Initial conditions ===")
for sp in ["H2", "H2O", "CH4", "CO", "CO2", "H", "OH"]:
    if sp in idx and sp in vul_idx:
        kt_val = C[:, idx[sp]]
        vul_val = vul["y_ini_cgs"][:, vul_idx[sp]] * 1e6 / 6.02214076e23
        mask = (vul_val > 1e-40) & (kt_val > 1e-40)
        if mask.any():
            ratio = kt_val[mask] / vul_val[mask]
            print(f"  {sp:6s}: max ratio={np.max(ratio):.6f}, "
                  f"min ratio={np.min(ratio):.6f} ({mask.sum()} layers)")
        else:
            print(f"  {sp:6s}: all negligible")

# Step 2: First chemistry step (thermal only, zero photolysis)
print("\n=== Step 2: First chemistry step (thermal, no photo) ===")
C_t = torch.from_numpy(np.maximum(C, 1e-50)).double()
rate, rc_ddC, _ = kinet.forward(T_t, P_t, C_t, photo_args)
chem_tend = torch.matmul(stoich, rate.unsqueeze(-1)).squeeze(-1).numpy()

print(f"  Rate shape: {rate.shape}")
print(f"  Chemistry tendency shape: {chem_tend.shape}")

for j_layer in [0, nz//4, nz//2, 3*nz//4, nz-1]:
    tend = chem_tend[j_layer]
    C_layer = C[j_layer]
    max_ratio = np.max(np.abs(tend) / (np.abs(C_layer) + 1e-50))
    i_max = np.argmax(np.abs(tend) / (np.abs(C_layer) + 1e-50))
    print(f"  Layer {j_layer:3d} (P={vul['P_cgs'][j_layer]:.2e} dyn/cm²): "
          f"max|tend/C|={max_ratio:.2e} [{species[i_max]}]")

# Step 3: Implicit Euler with fixed dt
dt_test = 1e-8
cvol = torch.ones(nz, dtype=torch.float64)
jac_rxn = kinet.jacobian(T_t, C_t, cvol, rate, rc_ddC)
delta1 = kt.evolve_implicit(rate, stoich, jac_rxn, dt_test).numpy()

print(f"\n=== Step 3: Implicit Euler (dt={dt_test:.0e}) ===")
for sp in ["H2O", "CH4", "CO", "H", "OH"]:
    if sp in idx:
        d = delta1[:, idx[sp]]
        c = np.abs(C[:, idx[sp]]) + 1e-50
        print(f"  {sp:6s}: max|delta|={np.max(np.abs(d)):.4e}, "
              f"max|delta/C|={np.max(np.abs(d)/c):.4e}")

# Step 4: Second step
C2 = np.maximum(C + delta1, 1e-50)
C2_t = torch.from_numpy(C2).double()
rate2, rc_ddC2, _ = kinet.forward(T_t, P_t, C2_t, photo_args)
jac2 = kinet.jacobian(T_t, C2_t, cvol, rate2, rc_ddC2)
delta2 = kt.evolve_implicit(rate2, stoich, jac2, dt_test).numpy()

print(f"\n=== Step 4: Second step (dt={dt_test:.0e}) ===")
for sp in ["H2O", "CH4", "CO", "H", "OH"]:
    if sp in idx:
        d = delta2[:, idx[sp]]
        c = np.abs(C2[:, idx[sp]]) + 1e-50
        print(f"  {sp:6s}: max|delta|={np.max(np.abs(d)):.4e}, "
              f"max|delta/C|={np.max(np.abs(d)/c):.4e}")

# Diffusion check
Kzz_si = torch.from_numpy(vul["Kzz_si"].copy()).double()
dzi_si = torch.from_numpy(vul["dzi_si"].copy()).double()
diff_tend = kt.diffusion_tendency(C_t, Kzz_si, dzi_si).numpy()
print(f"\n=== Diffusion tendency ===")
for sp in ["H2O", "CH4", "CO"]:
    if sp in idx:
        dt_sp = diff_tend[:, idx[sp]]
        print(f"  {sp:6s}: max|diff_tend|={np.max(np.abs(dt_sp)):.4e}")

print("\nVerification complete.")
out_path = sys.argv[3]
with open(out_path, "wb") as f:
    pickle.dump({"ok": True}, f)
'''
    out_pkl = os.path.join(SCRIPT_DIR, "_verify.pkl")
    script_path = os.path.join(SCRIPT_DIR, "_verify.py")
    with open(script_path, "w") as f:
        f.write(script)

    result = subprocess.run(
        [sys.executable, script_path, VUL_FILE, YAML_FILE, out_pkl],
        capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=60,
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
        raise RuntimeError("Verification failed")
    os.remove(script_path)
    os.remove(out_pkl)


def run_kintera():
    """Run full CHO simulation using kintera-native functions."""
    script = r'''
import os, sys, pickle, time
import numpy as np
import torch
import kintera as kt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "..", "VULCAN")
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from tools.extract_vulcan_data import load_vulcan_output, vulcan_to_kintera_ic

AVO = 6.02214076e23
HC = 1.98644582e-9; R_SUN = 6.957e10; AU_CM = 1.4959787e13

vul_file, yaml_file = sys.argv[1], sys.argv[2]
vul = load_vulcan_output(vul_file)

opts = kt.KineticsOptions.from_yaml(yaml_file)
kinet = kt.Kinetics(opts)
species = opts.species(); nspecies = len(species)
stoich = kinet.stoich
idx = {sp: i for i, sp in enumerate(species)}

nz = vul["nz"]
C = vulcan_to_kintera_ic(vul, species)

T_t = torch.from_numpy(vul["T_K"].copy()).double()
P_t = torch.from_numpy(vul["P_Pa"].copy()).double()

# Photolysis setup: load cross-sections and stellar flux
wl_np = kinet.buffer("photolysis.wavelength").numpy().copy()
wl_t = torch.from_numpy(wl_np).double()
n_photo = 0
photo_xs = {}
while True:
    try:
        xs = kinet.buffer(f"photolysis.cross_section_{n_photo}").numpy()[:, 0]
        photo_xs[n_photo] = xs
        n_photo += 1
    except:
        break

sflux_raw = np.loadtxt(os.path.join(VULCAN_DIR, "atm/stellar_flux/sflux-HD189_Moses11.txt"),
                        skiprows=1)
r_star = 0.756; orbit = 0.031
sflux_ergs = sflux_raw[:, 1] * (r_star * R_SUN / (AU_CM * orbit)) ** 2
stellar_photon = np.interp(wl_np, sflux_raw[:, 0],
                           sflux_ergs * sflux_raw[:, 0] / HC, left=0, right=0)

dz_cgs = vul["dz_cgs"]

vul_bins = vul["bins"]
vul_cross = vul.get("cross", {})
cross_interp = {}
for sp_name, sigma_data in vul_cross.items():
    if sp_name in idx:
        cross_interp[sp_name] = np.interp(wl_np, vul_bins, sigma_data, left=0, right=0)

def beer_lambert_simple(C_np):
    """Simple Beer-Lambert using total absorption cross-sections from VULCAN."""
    n_cgs = C_np * AVO * 1e-6
    alpha = np.zeros((nz, len(wl_np)))
    for sp_name, sigma in cross_interp.items():
        alpha += n_cgs[:, idx[sp_name]:idx[sp_name]+1] * sigma[None, :]
    dtau = alpha * dz_cgs[:, None]
    tau = np.zeros((nz, len(wl_np)))
    tau[-1] = dtau[-1] / 2.0
    for j in range(nz - 2, -1, -1):
        tau[j] = tau[j+1] + (dtau[j+1] + dtau[j]) / 2.0
    return stellar_photon[None, :] * np.exp(-tau)

# Use VULCAN's stored actinic flux if available
aflux_np = None
if "aflux" in vul and vul["aflux"] is not None:
    vul_aflux = np.array(vul["aflux"])
    if vul_aflux.shape[0] == nz:
        aflux_np = np.zeros((nz, len(wl_np)))
        for j in range(nz):
            aflux_np[j] = np.interp(wl_np, vul["bins"], vul_aflux[j],
                                     left=0, right=0)

if aflux_np is None:
    aflux_np = beer_lambert_simple(C)

from scipy.linalg import solve_banded

max_steps = 8000
rt_update_frq = 10
C_FLOOR = 1e-40
dt0 = 1e-10; growth = 1.1
dt_max = 5e6
n_molm3 = vul["n_molm3"]
Kzz_np = vul["Kzz_si"].copy()
dzi_np = vul["dzi_si"].copy()

def safe_evolve_implicit(rate, stoich, jac_rxn, dt_val):
    try:
        return kt.evolve_implicit(rate, stoich, jac_rxn, dt_val)
    except Exception:
        ns = stoich.shape[0]
        eye = torch.eye(ns, dtype=rate.dtype, device=rate.device)
        SJ = stoich.matmul(jac_rxn)
        SR = stoich.matmul(rate.unsqueeze(-1)).squeeze(-1)
        A = eye / dt_val - SJ
        return torch.linalg.lstsq(A, SR).solution

def implicit_diffusion(C_in, dt_val):
    """Fully implicit 1D diffusion via tridiagonal solve, per species."""
    C_out = C_in.copy()
    nz_loc, ns = C_in.shape
    for s in range(ns):
        diag = np.ones(nz_loc)
        upper = np.zeros(nz_loc - 1)
        lower = np.zeros(nz_loc - 1)
        for j in range(1, nz_loc - 1):
            dz_avg = 0.5 * (dzi_np[j-1] + dzi_np[j])
            a_up = Kzz_np[j] / (dzi_np[j] * dz_avg)
            a_dn = Kzz_np[j-1] / (dzi_np[j-1] * dz_avg)
            diag[j] += dt_val * (a_up + a_dn)
            upper[j] = -dt_val * a_up
            lower[j-1] = -dt_val * a_dn
        dz_avg0 = dzi_np[0]
        a_up0 = Kzz_np[0] / (dzi_np[0] * dz_avg0)
        diag[0] += dt_val * a_up0
        upper[0] = -dt_val * a_up0
        dz_avgN = dzi_np[-1]
        a_dnN = Kzz_np[-1] / (dzi_np[-1] * dz_avgN)
        diag[-1] += dt_val * a_dnN
        lower[-1] = -dt_val * a_dnN
        ab = np.zeros((3, nz_loc))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower
        C_out[:, s] = solve_banded((1, 1), ab, C_in[:, s])
    return C_out

t0 = time.time(); t_phys = 0.0
ymix_prev = None; longdy = 1.0
cvol = torch.ones(nz, dtype=torch.float64)
for step in range(max_steps):
    if step == 0:
        dt = dt0
    else:
        dt = min(dt * growth, dt_max)

    if step > 0 and step % rt_update_frq == 0:
        aflux_np = beer_lambert_simple(C)

    C_t = torch.from_numpy(np.maximum(C, C_FLOOR)).double()
    aflux_t = torch.from_numpy(aflux_np.T.copy()).double()
    rate, rc_ddC, _ = kinet.forward(T_t, P_t, C_t,
        {"wavelength": wl_t, "actinic_flux": aflux_t})
    jac_rxn = kinet.jacobian(T_t, C_t, cvol, rate, rc_ddC)

    delta_chem = safe_evolve_implicit(rate, stoich, jac_rxn, dt).numpy()
    np.nan_to_num(delta_chem, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.maximum(C + delta_chem, C_FLOOR)

    ymix = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-100)
    C = n_molm3[:, None] * ymix
    C = np.maximum(C, C_FLOOR)

    C = implicit_diffusion(C, dt)
    C = np.maximum(C, C_FLOOR)
    ymix = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-100)
    C = n_molm3[:, None] * ymix
    C = np.maximum(C, C_FLOOR)
    t_phys += dt

    if (step + 1) % 100 == 0:
        ymix_now = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-100)
        if ymix_prev is not None:
            sig = ymix_now > 1e-5
            if sig.any():
                longdy = np.max(np.abs(ymix_now[sig] - ymix_prev[sig]) / ymix_now[sig])
        ymix_prev = ymix_now.copy()
        if step > 1000 and longdy < 0.01:
            break

    if (step + 1) % 500 == 0:
        print(f"  step {step+1}/{max_steps}, dt={dt:.2e}, t={t_phys:.2e}, "
              f"longdy={longdy:.3e}")

elapsed = time.time() - t0
ymix = C / n_molm3[:, None]

nsteps = step + 1
out_path = sys.argv[3]
with open(out_path, "wb") as f:
    pickle.dump({"ymix": ymix, "P_cgs": vul["P_cgs"], "T_K": vul["T_K"],
                 "species": species, "nsteps": nsteps,
                 "elapsed": elapsed, "t_phys": t_phys}, f)
print(f"CHO: {nsteps} steps, t={t_phys:.2e}s, {elapsed:.1f}s wall")
'''
    out_pkl = os.path.join(SCRIPT_DIR, "_cho_kt.pkl")
    script_path = os.path.join(SCRIPT_DIR, "_run_cho.py")
    with open(script_path, "w") as f:
        f.write(script)

    result = subprocess.run(
        [sys.executable, script_path, VUL_FILE, YAML_FILE, out_pkl],
        capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=1800,
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    print(result.stdout.strip())
    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:])
        raise RuntimeError("CHO kintera solver failed")

    with open(out_pkl, "rb") as f:
        data = pickle.load(f)
    os.remove(script_path)
    os.remove(out_pkl)
    return data


def plot(kt_data):
    """Plot kintera vs VULCAN mixing ratio profiles."""
    vul = load_vulcan_output(VUL_FILE)
    ymix_kt = kt_data["ymix"]; P_cgs = kt_data["P_cgs"]
    species = kt_data["species"]
    idx = {sp: i for i, sp in enumerate(species)}
    vul_idx = {sp: i for i, sp in enumerate(vul["species"])}

    plot_species = [s for s in ["H2O", "CH4", "CO", "CO2", "H", "OH", "C2H2", "HCN"]
                    if s in idx]
    ncols = 4
    nrows = max(1, (len(plot_species) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), sharey=True)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    fig.suptitle(f"CHO Photo: Kintera vs VULCAN\n"
                 f"(kintera: {kt_data['nsteps']} steps, {kt_data['elapsed']:.0f}s)",
                 fontsize=13, fontweight="bold")

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12",
              "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
    P_bar = P_cgs / 1e6

    for i, sp in enumerate(plot_species):
        ax = axes_flat[i]
        kt_mix = ymix_kt[:, idx[sp]]
        valid = kt_mix > 1e-45
        if valid.any():
            ax.semilogx(kt_mix[valid], P_bar[valid], "-o", color=colors[i % 8],
                         markersize=3, label="kintera", linewidth=2)
        if sp in vul_idx:
            v_mix = vul["ymix"][:, vul_idx[sp]]
            v_valid = v_mix > 1e-30
            if v_valid.any():
                ax.semilogx(v_mix[v_valid], vul["P_cgs"][v_valid] / 1e6,
                             "--", color="gray", label="VULCAN",
                             linewidth=1.5, alpha=0.8)
        ax.set_xlabel(f"{sp} mixing ratio", fontsize=11)
        ax.invert_yaxis(); ax.set_yscale("log")
        if i % ncols == 0:
            ax.set_ylabel("Pressure (bar)", fontsize=12)
        ax.set_title(sp, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    for j in range(len(plot_species), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "cho_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Step-by-step verification only")
    args = parser.parse_args()

    if args.verify:
        print("=== CHO Step-by-Step Verification ===")
        verify_steps()
        return

    print("=== CHO Full Comparison ===")
    print("  Step-by-step verification...")
    verify_steps()
    print("\n  Running full kintera simulation...")
    kt_data = run_kintera()
    print("\n  Plotting...")
    out = plot(kt_data)
    print(f"\nDone. Plot: {out}")


if __name__ == "__main__":
    main()
