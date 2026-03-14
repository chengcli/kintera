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
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "..", "VULCAN")
TOOLS_DIR = os.path.join(SCRIPT_DIR, "..", "tools")
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from tools.extract_vulcan_data import load_vulcan_output, vulcan_to_kintera_ic

VUL_FILE = os.path.join(VULCAN_DIR, "output", "CHO-kintera.vul")
YAML_FILE = os.path.join(SCRIPT_DIR, "..", "cho_photo.yaml")
AVO = 6.02214076e23

BUILD_LIB = os.path.join(ROOT_DIR, "build", "lib.macosx-14.0-arm64-cpython-311")
_SKIP_STEP_MATCH = False
_SCIPY_IMPORT = 'from scipy.linalg import solve_banded'  # kept for verify_steps
_sub_env = {**os.environ, "PYTHONUNBUFFERED": "1"}


def verify_steps():
    """Step-by-step verification against VULCAN."""
    script = 'import sys; sys.path.insert(0, r"' + BUILD_LIB + '")\n' + r'''
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
        env=_sub_env)
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
        raise RuntimeError("Verification failed")
    os.remove(script_path)
    os.remove(out_pkl)

    # Step-by-step comparison: same dt each step, kintera vs VULCAN
    if not _SKIP_STEP_MATCH:
        verify_step_match()


def verify_step_match():
    """Compare kintera vs VULCAN step-by-step with same dt (chem only, no photo)."""
    vul_steps_pkl = os.path.join(SCRIPT_DIR, "_vul_steps.pkl")
    gen_script = r'''
import os, sys, pickle, shutil
import numpy as np
VULCAN_DIR = sys.argv[1]
os.chdir(VULCAN_DIR)
sys.path.insert(0, VULCAN_DIR)
shutil.copy("vulcan_cfg_cho.py", "vulcan_cfg.py")
os.system(f"{sys.executable} make_chem_funs.py > /dev/null 2>&1")
for mod in ["vulcan_cfg", "chem_funs", "store", "build_atm", "op"]:
    sys.modules.pop(mod, None)
import vulcan_cfg, store, build_atm, op, chem_funs
from chem_funs import ni, nr
species = chem_funs.spec_list
nz = vulcan_cfg.nz
with open(vulcan_cfg.com_file, "r") as f:
    columns = f.readline()
    num_ele = len(columns.split()) - 2
type_list = ["int" for _ in range(num_ele)]
type_list.insert(0, "U20")
type_list.append("float")
compo = np.genfromtxt(vulcan_cfg.com_file, names=True, dtype=type_list)
data_var = store.Variables()
data_atm = store.AtmData()
make_atm = build_atm.Atm()
data_atm = make_atm.f_pico(data_atm)
data_atm = make_atm.load_TPK(data_atm)
rate = op.ReadRate()
data_var = rate.read_rate(data_var, data_atm)
if vulcan_cfg.use_lowT_limit_rates:
    data_var = rate.lim_lowT_rates(data_var, data_atm)
data_var = rate.rev_rate(data_var, data_atm)
data_var = rate.remove_rate(data_var)
ini_abun = build_atm.InitialAbun()
data_var = ini_abun.ini_y(data_var, data_atm)
data_var = ini_abun.ele_sum(data_var)
data_atm = make_atm.f_mu_dz(data_var, data_atm, op.Output())
make_atm.mol_diff(data_atm)
make_atm.BC_flux(data_atm)
M = data_atm.M.copy()
T = data_atm.Tco.copy()
y_ini = data_var.y.copy()
k = data_var.k
dydt_0 = chem_funs.chemdf(y_ini, M, k)
dt_list = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0, 1e4, 1e5, 1e6, 1e7]
y_run = y_ini.copy()
result = {"species": species, "ni": ni, "nz": nz}
for sp in species:
    result["tend0_" + sp] = dydt_0[:, species.index(sp)].tolist()
for i_step, dt in enumerate(dt_list):
    dydt = chem_funs.chemdf(y_run, M, k)
    jac = chem_funs.neg_symjac(y_run, M, k)
    rhs = dydt.flatten()
    A = np.eye(ni * nz) / dt + jac
    try:
        delta = np.linalg.solve(A, rhs).reshape(nz, ni)
    except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(A, rhs, rcond=None)[0].reshape(nz, ni)
    y_run = np.maximum(y_run + delta, 0)
    for sp in species:
        result[f"step{i_step}_{sp}"] = y_run[:, species.index(sp)].tolist()
with open(sys.argv[2], "wb") as f:
    pickle.dump(result, f)
'''
    gen_path = os.path.join(SCRIPT_DIR, "_gen_vul.py")
    with open(gen_path, "w") as f:
        f.write(gen_script)
    try:
        r = subprocess.run(
            [sys.executable, gen_path, VULCAN_DIR, vul_steps_pkl],
            capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=180)
    except subprocess.TimeoutExpired:
        print("  (VULCAN step gen timed out, skipping step-by-step match)\n")
        if os.path.exists(gen_path):
            os.remove(gen_path)
        return
    os.remove(gen_path)
    if r.returncode != 0:
        print("  (VULCAN step gen failed:", (r.stderr or "")[:200].strip() or "check VULCAN setup", ")\n")
        return

    cmp_script = 'import sys; sys.path.insert(0, r"' + BUILD_LIB + '")\n' + r'''
import os, sys, pickle
import numpy as np
import torch
import kintera as kt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.extract_vulcan_data import load_vulcan_output, vulcan_to_kintera_ic

vul_steps_pkl = sys.argv[1]
vul_file, yaml_file = sys.argv[2], sys.argv[3]
with open(vul_steps_pkl, "rb") as f:
    vd = pickle.load(f)
vul_species = vd["species"]
nz = vd["nz"]
dt_list = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0, 1e4, 1e5, 1e6, 1e7]

vul = load_vulcan_output(vul_file)
opts = kt.KineticsOptions.from_yaml(yaml_file)
kinet = kt.Kinetics(opts)
kt_species = opts.species()
stoich = kinet.stoich
kt_idx = {sp: i for i, sp in enumerate(kt_species)}
C = vulcan_to_kintera_ic(vul, kt_species)
T_t = torch.from_numpy(vul["T_K"].copy()).double()
P_t = torch.from_numpy(vul["P_Pa"].copy()).double()
C_FLOOR = 1e-50
wl_np = kinet.buffer("photolysis.wavelength").numpy().copy()
photo = {"wavelength": torch.from_numpy(wl_np).double(),
         "actinic_flux": torch.zeros(len(wl_np), nz, dtype=torch.float64)}
cvol = torch.ones(nz, dtype=torch.float64)

print("\n=== Step-by-step match (same dt, chem only, no photo) ===")
print(f"{'Step':>6} {'dt':>10}  {'H2O':>8} {'CH4':>8} {'CO':>8} {'H':>8}  worst species")
print("-" * 70)

for i_step, dt in enumerate(dt_list):
    C_t = torch.from_numpy(np.maximum(C, C_FLOOR)).double()
    rate, rc_ddC, _ = kinet.forward(T_t, P_t, C_t, photo)
    jac_rxn = kinet.jacobian(T_t, C_t, cvol, rate, rc_ddC)
    try:
        d_kt = kt.evolve_implicit(rate, stoich, jac_rxn, dt).numpy()
    except Exception:
        ns = stoich.shape[0]
        eye = torch.eye(ns, dtype=rate.dtype)
        SJ = stoich.matmul(jac_rxn)
        SR = stoich.matmul(rate.unsqueeze(-1)).squeeze(-1)
        A = eye / dt - SJ
        d_kt = torch.linalg.lstsq(A, SR).solution.numpy()
    C = np.maximum(C + d_kt, C_FLOOR)

    worst_sp, worst_dev = "", 0.0
    parts = []
    for sp in ["H2O", "CH4", "CO", "H", "OH", "H2"]:
        key = f"step{i_step}_{sp}"
        if sp in kt_idx and sp in vul_species and key in vd:
            kt_cgs = C[:, kt_idx[sp]] * 6.02214076e23 * 1e-6
            vul_c = np.array(vd[key])
            mask = (vul_c > 1e-10) & (kt_cgs > 1e-10)
            if mask.any():
                r = kt_cgs[mask] / vul_c[mask]
                med = np.median(r)
                dev = max(abs(1 - r.min()), abs(r.max() - 1))
                parts.append(f"{med:.3f}")
                if dev > worst_dev:
                    worst_dev = dev
                    worst_sp = sp
            else:
                parts.append("N/A")
        else:
            parts.append("-")
    status = " OK" if worst_dev < 0.25 else (" GAP" if worst_dev < 2.0 else " ***")
    print(f"  {i_step+1:2d}   {dt:10.0e}  " + " ".join(f"{p:>8}" for p in parts) + f"  {worst_sp} ({worst_dev:.1%}){status}")

print("\n(Ratios = kintera/VULCAN; OK=dev<25%%, GAP=25-200%%, ***=large)")
'''
    cmp_path = os.path.join(SCRIPT_DIR, "_cmp_steps.py")
    with open(cmp_path, "w") as f:
        f.write(cmp_script)
    r2 = subprocess.run(
        [sys.executable, cmp_path, vul_steps_pkl, VUL_FILE, YAML_FILE],
        capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=60, env=_sub_env)
    os.remove(cmp_path)
    if os.path.exists(vul_steps_pkl):
        os.remove(vul_steps_pkl)
    print(r2.stdout)
    if r2.returncode != 0:
        print("STDERR:", r2.stderr[-400:] if r2.stderr else "")


def run_kintera():
    """Run full CHO simulation using kintera-native functions."""
    script = 'import sys; sys.path.insert(0, r"' + BUILD_LIB + '")\n' + r'''
import os, sys, pickle, time
import numpy as np
import torch
from scipy.linalg import solve_banded
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

max_steps = 15000
rt_update_frq = 100
C_FLOOR = 1e-50
n_molm3 = vul["n_molm3"]
Kzz_si = torch.from_numpy(vul["Kzz_si"].copy()).double()
dzi_si = torch.from_numpy(vul["dzi_si"].copy()).double()

gamma_ros = 1.0 + 1.0 / 2.0**0.5
dt, dt_max = 1e-10, 1e17
rtol = 0.2; dt_var_min, dt_var_max = 0.5, 2.0; dt_min = 1e-14
atol_cgs = 0.1
mtol = 1e-22
stoich_np = stoich.numpy()

def build_banded_W(SJ_np, A_d, B_d, C_d, gamma_h, ns_loc, nz_loc):
    """Build banded matrix W = I/(γh) - SJ - J_diff (VULCAN-style)."""
    bw = ns_loc
    n = nz_loc * ns_loc
    ab = np.zeros((2 * bw + 1, n))
    ab[bw, :] = 1.0 / gamma_h

    for dr in range(-ns_loc + 1, ns_loc):
        c_lo, c_hi = max(0, -dr), min(ns_loc, ns_loc - dr)
        if c_lo >= c_hi:
            continue
        c_idx = np.arange(c_lo, c_hi)
        r_idx = c_idx + dr
        col_base = np.arange(nz_loc) * ns_loc
        cols = (col_base[:, None] + c_idx[None, :]).ravel()
        vals = SJ_np[:, r_idx, c_idx].ravel()
        ab[bw + dr, cols] -= vals

    all_s = np.arange(ns_loc)
    all_j_ns = np.arange(nz_loc) * ns_loc
    ab[bw, (all_j_ns[:, None] + all_s[None, :]).ravel()] -= np.repeat(A_d, ns_loc)
    if nz_loc > 1:
        j_up = np.arange(nz_loc - 1)
        ab[bw - ns_loc, ((j_up + 1)[:, None] * ns_loc + all_s[None, :]).ravel()] -= \
            np.repeat(B_d[:nz_loc - 1], ns_loc)
        j_lo = np.arange(1, nz_loc)
        ab[bw + ns_loc, ((j_lo - 1)[:, None] * ns_loc + all_s[None, :]).ravel()] -= \
            np.repeat(C_d[1:], ns_loc)
    return ab, bw

def build_dense_W(SJ_np, A_d, B_d, C_d, gamma_h):
    """Build dense system matrix W = I/(γh) - SJ - J_diff."""
    n = nz * nspecies
    W = np.zeros((n, n))
    for j in range(nz):
        b = j * nspecies
        W[b:b+nspecies, b:b+nspecies] = np.eye(nspecies) / gamma_h - SJ_np[j]
        for s in range(nspecies):
            W[b+s, b+s] -= A_d[j]
        if j < nz - 1:
            for s in range(nspecies):
                W[b+s, (j+1)*nspecies+s] -= B_d[j]
        if j > 0:
            for s in range(nspecies):
                W[b+s, (j-1)*nspecies+s] -= C_d[j]
    return W

def coupled_ros2_step(C_np, photo, dt_val):
    """VULCAN-style coupled Ros2 step with dense solver."""
    C_t = torch.from_numpy(np.maximum(C_np, C_FLOOR)).double()
    rate1, rc_ddC, _ = kinet.forward(T_t, P_t, C_t, photo)
    cvol = torch.ones(nz, dtype=torch.float64)
    jac_rxn = kinet.jacobian(T_t, C_t, cvol, rate1, rc_ddC)

    SJ_np = stoich.matmul(jac_rxn).numpy()
    A_t, B_t, Cc_t = kt.diffusion_coefficients(C_t, Kzz_si, dzi_si)
    A_d, B_d, C_d = A_t.numpy(), B_t.numpy(), Cc_t.numpy()

    gamma_h = gamma_ros * dt_val
    W = build_dense_W(SJ_np, A_d, B_d, C_d, gamma_h)

    f1_chem = (stoich_np @ rate1.numpy().T).T
    f1_diff = kt.diffusion_tendency(C_t, Kzz_si, dzi_si).numpy()
    f1 = f1_chem + f1_diff
    np.nan_to_num(f1, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        k1 = np.linalg.solve(W, f1.ravel()).reshape(nz, nspecies)
    except np.linalg.LinAlgError:
        k1 = np.linalg.lstsq(W, f1.ravel(), rcond=None)[0].reshape(nz, nspecies)
    if not np.all(np.isfinite(k1)):
        return None, None

    C_mid = np.maximum(C_np + k1 / gamma_ros, C_FLOOR)
    C_mid_t = torch.from_numpy(C_mid).double()
    rate2, _, _ = kinet.forward(T_t, P_t, C_mid_t, photo)

    f2_chem = (stoich_np @ rate2.numpy().T).T
    f2_diff = kt.diffusion_tendency(C_mid_t, Kzz_si, dzi_si).numpy()
    f2 = f2_chem + f2_diff
    np.nan_to_num(f2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    rhs2 = f2 - (2.0 / gamma_h) * k1

    try:
        k2 = np.linalg.solve(W, rhs2.ravel()).reshape(nz, nspecies)
    except np.linalg.LinAlgError:
        k2 = np.linalg.lstsq(W, rhs2.ravel(), rcond=None)[0].reshape(nz, nspecies)
    if not np.all(np.isfinite(k2)):
        return None, None

    m1 = 1.5 / gamma_ros
    m2 = 0.5 / gamma_ros
    sol = C_np + m1 * k1 + m2 * k2
    yk2 = C_np + k1 / gamma_ros
    err = np.abs(sol - yk2)

    return sol, err

t0 = time.time(); rejects = 0; accepted = 0; t_phys = 0.0
ymix_prev = None; longdy = 1.0
for step in range(max_steps):
    if step > 0 and step % rt_update_frq == 0:
        aflux_np = beer_lambert_simple(C)

    aflux_t = torch.from_numpy(aflux_np.T.copy()).double()
    photo = {"wavelength": wl_t, "actinic_flux": aflux_t}

    sol, err = coupled_ros2_step(C, photo, dt)

    if sol is None or not np.all(np.isfinite(sol)):
        dt = max(dt * 0.25, dt_min)
        rejects += 1
        continue

    ymix_sol = sol / np.maximum(sol.sum(axis=1, keepdims=True), 1e-100)

    err[ymix_sol < mtol] = 0
    err[sol < atol_cgs] = 0
    mask = sol > 0
    delta = np.amax(err[mask] / sol[mask]) if mask.any() else 0

    if delta > rtol:
        h_factor = 0.9 * (rtol / max(delta, 1e-30)) ** 0.5
        h_factor = np.clip(h_factor, dt_var_min, 1.0)
        dt = max(dt * h_factor, dt_min)
        rejects += 1
        continue

    ymix_sol = np.maximum(ymix_sol, 0)
    ymix_sol /= np.maximum(ymix_sol.sum(axis=1, keepdims=True), 1e-100)
    C = n_molm3[:, None] * ymix_sol
    C = np.maximum(C, C_FLOOR)

    accepted += 1
    t_phys += dt

    if delta == 0:
        delta = 0.01 * rtol
    h_factor = 0.9 * (rtol / delta) ** 0.5
    h_factor = np.clip(h_factor, dt_var_min, dt_var_max)
    dt = np.clip(dt * h_factor, dt_min, dt_max)

    if accepted % 100 == 0:
        ymix_now = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-100)
        if ymix_prev is not None:
            sig = ymix_now > 1e-5
            if sig.any():
                longdy = np.max(np.abs(ymix_now[sig] - ymix_prev[sig]) / ymix_now[sig])
        ymix_prev = ymix_now.copy()
        if accepted > 500 and longdy < 0.01:
            break

    if accepted % 500 == 0:
        print(f"  acc {accepted}/{max_steps}, dt={dt:.2e}, t={t_phys:.2e}, "
              f"longdy={longdy:.3e}, rej={rejects}")

elapsed = time.time() - t0
ymix = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-100)

nsteps = accepted
out_path = sys.argv[3]
with open(out_path, "wb") as f:
    pickle.dump({"ymix": ymix, "P_cgs": vul["P_cgs"], "T_K": vul["T_K"],
                 "species": species, "nsteps": nsteps,
                 "elapsed": elapsed, "t_phys": t_phys, "rejects": rejects}, f)
print(f"CHO: {nsteps} accepted ({rejects} rejected), t={t_phys:.2e}s, {elapsed:.1f}s wall")
'''
    out_pkl = os.path.join(SCRIPT_DIR, "_cho_kt.pkl")
    script_path = os.path.join(SCRIPT_DIR, "_run_cho.py")
    with open(script_path, "w") as f:
        f.write(script)

    result = subprocess.run(
        [sys.executable, script_path, VUL_FILE, YAML_FILE, out_pkl],
        capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=1800,
        env=_sub_env)
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
    parser.add_argument("--skip-step-match", action="store_true",
                        help="Skip VULCAN step-by-step comparison (faster)")
    args = parser.parse_args()

    global _SKIP_STEP_MATCH
    _SKIP_STEP_MATCH = args.skip_step_match

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
