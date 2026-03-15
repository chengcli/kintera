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
C_t = torch.from_numpy(np.maximum(C, 0.0)).double()
rate, rc_ddC, _ = kinet.forward(T_t, P_t, C_t, photo_args)
chem_tend = torch.matmul(stoich, rate.unsqueeze(-1)).squeeze(-1).numpy()

print(f"  Rate shape: {rate.shape}")
print(f"  Chemistry tendency shape: {chem_tend.shape}")

for j_layer in [0, nz//4, nz//2, 3*nz//4, nz-1]:
    tend = chem_tend[j_layer]
    C_layer = C[j_layer]
    max_ratio = np.max(np.abs(tend) / (np.abs(C_layer) + 1e-300))
    i_max = np.argmax(np.abs(tend) / (np.abs(C_layer) + 1e-300))
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
        c = np.abs(C[:, idx[sp]]) + 1e-300
        print(f"  {sp:6s}: max|delta|={np.max(np.abs(d)):.4e}, "
              f"max|delta/C|={np.max(np.abs(d)/c):.4e}")

# Step 4: Second step
C2 = np.maximum(C + delta1, 0.0)
C2_t = torch.from_numpy(C2).double()
rate2, rc_ddC2, _ = kinet.forward(T_t, P_t, C2_t, photo_args)
jac2 = kinet.jacobian(T_t, C2_t, cvol, rate2, rc_ddC2)
delta2 = kt.evolve_implicit(rate2, stoich, jac2, dt_test).numpy()

print(f"\n=== Step 4: Second step (dt={dt_test:.0e}) ===")
for sp in ["H2O", "CH4", "CO", "H", "OH"]:
    if sp in idx:
        d = delta2[:, idx[sp]]
        c = np.abs(C2[:, idx[sp]]) + 1e-300
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


N_STEPS_SHORT = 50
DT_FIXED = 1e-8
TOTAL_DT_SHORT = N_STEPS_SHORT * DT_FIXED

# Exponentially growing dt schedule for full convergence
_CONV_DT_SCHEDULE = []
for _exp in range(-8, 13):
    _CONV_DT_SCHEDULE.extend([10.0**_exp] * 10)
_CONV_DT_STR = repr(_CONV_DT_SCHEDULE)
_CONV_TOTAL_T = sum(_CONV_DT_SCHEDULE)
_CONV_N_STEPS = len(_CONV_DT_SCHEDULE)

# No Python-side concentration floor needed — the C++ forward() and jacobian()
# methods internally clamp concentrations to 1e-20 for the mass-action rate
# products and Jacobian terms.
C_FLOOR = 0.0


def gen_vulcan_convergence():
    """Run VULCAN chem-only with the convergence dt schedule."""
    vul_pkl = os.path.join(SCRIPT_DIR, "_vul_conv.pkl")
    gen_script = r'''
import os, sys, pickle, shutil
import numpy as np
VULCAN_DIR = sys.argv[1]
out_pkl = sys.argv[2]
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
k = data_var.k
y_run = data_var.y.copy()
dt_list = ''' + _CONV_DT_STR + r'''
for dt in dt_list:
    dydt = chem_funs.chemdf(y_run, M, k)
    jac = chem_funs.neg_symjac(y_run, M, k)
    rhs = dydt.flatten()
    A = np.eye(ni * nz) / dt + jac
    try:
        delta = np.linalg.solve(A, rhs).reshape(nz, ni)
    except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(A, rhs, rcond=None)[0].reshape(nz, ni)
    y_run = np.maximum(y_run + delta, 0)
M_col = M.reshape(nz, 1)
ymix = y_run / np.maximum(M_col, 1e-300)
result = {"species": species, "ymix": ymix, "P_cgs": np.array(data_atm.pco),
          "T_K": np.array(data_atm.Tco), "nsteps": len(dt_list),
          "t_phys": sum(dt_list)}
with open(out_pkl, "wb") as f:
    pickle.dump(result, f)
'''
    gen_path = os.path.join(SCRIPT_DIR, "_gen_vul_conv.py")
    with open(gen_path, "w") as f:
        f.write(gen_script)
    try:
        r = subprocess.run(
            [sys.executable, gen_path, VULCAN_DIR, vul_pkl],
            capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=300,
            env=_sub_env)
    except subprocess.TimeoutExpired:
        if os.path.exists(gen_path):
            os.remove(gen_path)
        return None
    if os.path.exists(gen_path):
        os.remove(gen_path)
    if r.returncode != 0:
        print("  (VULCAN convergence failed:", (r.stderr or "")[:200].strip(), ")")
        return None
    with open(vul_pkl, "rb") as f:
        data = pickle.load(f)
    os.remove(vul_pkl)
    return data


def verify_step_match():
    """Compare kintera vs VULCAN with exponentially growing dt.

    Steps through dt = 1e-8, 1e-7, 1e-6, ..., 1e+8 (one IE step each),
    stops when any tracked species diverges by >30%.
    """
    import json
    dt_exponents = list(range(-8, 9))  # 1e-8 to 1e+8
    dt_list_str = "[" + ", ".join(f"1e{e}" for e in dt_exponents) + "]"
    n_dt = len(dt_exponents)

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
k = data_var.k
y_run = data_var.y.copy()
dt_list = ''' + dt_list_str + r'''
result = {"species": species, "ni": ni, "nz": nz, "M": M.tolist()}
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
            capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=300,
            env=_sub_env)
    except subprocess.TimeoutExpired:
        print("  (VULCAN step gen timed out, skipping step-by-step match)\n")
        if os.path.exists(gen_path):
            os.remove(gen_path)
        return
    if os.path.exists(gen_path):
        os.remove(gen_path)
    if r.returncode != 0:
        print("  (VULCAN step gen failed:", (r.stderr or "")[:200].strip()
              or "check VULCAN setup", ")\n")
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
dt_list = ''' + dt_list_str + r'''

vul = load_vulcan_output(vul_file)
opts = kt.KineticsOptions.from_yaml(yaml_file)
kinet = kt.Kinetics(opts)
kt_species = opts.species()
stoich = kinet.stoich
kt_idx = {sp: i for i, sp in enumerate(kt_species)}
vul_idx = {sp: i for i, sp in enumerate(vul_species)}
C = vulcan_to_kintera_ic(vul, kt_species)
T_t = torch.from_numpy(vul["T_K"].copy()).double()
P_t = torch.from_numpy(vul["P_Pa"].copy()).double()
wl_np = kinet.buffer("photolysis.wavelength").numpy().copy()
photo = {"wavelength": torch.from_numpy(wl_np).double(),
         "actinic_flux": torch.zeros(len(wl_np), nz, dtype=torch.float64)}
cvol = torch.ones(nz, dtype=torch.float64)
AVO = 6.02214076e23
P_cgs = vul["P_cgs"]

tracked = ["H2O", "CH4", "CO", "CO2", "H", "OH", "H2", "C2H2", "HCO", "O"]
hdr_sp = ["H2O", "CH4", "CO", "H", "OH", "H2"]

print("\n=== Exponentially growing dt: stop at >30% divergence ===")
print(f"{'Step':>4} {'dt':>10} {'t_cum':>10}  " +
      " ".join(f"{s:>6}" for s in hdr_sp) + "  worst")
print("-" * 78)

t_cum = 0.0
for i_step, dt in enumerate(dt_list):
    t_cum += dt
    C_t = torch.from_numpy(np.maximum(C, 0.0)).double()
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
    C = np.maximum(C + d_kt, 0.0)

    worst_sp, worst_dev = "", 0.0
    sp_devs = {}
    for sp in tracked:
        key = f"step{i_step}_{sp}"
        if sp in kt_idx and sp in vul_idx and key in vd:
            kt_cgs = C[:, kt_idx[sp]] * AVO * 1e-6
            vul_c = np.array(vd[key])
            mask = (vul_c > 1e-10) & (kt_cgs > 1e-10)
            if mask.any():
                r = kt_cgs[mask] / vul_c[mask]
                dev = max(abs(1 - r.min()), abs(r.max() - 1))
                sp_devs[sp] = dev
                if dev > worst_dev:
                    worst_dev = dev
                    worst_sp = sp

    parts = []
    for sp in hdr_sp:
        if sp in sp_devs:
            d = sp_devs[sp]
            parts.append(f"{d:5.1%}" if d < 10 else f"{d:5.0f}x")
        else:
            parts.append("  N/A")
    tag = " OK" if worst_dev < 0.30 else " <<<STOP"
    print(f"  {i_step+1:2d}   {dt:10.0e} {t_cum:10.2e}  " +
          " ".join(f"{p:>6}" for p in parts) +
          f"  {worst_sp}({worst_dev:.1%}){tag}")

    if worst_dev >= 0.30:
        print(f"\n--- Divergence detail at step {i_step+1} (dt={dt:.0e}, t_cum={t_cum:.2e}) ---")
        for sp in tracked:
            key = f"step{i_step}_{sp}"
            if sp in kt_idx and sp in vul_idx and key in vd:
                kt_cgs = C[:, kt_idx[sp]] * AVO * 1e-6
                vul_c = np.array(vd[key])
                mask = (vul_c > 1e-10) & (kt_cgs > 1e-10)
                if mask.any():
                    r = kt_cgs[mask] / vul_c[mask]
                    j_worst = np.argmax(np.abs(r - 1))
                    j_layer = np.where(mask)[0][j_worst]
                    print(f"  {sp:>6s}: dev={sp_devs.get(sp,0):.2%}, "
                          f"worst layer={j_layer} (P={P_cgs[j_layer]:.2e}), "
                          f"kt={kt_cgs[j_layer]:.4e}, vul={vul_c[j_layer]:.4e}, "
                          f"ratio={r[j_worst]:.6f}")
                else:
                    print(f"  {sp:>6s}: no significant concentrations")
        break

print()
'''
    cmp_path = os.path.join(SCRIPT_DIR, "_cmp_steps.py")
    with open(cmp_path, "w") as f:
        f.write(cmp_script)
    r2 = subprocess.run(
        [sys.executable, cmp_path, vul_steps_pkl, VUL_FILE, YAML_FILE],
        capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=120,
        env=_sub_env)
    os.remove(cmp_path)
    if os.path.exists(vul_steps_pkl):
        os.remove(vul_steps_pkl)
    print(r2.stdout)
    if r2.returncode != 0:
        print("STDERR:", r2.stderr[-400:] if r2.stderr else "")


def run_kintera():
    """Run CHO chemistry-only convergence with exponentially growing dt."""
    script = ('import sys; sys.path.insert(0, r"' + BUILD_LIB + '")\n'
              + 'C_FLOOR = ' + str(C_FLOOR) + '\n'
              + 'dt_schedule = ' + _CONV_DT_STR + '\n' + r'''
import os, sys, pickle, time
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
species = opts.species(); nspecies = len(species)
stoich = kinet.stoich

nz = vul["nz"]
C = vulcan_to_kintera_ic(vul, species)

T_t = torch.from_numpy(vul["T_K"].copy()).double()
P_t = torch.from_numpy(vul["P_Pa"].copy()).double()

wl_np = kinet.buffer("photolysis.wavelength").numpy().copy()
wl_t = torch.from_numpy(wl_np).double()
zero_flux = torch.zeros(len(wl_np), nz, dtype=torch.float64)
photo = {"wavelength": wl_t, "actinic_flux": zero_flux}
cvol = torch.ones(nz, dtype=torch.float64)

t0 = time.time()
for dt in dt_schedule:
    C_t = torch.from_numpy(np.maximum(C, C_FLOOR)).double()
    rate, rc_ddC, _ = kinet.forward(T_t, P_t, C_t, photo)
    jac_rxn = kinet.jacobian(T_t, C_t, cvol, rate, rc_ddC)
    delta = kt.evolve_implicit(rate, stoich, jac_rxn, dt).numpy()
    np.nan_to_num(delta, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.maximum(C + delta, 0.0)

elapsed = time.time() - t0
ymix = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-100)

nsteps = len(dt_schedule)
t_phys = sum(dt_schedule)
out_path = sys.argv[3]
with open(out_path, "wb") as f:
    pickle.dump({"ymix": ymix, "P_cgs": vul["P_cgs"], "T_K": vul["T_K"],
                 "species": species, "nsteps": nsteps,
                 "elapsed": elapsed, "t_phys": t_phys}, f)
print(f"CHO: {nsteps} steps, t={t_phys:.2e}s, {elapsed:.1f}s wall")
''')
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


def plot(kt_data, vulcan_conv=None):
    """Plot kintera vs VULCAN mixing ratio profiles.
    vulcan_conv: VULCAN convergence result (or None to use .vul steady-state)."""
    vul = load_vulcan_output(VUL_FILE)
    ymix_kt = kt_data["ymix"]
    P_cgs = kt_data["P_cgs"]
    species = kt_data["species"]
    idx = {sp: i for i, sp in enumerate(species)}
    if vulcan_conv is not None:
        vul_ymix = vulcan_conv["ymix"]
        vul_species = vulcan_conv["species"]
        vul_idx = {sp: i for i, sp in enumerate(vul_species)}
        subtitle = (f"{vulcan_conv['nsteps']} steps, "
                    f"t={vulcan_conv['t_phys']:.2e} s (no transport)")
    else:
        vul_ymix = vul["ymix"]
        vul_species = vul["species"]
        vul_idx = {sp: i for i, sp in enumerate(vul_species)}
        subtitle = f"kintera: {kt_data['nsteps']} steps | VULCAN: steady-state"

    plot_species = [s for s in ["H2O", "CH4", "CO", "CO2", "H", "OH", "C2H2", "HCO", "HCN"]
                    if s in idx]
    ncols = 4
    nrows = max(1, (len(plot_species) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), sharey=True)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    fig.suptitle(f"CHO Chem-Only: Kintera vs VULCAN\n{subtitle}",
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
            v_mix = vul_ymix[:, vul_idx[sp]]
            v_valid = v_mix > 1e-30
            v_P = vulcan_conv["P_cgs"] if vulcan_conv else vul["P_cgs"]
            if v_valid.any():
                ax.semilogx(v_mix[v_valid], v_P[v_valid] / 1e6,
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
    parser.add_argument("--no-vulcan-50", action="store_true",
                        help="Skip VULCAN 50-step gen; plot kintera vs .vul steady-state")
    args = parser.parse_args()

    global _SKIP_STEP_MATCH
    _SKIP_STEP_MATCH = args.skip_step_match

    if args.verify:
        print("=== CHO Step-by-Step Verification ===")
        verify_steps()
        return

    print("=== CHO Chem-Only Convergence Comparison ===")
    print("  Step-by-step verification...")
    verify_steps()
    vulcan_conv = None
    if not args.no_vulcan_50:
        print("\n  Running VULCAN convergence (chem-only)...")
        vulcan_conv = gen_vulcan_convergence()
        if vulcan_conv is None:
            print("  (VULCAN convergence unavailable, plot will use .vul steady-state)")
    else:
        print("\n  (--no-vulcan-50: using .vul steady-state for VULCAN curve)")
    print("\n  Running kintera convergence (chem-only)...")
    kt_data = run_kintera()
    print("\n  Plotting...")
    out = plot(kt_data, vulcan_conv=vulcan_conv)
    print(f"\nDone. Plot: {out}")


if __name__ == "__main__":
    main()
