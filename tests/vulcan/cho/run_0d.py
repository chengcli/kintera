#!/usr/bin/env python3
"""0D (single-layer) CHO photo-chemistry: kintera vs VULCAN.

Picks one or more atmospheric layers, uses VULCAN's actinic flux,
and runs both codes with identical dt schedules.  This isolates
chemistry + photolysis from any spatial/transport effects.

Usage:
    python3.11 run_0d.py                         # default: layers near P=0.1 bar
    python3.11 run_0d.py --layers 25 30 35 40    # specific layers
"""
import argparse
import os
import sys
import subprocess
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "..", "VULCAN")
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from tools.extract_vulcan_data import load_vulcan_output, vulcan_to_kintera_ic

VUL_FILE = os.path.join(VULCAN_DIR, "output", "CHO-kintera.vul")
YAML_FILE = os.path.join(SCRIPT_DIR, "..", "cho_photo.yaml")
BUILD_LIB = os.path.join(ROOT_DIR, "build", "lib.macosx-14.0-arm64-cpython-311")
AVO = 6.02214076e23
_sub_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

DT_SCHEDULE = []
for _exp in range(-8, 13):
    DT_SCHEDULE.extend([10.0**_exp] * 10)
DT_STR = repr(DT_SCHEDULE)


def find_layers_near_pressure(P_cgs, target_bar=0.1, n=3):
    """Find layer indices closest to target pressure."""
    P_bar = P_cgs / 1e6
    idx = np.argsort(np.abs(np.log10(P_bar) - np.log10(target_bar)))
    return sorted(idx[:n].tolist())


def run_vulcan_0d(layers):
    """Run VULCAN single-layer chemistry+photo convergence for multiple layers."""
    layers_str = repr(layers)
    script = r'''
import os, sys, pickle, shutil
import numpy as np
VULCAN_DIR = sys.argv[1]
LAYERS = ''' + layers_str + r'''
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

if vulcan_cfg.use_photo:
    rate.make_bins_read_cross(data_var, data_atm)
    make_atm.read_sflux(data_var, data_atm)
    solver = op.Ros2()
    solver.compute_tau(data_var, data_atm)
    solver.compute_flux(data_var, data_atm)
    solver.compute_J(data_var, data_atm)
    data_var = rate.remove_rate(data_var)
    aflux_save = data_var.aflux.copy()
    bins_save = data_var.bins.copy()
    np.nan_to_num(data_var.y, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    data_var.y[data_var.y < 0] = 0.0
    for _ki in data_var.k:
        if isinstance(data_var.k[_ki], np.ndarray):
            np.nan_to_num(data_var.k[_ki], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
else:
    aflux_save, bins_save = None, None

M = data_atm.M.copy()
k = data_var.k
y_full = data_var.y.copy()
dt_list = ''' + DT_STR + r'''

results = {}
for L in LAYERS:
    y_work = y_full.copy()
    history = []
    t_cum = 0.0
    for dt in dt_list:
        t_cum += dt
        dydt = chem_funs.chemdf(y_work, M, k)
        jac = chem_funs.neg_symjac(y_work, M, k)
        rhs_L = dydt[L]
        jac_L = jac[L*ni:(L+1)*ni, L*ni:(L+1)*ni]
        A = np.eye(ni) / dt + jac_L
        try:
            delta = np.linalg.solve(A, rhs_L)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A, rhs_L, rcond=None)[0]
        y_work[L] = np.maximum(y_work[L] + delta, 0)
        history.append(y_work[L].copy())
    history = np.array(history)
    P_bar = data_atm.pco[L] / 1e6
    print(f"  VULCAN L{L}: P={P_bar:.2e} bar, T={data_atm.Tco[L]:.0f} K")
    results[L] = {"history": history, "P_cgs": float(data_atm.pco[L]),
                  "T_K": float(data_atm.Tco[L]), "M": float(M[L]),
                  "y_ini": y_full[L].copy()}

results["_meta"] = {"species": species, "dt_list": dt_list,
                    "aflux": aflux_save, "bins": bins_save}
with open(sys.argv[2], "wb") as f:
    pickle.dump(results, f)
'''
    out_pkl = os.path.join(SCRIPT_DIR, "_vul_0d.pkl")
    script_path = os.path.join(SCRIPT_DIR, "_run_vul_0d.py")
    with open(script_path, "w") as f:
        f.write(script)
    r = subprocess.run(
        [sys.executable, script_path, VULCAN_DIR, out_pkl],
        capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=300,
        env=_sub_env)
    print(r.stdout.strip())
    if r.returncode != 0:
        print("STDERR:", r.stderr[-500:])
        raise RuntimeError("VULCAN 0D failed")
    os.remove(script_path)
    with open(out_pkl, "rb") as f:
        data = pickle.load(f)
    os.remove(out_pkl)
    return data


def run_kintera_0d(layers, vul_data):
    """Run kintera single-layer chemistry+photo convergence using C++ solver."""
    import torch
    sys.path.insert(0, BUILD_LIB)
    for k in list(sys.modules.keys()):
        if 'kintera' in k:
            del sys.modules[k]
    import kintera as kt

    vul = load_vulcan_output(VUL_FILE)
    opts = kt.KineticsOptions.from_yaml(YAML_FILE)
    kinet = kt.Kinetics(opts)
    species = opts.species()
    stoich = kinet.stoich
    idx = {sp: i for i, sp in enumerate(species)}
    ns = len(species)
    nz = vul["nz"]

    C_full = vulcan_to_kintera_ic(vul, species)

    wl_np = kinet.buffer("photolysis.wavelength").numpy().copy()
    wl_t = torch.from_numpy(wl_np).double()

    meta = vul_data["_meta"]
    aflux_vul = meta["aflux"]
    bins_vul = meta["bins"]

    results = {}
    for L in layers:
        C = C_full[L:L+1, :].copy()
        T_t = torch.tensor([vul["T_K"][L]], dtype=torch.float64)
        P_t = torch.tensor([vul["P_Pa"][L]], dtype=torch.float64)

        if aflux_vul is not None:
            from scipy.interpolate import interp1d
            f_interp = interp1d(bins_vul, aflux_vul[L], kind="linear",
                                bounds_error=False, fill_value=0.0)
            aflux_1 = np.maximum(f_interp(wl_np), 0.0)
            aflux_t = torch.from_numpy(aflux_1.reshape(-1, 1).copy()).double()
        else:
            aflux_t = torch.zeros(len(wl_np), 1, dtype=torch.float64)

        photo = {"wavelength": wl_t, "actinic_flux": aflux_t}
        cvol = torch.ones(1, dtype=torch.float64)

        history = []
        for dt in DT_SCHEDULE:
            C_t = torch.from_numpy(np.maximum(C, 0.0)).double()
            rate, rc_ddC, _ = kinet.forward(T_t, P_t, C_t, photo)
            jac_rxn = kinet.jacobian(T_t, C_t, cvol, rate,
                                     torch.zeros_like(rc_ddC))
            delta = kt.evolve_implicit(rate, stoich, jac_rxn, dt).numpy()
            np.nan_to_num(delta, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            C[0] = np.maximum(C[0] + delta[0], 0.0)
            history.append(C[0].copy())

        history = np.array(history)
        P_bar = vul["P_cgs"][L] / 1e6
        print(f"  kintera L{L}: P={P_bar:.2e} bar")
        results[L] = {"history": history}

    results["_meta"] = {"species": species, "idx": idx}
    return results


def plot_0d(vul_data, kt_data, layers):
    """Plot time evolution for key species at each layer."""
    meta_v = vul_data["_meta"]
    vul_sp = meta_v["species"]
    vul_idx = {sp: i for i, sp in enumerate(vul_sp)}

    meta_k = kt_data["_meta"]
    kt_sp = meta_k["species"]
    kt_idx = meta_k["idx"]

    dt_list = DT_SCHEDULE
    t_cum = np.cumsum(dt_list)

    plot_sp = ["H2O", "CH4", "CO", "CO2", "H", "OH", "H2", "C2H2", "HCO", "H2CO", "O", "O2"]
    plot_sp = [s for s in plot_sp if s in vul_idx and s in kt_idx]

    nrows = len(layers)
    ncols = len(plot_sp)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows),
                             squeeze=False)

    fig.suptitle("0D Chemistry+Photo: kintera vs VULCAN (per layer)",
                 fontsize=14, fontweight="bold")

    for row, L in enumerate(layers):
        vul_hist = vul_data[L]["history"]
        kt_hist = kt_data[L]["history"]
        M = vul_data[L]["M"]
        P_bar = vul_data[L]["P_cgs"] / 1e6
        T_K = vul_data[L]["T_K"]

        # Convert kintera mol/m³ to molecules/cm³ so both use VULCAN's M
        kt_hist_cgs = kt_hist * AVO * 1e-6

        for col, sp in enumerate(plot_sp):
            ax = axes[row, col]
            vul_mix = vul_hist[:, vul_idx[sp]] / M
            kt_mix = kt_hist_cgs[:, kt_idx[sp]] / M

            ax.loglog(t_cum, np.maximum(vul_mix, 1e-45), "--", color="gray",
                      label="VULCAN", linewidth=1.5)
            ax.loglog(t_cum, np.maximum(kt_mix, 1e-45), "-", color="C0",
                      label="kintera", linewidth=1.8)

            vf, kf = vul_mix[-1], kt_mix[-1]
            if vf > 1e-30 and kf > 1e-30:
                dev = abs(kf / vf - 1)
                title = f"{sp} (dev={dev:.1%})"
            else:
                title = sp
            ax.set_title(title, fontsize=10, fontweight="bold")

            if row == 0:
                ax.legend(fontsize=7)
            if col == 0:
                ax.set_ylabel(f"L{L}\nP={P_bar:.1e} bar\nT={T_K:.0f} K",
                              fontsize=9)
            if row == nrows - 1:
                ax.set_xlabel("Time (s)", fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if len(layers) == 1:
        out = os.path.join(SCRIPT_DIR, f"cho_0d_L{layers[0]}.png")
    else:
        out = os.path.join(SCRIPT_DIR, "cho_0d_multi.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)

    print(f"\n{'Layer':>6} {'P(bar)':>10} {'Species':>8} "
          f"{'VULCAN':>12} {'kintera':>12} {'ratio':>10}")
    print("-" * 65)
    for L in layers:
        vul_hist = vul_data[L]["history"]
        kt_hist = kt_data[L]["history"]
        M = vul_data[L]["M"]
        P_bar = vul_data[L]["P_cgs"] / 1e6
        kt_hist_cgs = kt_hist * AVO * 1e-6
        for sp in plot_sp:
            vf = vul_hist[-1, vul_idx[sp]] / M
            kf = kt_hist_cgs[-1, kt_idx[sp]] / M
            ratio = kf / vf if vf > 1e-30 else float("inf")
            print(f"{L:>6d} {P_bar:>10.2e} {sp:>8} {vf:12.4e} {kf:12.4e} "
                  f"{ratio:10.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices (default: auto-pick near P=0.1 bar)")
    args = parser.parse_args()

    vul = load_vulcan_output(VUL_FILE)
    if args.layers is None:
        layers = find_layers_near_pressure(vul["P_cgs"], target_bar=0.1, n=3)
        layers_wide = find_layers_near_pressure(vul["P_cgs"], target_bar=0.01, n=1)
        layers = sorted(set(layers + layers_wide))
    else:
        layers = args.layers

    print(f"=== 0D Comparison at Layers {layers} ===")
    for L in layers:
        P_bar = vul["P_cgs"][L] / 1e6
        print(f"  L{L}: P={P_bar:.2e} bar, T={vul['T_K'][L]:.0f} K")

    print("\n  Running VULCAN 0D...")
    vul_data = run_vulcan_0d(layers)
    print("  Running kintera 0D...")
    kt_data = run_kintera_0d(layers, vul_data)
    print("\n  Plotting...")
    plot_0d(vul_data, kt_data, layers)
    print("\nDone.")


if __name__ == "__main__":
    main()
