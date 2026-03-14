#!/usr/bin/env python3
"""Chapman cycle: kintera-native vs VULCAN comparison.

Uses kintera's evolve_ros2() (2nd-order Rosenbrock, same as VULCAN)
for chemistry and diffusion_tendency() for transport, with hydrostatic
normalization after each step.

Runs kintera in a subprocess (kintera's global species list can only
be initialized once per process), then plots against VULCAN reference.

Usage:
    python3.11 run_comparison.py
"""
import os, sys, subprocess, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "..", "VULCAN")


def run_kintera():
    script = r'''
import os, sys, pickle, time, yaml
import numpy as np
import torch
import kintera as kt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "..", "VULCAN")
AVOGADRO = 6.02214076e23
RGAS = 8.314462
R_SUN = 6.957e10; AU_CM = 1.4959787e13; HC = 1.98644582e-9

def build_matched_yaml():
    O2 = np.loadtxt(os.path.join(VULCAN_DIR, "thermo/photo_cross/O2/O2_cross.csv"),
                    skiprows=2, delimiter=",")
    O3 = np.loadtxt(os.path.join(VULCAN_DIR, "thermo/photo_cross/O3/O3_cross.csv"),
                    skiprows=2, delimiter=",")
    O2_br1 = np.where(O2[:, 0] >= 176, 1.0, 0.0)
    O3_br1 = np.where(O3[:, 0] <= 310, 0.0,
             np.where(O3[:, 0] <= 320, (O3[:, 0] - 310) / 10 * 0.1,
             np.where(O3[:, 0] <= 340, 0.1 + (O3[:, 0] - 320) / 20 * 0.9, 1.0)))
    O2_eff = O2.copy(); O3_eff = O3.copy()
    O2_eff[:, 2] *= O2_br1; O3_eff[:, 2] *= O3_br1
    def to_yaml(data):
        return [[int(r[0]), float(f"{r[1]:.6e}"), float(f"{r[2]:.6e}")] for r in data]
    cfg = {
        "reference-state": {"Tref": 300.0, "Pref": 1.0e5},
        "species": [
            {"name": "N2", "composition": {"N": 2}, "cv_R": 2.5},
            {"name": "O2", "composition": {"O": 2}, "cv_R": 2.5},
            {"name": "O",  "composition": {"O": 1}, "cv_R": 1.5},
            {"name": "O3", "composition": {"O": 3}, "cv_R": 3.0},
        ],
        "reactions": [
            {"equation": "O + O2 <=> O3", "type": "arrhenius",
             "rate-constant": {"A": 1.7e-14, "b": -2.4, "Ea_R": 0.0}},
            {"equation": "O + O3 <=> 2 O2", "type": "arrhenius",
             "rate-constant": {"A": 8.0e-12, "b": 0.0, "Ea_R": 2060.0}},
            {"equation": "O + O2 + M <=> O3 + M", "type": "three-body",
             "rate-constant": {"A": 6.0e-34, "b": -2.4, "Ea_R": 0.0},
             "efficiencies": {"N2": 1.0, "O2": 1.0}},
            {"equation": "O2 => 2 O", "type": "photolysis",
             "cross-section": [{"format": "YAML", "data": to_yaml(O2_eff)}]},
            {"equation": "O3 => O2 + O", "type": "photolysis",
             "cross-section": [{"format": "YAML", "data": to_yaml(O3_eff)}]},
        ],
    }
    out_path = os.path.join(SCRIPT_DIR, "_chapman_matched.yaml")
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
    return out_path

def load_stellar_flux(wl_target):
    sflux = np.loadtxt(os.path.join(VULCAN_DIR, "atm/stellar_flux/sflux_chapman.txt"),
                       skiprows=1)
    wl_src, flux_ergs = sflux[:, 0], sflux[:, 1]
    flux_ergs *= (1.0 * R_SUN / (AU_CM * 4.6524e-3)) ** 2
    return np.interp(wl_target, wl_src, flux_ergs * wl_src / HC, left=0, right=0)

def beer_lambert(n_absorbers, cross_sections, stellar_flux, dz):
    nz, nw = len(dz), len(stellar_flux)
    alpha = np.zeros((nz, nw))
    for n_sp, sigma in zip(n_absorbers, cross_sections):
        alpha += n_sp[:, None] * sigma[None, :]
    dtau = alpha * dz[:, None]
    tau = np.zeros((nz, nw))
    tau[-1] = dtau[-1] / 2.0
    for j in range(nz - 2, -1, -1):
        tau[j] = tau[j+1] + (dtau[j+1] + dtau[j]) / 2.0
    return stellar_flux[None, :] * np.exp(-tau)

yaml_path = build_matched_yaml()
opts = kt.KineticsOptions.from_yaml(yaml_path)
kinet = kt.Kinetics(opts)
species = opts.species(); nspecies = len(species)
stoich = kinet.stoich
idx = {sp: i for i, sp in enumerate(species)}

wl_np = kinet.buffer("photolysis.wavelength").numpy().copy()
xs_O2 = kinet.buffer("photolysis.cross_section_0").numpy()[:, 0]
xs_O3 = kinet.buffer("photolysis.cross_section_1").numpy()[:, 0]
xs_N2_scat = 4.577e-21 / wl_np**4
xs_O2_scat = 4.12e-21 / wl_np**4

with open(os.path.join(VULCAN_DIR, "output/chapman_plot_cmp.vul"), "rb") as f:
    vdata = pickle.load(f)
v_atm = vdata["atm"]
nz = len(v_atm["pco"])
P_cgs = v_atm["pco"]; P_Pa = P_cgs * 0.1
T_K = 250.0; n_cgs = v_atm["n_0"]
dz = v_atm["dz"]; dzi = v_atm["dzi"]; Kzz = v_atm["Kzz"]
stellar_flux = load_stellar_flux(wl_np)

n_molm3 = P_Pa / (RGAS * T_K)
C = np.zeros((nz, nspecies))
for j in range(nz):
    C[j, idx["N2"]] = 0.79 * n_molm3[j]
    C[j, idx["O2"]] = 0.21 * n_molm3[j]
    C[j, idx["O"]]  = 1e-10 * n_molm3[j]
    C[j, idx["O3"]] = 1e-8  * n_molm3[j]

T_t = torch.full((nz,), T_K, dtype=torch.float64)
P_t = torch.from_numpy(P_Pa.copy()).double()
wl_t = torch.from_numpy(wl_np).double()
Kzz_si = torch.from_numpy(Kzz * 1e-4).double()
dzi_si = torch.from_numpy(dzi * 0.01).double()

gamma_ros = 1.0 + 1.0 / 2.0**0.5
dt, dt_max = 1e-10, 1e6
max_steps = 8000; rt_update_frq = 10
rtol = 0.25; dt_var_min, dt_var_max = 0.5, 2.0; dt_min = 1e-14; atol = 0.1
C_FLOOR = 1e-50

C_cgs = C * AVOGADRO * 1e-6
aflux = beer_lambert(
    [C_cgs[:, idx["O2"]], C_cgs[:, idx["O3"]],
     C_cgs[:, idx["N2"]], C_cgs[:, idx["O2"]]],
    [xs_O2, xs_O3, xs_N2_scat, xs_O2_scat], stellar_flux, dz)

def compute_ros2_k1(rate, jac_rxn, dt_val):
    ns = stoich.shape[0]
    eye = torch.eye(ns, dtype=rate.dtype)
    SJ = stoich.matmul(jac_rxn)
    W = eye / (gamma_ros * dt_val) - SJ
    f1 = stoich.matmul(rate.unsqueeze(-1)).squeeze(-1)
    return torch.linalg.solve(W, f1)

t0 = time.time(); rejects = 0; accepted = 0
ymix_prev = None; longdy = 1.0
for step in range(max_steps):
    if step > 0 and step % rt_update_frq == 0:
        C_cgs = C * AVOGADRO * 1e-6
        sc = n_cgs / C_cgs.sum(axis=1)
        aflux = beer_lambert(
            [C_cgs[:, idx["O2"]]*sc, C_cgs[:, idx["O3"]]*sc,
             C_cgs[:, idx["N2"]]*sc, C_cgs[:, idx["O2"]]*sc],
            [xs_O2, xs_O3, xs_N2_scat, xs_O2_scat], stellar_flux, dz)

    C_t = torch.from_numpy(np.maximum(C, C_FLOOR)).double()
    aflux_t = torch.from_numpy(aflux.T.copy()).double()
    photo = {"wavelength": wl_t, "actinic_flux": aflux_t}
    rate1, rc_ddC, _ = kinet.forward(T_t, P_t, C_t, photo)
    cvol = torch.ones(nz, dtype=torch.float64)
    jac_rxn = kinet.jacobian(T_t, C_t, cvol, rate1, rc_ddC)

    k1 = compute_ros2_k1(rate1, jac_rxn, dt)
    C_mid = np.maximum(C + (k1 / gamma_ros).numpy(), C_FLOOR)
    C_mid_t = torch.from_numpy(C_mid).double()
    rate2, _, _ = kinet.forward(T_t, P_t, C_mid_t, photo)

    delta_chem, error_chem = kt.evolve_ros2(rate1, rate2, stoich, jac_rxn, dt)
    diff_tend = kt.diffusion_tendency(C_t, Kzz_si, dzi_si)

    delta = (delta_chem + dt * diff_tend).numpy()
    C_old = C.copy()
    C_trial = np.maximum(C_old + delta, C_FLOOR)

    ymix_trial = C_trial / np.maximum(C_trial.sum(axis=1, keepdims=True), 1e-100)
    err_np = np.abs(error_chem.numpy())
    err_np[ymix_trial < 1e-22] = 0
    err_np[C_trial < atol] = 0
    mask = C_trial > 0
    ros_delta = np.max(err_np[mask] / C_trial[mask]) if mask.any() else 0

    if np.any(C_old + delta < 0) or ros_delta > rtol:
        h_factor = 0.9 * (rtol / max(ros_delta, 1e-30)) ** 0.5 if ros_delta > rtol else 0.5
        h_factor = np.clip(h_factor, dt_var_min, 1.0)
        dt = max(dt * h_factor, dt_min)
        rejects += 1
        continue

    # Hydrostatic normalization: enforce total density = n_molm3 per layer
    C = n_molm3[:, None] * ymix_trial
    C = np.maximum(C, C_FLOOR)

    accepted += 1

    # Adaptive dt from Ros2 error estimate (VULCAN-style)
    if ros_delta == 0:
        ros_delta = 0.01 * rtol
    h_factor = 0.9 * (rtol / ros_delta) ** 0.5
    h_factor = np.clip(h_factor, dt_var_min, dt_var_max)
    dt = np.clip(dt * h_factor, dt_min, dt_max)

    # Convergence check: long-term change in mixing ratios (VULCAN-style)
    if accepted % 100 == 0:
        ymix_now = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-100)
        if ymix_prev is not None:
            sig = ymix_now > 1e-5
            if sig.any():
                longdy = np.max(np.abs(ymix_now[sig] - ymix_prev[sig]) / ymix_now[sig])
        ymix_prev = ymix_now.copy()
        if accepted > 500 and longdy < 0.01:
            break

elapsed = time.time() - t0
nsteps = step + 1
ymix = C / C.sum(axis=1, keepdims=True)
os.remove(yaml_path)
out_path = sys.argv[1]
with open(out_path, "wb") as f:
    pickle.dump({"ymix": ymix, "P_cgs": P_cgs, "species": species,
                 "nsteps": nsteps, "elapsed": elapsed, "rejects": rejects}, f)
print(f"Chapman: {nsteps} steps ({rejects} rejected), {elapsed:.1f}s")
'''
    out_pkl = os.path.join(SCRIPT_DIR, "_chapman_kt.pkl")
    script_path = os.path.join(SCRIPT_DIR, "_run_chapman.py")
    with open(script_path, "w") as f:
        f.write(script)

    result = subprocess.run(
        [sys.executable, script_path, out_pkl],
        capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=120,
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    print(result.stdout.strip())
    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:])
        raise RuntimeError("Chapman kintera solver failed")

    with open(out_pkl, "rb") as f:
        data = pickle.load(f)
    os.remove(script_path)
    os.remove(out_pkl)
    return data


def run_vulcan(nz=20):
    cfg_template = os.path.join(VULCAN_DIR, "vulcan_cfg_chapman.py")
    if not os.path.exists(cfg_template):
        return None, None, 0, 0.0

    with open(cfg_template) as f:
        cfg = f.read()
    cfg = cfg.replace("use_Kzz = False", "use_Kzz = True")
    cfg = cfg.replace("ini_update_photo_frq = 999999", "ini_update_photo_frq = 100")
    cfg = cfg.replace("final_update_photo_frq = 999999", "final_update_photo_frq = 5")
    cfg = cfg.replace("out_name = 'chapman.vul'", "out_name = 'chapman_plot_cmp.vul'")
    cfg = cfg.replace("P_b = 1e-1", "P_b = 1e1")
    cfg = cfg.replace("P_t = 1e-4", "P_t = 1e-2")
    cfg = cfg.replace("nz = 10", f"nz = {nz}")

    with open(os.path.join(VULCAN_DIR, "vulcan_cfg.py"), "w") as f:
        f.write(cfg)
    P = np.logspace(np.log10(1e1), np.log10(1e-2), nz)
    with open(os.path.join(VULCAN_DIR, "atm", "atm_chapman.txt"), "w") as f:
        f.write("# Pressure(dyne/cm2)  Temp(K)  Kzz(cm2/s)\nPressure\tTemp\tKzz\n")
        for p in P:
            f.write(f"{p:.6e}\t250.0\t1.0e+06\n")

    out = os.path.join(VULCAN_DIR, "output", "chapman_plot_cmp.vul")
    if os.path.exists(out):
        os.remove(out)

    import time as _time
    t0 = _time.time()
    res = subprocess.run([sys.executable, "vulcan.py"],
                         capture_output=True, text=True, cwd=VULCAN_DIR, timeout=120)
    v_elapsed = _time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(f"VULCAN failed:\n{res.stderr[-300:]}")
    with open(out, "rb") as f:
        vdata = pickle.load(f)
    v_steps = vdata["parameter"].get("count", 0)
    return vdata["variable"], vdata["atm"], v_steps, v_elapsed


def plot():
    print("  Running kintera Chapman solver...")
    kt_data = run_kintera()
    ymix_kt = kt_data["ymix"]; P_cgs = kt_data["P_cgs"]
    species = kt_data["species"]
    idx = {sp: i for i, sp in enumerate(species)}

    has_vulcan = False
    try:
        print("  Running VULCAN Chapman...")
        v_var, v_atm, v_steps, v_elapsed = run_vulcan(nz=20)
        if v_var is not None:
            v_species = v_var["species"]
            v_idx = {sp: i for i, sp in enumerate(v_species)}
            v_ymix = v_var["ymix"]; v_pco = v_atm["pco"]
            has_vulcan = True
            print(f"  VULCAN: {v_steps} steps, {v_elapsed:.1f}s")
    except Exception as e:
        print(f"  VULCAN skipped: {e}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle("Chapman Cycle: Kintera-Native vs VULCAN",
                 fontsize=14, fontweight="bold")

    for ax, sp, color in zip(axes, ["O", "O2", "O3"],
                              ["#e74c3c", "#3498db", "#2ecc71"]):
        ax.semilogx(ymix_kt[:, idx[sp]], P_cgs, "-o", color=color,
                     markersize=4, label="kintera", linewidth=2)
        if has_vulcan and sp in v_idx:
            ax.semilogx(v_ymix[:, v_idx[sp]], v_pco, "--s", color="gray",
                         markersize=4, label="VULCAN", linewidth=1.5, alpha=0.8)
        ax.set_xlabel(f"{sp} mixing ratio", fontsize=12)
        ax.invert_yaxis(); ax.set_yscale("log")
        if ax == axes[0]:
            ax.set_ylabel("Pressure (dyn/cm²)", fontsize=12)
        ax.set_title(sp, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    info = f"kintera: {kt_data['nsteps']} steps, {kt_data['elapsed']:.1f}s"
    if has_vulcan:
        info += f"  |  VULCAN: {v_steps} steps, {v_elapsed:.1f}s"
    fig.text(0.5, -0.02, info, ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "chapman_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)
    return out


if __name__ == "__main__":
    plot()
