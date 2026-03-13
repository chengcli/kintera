"""
Generate vertical abundance comparison plots:
  1. Chapman cycle: kintera-native vs VULCAN
  2. HD 189733b:    kintera-native vs VULCAN

Each kintera solver runs in a separate subprocess because kintera's global
species list can only be initialized once per process.

Usage:
    python3.11 plot_comparisons.py
"""
import os, sys, subprocess, pickle, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "VULCAN")


# ======================================================================
#  Subprocess runners
# ======================================================================

def run_kintera_chapman():
    """Run Chapman solver in a subprocess with ALL inputs matched to VULCAN:
    same cross-sections, stellar flux, atmospheric profile, and reactions.
    Only the solver (kintera's Beer-Lambert RT, chemistry, diffusion,
    implicit Euler) differs from VULCAN."""
    script = r'''
import os, sys, pickle, time, yaml
import numpy as np
import torch
import kintera as kt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "VULCAN")
AVOGADRO = 6.02214076e23
KB_CGS = 1.380649e-16
RGAS = 8.314462

# ── Step 1: Build YAML with VULCAN's cross-sections ──────────────────
def build_matched_yaml():
    """Create a chapman_cycle YAML using VULCAN's O2/O3 cross-sections."""
    O2 = np.loadtxt(os.path.join(VULCAN_DIR, "thermo/photo_cross/O2/O2_cross.csv"),
                    skiprows=2, delimiter=",")
    O3 = np.loadtxt(os.path.join(VULCAN_DIR, "thermo/photo_cross/O3/O3_cross.csv"),
                    skiprows=2, delimiter=",")

    def xsec_to_yaml_list(data):
        rows = []
        for r in data:
            rows.append([int(r[0]), float(f"{r[1]:.6e}"), float(f"{r[2]:.6e}")])
        return rows

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
            {"equation": "O2 <=> 2 O", "type": "photolysis",
             "cross-section": [{"format": "YAML",
                                "data": xsec_to_yaml_list(O2)}]},
            {"equation": "O3 <=> O2 + O", "type": "photolysis",
             "cross-section": [{"format": "YAML",
                                "data": xsec_to_yaml_list(O3)}]},
        ],
    }
    out_path = os.path.join(SCRIPT_DIR, "_chapman_matched.yaml")
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
    return out_path

# ── Step 2: Load VULCAN's stellar flux ───────────────────────────────
R_SUN = 6.957e10    # cm
AU_CM = 1.4959787e13  # cm
HC = 1.98644582e-9  # erg*nm (Planck const * speed of light)

def load_vulcan_stellar_flux(wl_target):
    """Read stellar flux file (ergs/cm2/s/nm at surface), scale to planet
    distance, convert to photon flux (photons/cm2/s/nm)."""
    sflux = np.loadtxt(os.path.join(VULCAN_DIR, "atm/stellar_flux/sflux_chapman.txt"),
                       skiprows=1)
    wl_src, flux_ergs = sflux[:, 0], sflux[:, 1]
    r_star, orbit_radius = 1.0, 4.6524e-3  # solar radii, AU
    flux_ergs = flux_ergs * (r_star * R_SUN / (AU_CM * orbit_radius)) ** 2
    photon_flux = flux_ergs * wl_src / HC
    return np.interp(wl_target, wl_src, photon_flux, left=0.0, right=0.0)

# ── Step 3: Beer-Lambert RT ──────────────────────────────────────────
def beer_lambert(n_absorbers, cross_sections, stellar_flux, dz, cos_zen):
    nz, nwave = len(dz), len(stellar_flux)
    alpha = np.zeros((nz, nwave))
    for n_sp, sigma in zip(n_absorbers, cross_sections):
        alpha += n_sp[:, None] * sigma[None, :]
    dtau = alpha * dz[:, None]
    tau = np.zeros((nz, nwave))
    tau[nz - 1] = dtau[nz - 1] / 2.0
    for j in range(nz - 2, -1, -1):
        tau[j] = tau[j + 1] + (dtau[j + 1] + dtau[j]) / 2.0
    return stellar_flux[None, :] * np.exp(-tau / cos_zen) / cos_zen

def diffusion_tendency(y, A, B, C):
    tend = A[:, None] * y.copy()
    tend[:-1] += B[:-1, None] * y[1:]
    tend[1:] += C[1:, None] * y[:-1]
    return tend

# ── Build matched YAML and load kintera ──────────────────────────────
yaml_path = build_matched_yaml()
opts = kt.KineticsOptions.from_yaml(yaml_path)
kinet = kt.Kinetics(opts)
species = opts.species()
nspecies = len(species)
stoich = kinet.stoich
idx = {sp: i for i, sp in enumerate(species)}

wl_np = kinet.buffer("photolysis.wavelength").numpy().copy()
xs_O2 = kinet.buffer("photolysis.cross_section_0").numpy().sum(axis=1)
xs_O3 = kinet.buffer("photolysis.cross_section_1").numpy().sum(axis=1)

# ── Load VULCAN's atmosphere ─────────────────────────────────────────
with open(os.path.join(VULCAN_DIR, "output/chapman_plot_cmp.vul"), "rb") as f:
    vdata = pickle.load(f)
v_atm = vdata["atm"]
nz = len(v_atm["pco"])
P_cgs = v_atm["pco"]
P_Pa = P_cgs * 0.1
T_K = 250.0
n_cgs = v_atm["n_0"]
dz = v_atm["dz"]
dzi = v_atm["dzi"]
Kzz = v_atm["Kzz"]

stellar_flux = load_vulcan_stellar_flux(wl_np)

n_total_molm3 = P_Pa / (RGAS * T_K)
C = np.zeros((nz, nspecies))
for j in range(nz):
    C[j, idx["N2"]] = 0.79 * n_total_molm3[j]
    C[j, idx["O2"]] = 0.21 * n_total_molm3[j]
    C[j, idx["O"]]  = 1e-10 * n_total_molm3[j]
    C[j, idx["O3"]] = 1e-8  * n_total_molm3[j]

active_idx = [idx["O2"], idx["O"], idx["O3"]]

y_dum = torch.zeros(nz, 1, dtype=torch.float64)
y_dum[:, 0] = torch.from_numpy(n_total_molm3.copy())
A, B, Cd = kt.diffusion_coefficients(y_dum,
    torch.from_numpy(Kzz * 1e-4).double(),
    torch.from_numpy(dzi * 0.01).double())
diff_A, diff_B, diff_C = A.numpy(), B.numpy(), Cd.numpy()

T_t = torch.full((nz,), T_K, dtype=torch.float64)
P_t = torch.from_numpy(P_Pa.copy()).double()
wl_t = torch.from_numpy(wl_np).double()
eye_ns = torch.eye(nspecies, dtype=torch.float64)
dt, dt_max = 1e-10, 1e6
cos_zen = 1.0
max_steps = 8000
rt_update_frq = 10

C_cgs = C * AVOGADRO * 1e-6
aflux = beer_lambert([C_cgs[:, idx["O2"]], C_cgs[:, idx["O3"]]],
                     [xs_O2, xs_O3], stellar_flux, dz, cos_zen)

t0 = time.time()
for step in range(max_steps):
    if step > 0 and step % rt_update_frq == 0:
        C_cgs = C * AVOGADRO * 1e-6
        sc = n_cgs / C_cgs.sum(axis=1)
        aflux = beer_lambert(
            [C_cgs[:, idx["O2"]] * sc, C_cgs[:, idx["O3"]] * sc],
            [xs_O2, xs_O3], stellar_flux, dz, cos_zen)

    C_t = torch.from_numpy(C.copy()).double()
    aflux_t = torch.from_numpy(aflux.T.copy()).double()
    rate, rc_ddC, _ = kinet.forward(T_t, P_t, C_t,
        {"wavelength": wl_t, "actinic_flux": aflux_t})
    chem_tend = torch.matmul(stoich, rate.unsqueeze(-1)).squeeze(-1).numpy()
    cvol = torch.ones(nz, dtype=torch.float64)
    jac_rxn = kinet.jacobian(T_t, C_t, cvol, rate, rc_ddC)
    chem_jac = torch.matmul(stoich, jac_rxn).numpy()

    diff_tend = np.zeros_like(C)
    diff_tend[:, active_idx] = diffusion_tendency(C[:, active_idx], diff_A, diff_B, diff_C)
    diff_jac = np.zeros((nz, nspecies, nspecies))
    for ai in active_idx:
        diff_jac[:, ai, ai] = diff_A.ravel()

    C_old = C.copy()
    try:
        delta = kt.evolve_implicit(
            torch.from_numpy(np.ascontiguousarray(chem_tend + diff_tend)).double(),
            eye_ns,
            torch.from_numpy(np.ascontiguousarray(chem_jac + diff_jac)).double(),
            dt).numpy()
    except RuntimeError:
        dt = max(dt * 0.5, 1e-14); continue

    C = np.maximum(C_old + delta, 1e-50)
    rc = np.max(np.abs(C - C_old) / (np.abs(C_old) + 1e-50))
    if rc < 0.3: dt = min(dt * 1.2, dt_max)
    elif rc > 1.0: dt = max(dt * 0.5, 1e-14)

    if step > 500 and step % 20 == 0:
        C_cgs2 = C * AVOGADRO * 1e-6
        sc2 = n_cgs / C_cgs2.sum(axis=1)
        af2 = beer_lambert(
            [C_cgs2[:, idx["O2"]] * sc2, C_cgs2[:, idx["O3"]] * sc2],
            [xs_O2, xs_O3], stellar_flux, dz, cos_zen)
        C_t2 = torch.from_numpy(C.copy()).double()
        r2, _, _ = kinet.forward(T_t, P_t, C_t2,
            {"wavelength": wl_t, "actinic_flux": torch.from_numpy(af2.T.copy()).double()})
        t2 = torch.matmul(stoich, r2.unsqueeze(-1)).squeeze(-1).numpy()
        t2[:, active_idx] += diffusion_tendency(C[:, active_idx], diff_A, diff_B, diff_C)
        if np.max(np.abs(t2) / (np.abs(C) + 1e-50)) < 1e-6:
            break

elapsed = time.time() - t0
nsteps = step + 1
ymix = C / C.sum(axis=1, keepdims=True)

os.remove(yaml_path)
out_path = sys.argv[1]
with open(out_path, "wb") as f:
    pickle.dump({"ymix": ymix, "P_cgs": P_cgs, "species": species,
                 "nsteps": nsteps, "elapsed": elapsed}, f)
print(f"Chapman: {nsteps} steps, {elapsed:.1f}s")
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


def run_kintera_hd189():
    """Run HD189 solver in a subprocess, return results via pickle."""
    script = r'''
import os, sys, pickle, time
import numpy as np
import torch
import kintera as kt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "VULCAN")
AVOGADRO = 6.02214076e23
KB_CGS = 1.380649e-16
RGAS = 8.314462

nz_max = 50
max_steps = 500

yaml_path = os.path.join(SCRIPT_DIR, "ncho_thermal.yaml")
opts = kt.KineticsOptions.from_yaml(yaml_path)
kinet = kt.Kinetics(opts)
species = opts.species()
nspecies = len(species)
stoich = kinet.stoich
idx = {sp: i for i, sp in enumerate(species)}

atm_file = os.path.join(VULCAN_DIR, "atm", "atm_HD189_Kzz.txt")
data = np.loadtxt(atm_file, skiprows=2)
P_cgs_full, T_K_full, Kzz_full = data[:, 0], data[:, 1], data[:, 2]
sel = np.linspace(0, len(P_cgs_full) - 1, nz_max, dtype=int)
P_cgs, T_K, Kzz_cgs = P_cgs_full[sel], T_K_full[sel], Kzz_full[sel]
nz = len(P_cgs)
P_Pa = P_cgs * 0.1
n_molm3 = P_Pa / (RGAS * T_K)

mu, gs = 2.35, 2140.0
H = KB_CGS * T_K / (mu * 1.66054e-24 * gs)
z = np.zeros(nz)
for j in range(1, nz):
    z[j] = z[j-1] + 0.5 * (H[j-1] + H[j]) * np.log(P_cgs[j-1] / P_cgs[j])
dzi = np.diff(z)
Kzz_intf = 0.5 * (Kzz_cgs[:-1] + Kzz_cgs[1:])

C = np.ones((nz, nspecies)) * 1e-50
for j in range(nz):
    n = n_molm3[j]
    if "H2" in idx:  C[j, idx["H2"]]  = 0.85 * n
    if "He" in idx:  C[j, idx["He"]]  = 0.15 * n
    if "H2O" in idx: C[j, idx["H2O"]] = 4.9e-4 * 0.5 * n
    if "CO" in idx:  C[j, idx["CO"]]  = 2.7e-4 * 0.5 * n
    if "CH4" in idx: C[j, idx["CH4"]] = 2.7e-4 * 0.5 * n
    if "N2" in idx:  C[j, idx["N2"]]  = 6.8e-5 * 0.5 * n
    if "NH3" in idx: C[j, idx["NH3"]] = 6.8e-5 * 0.5 * n
    if "CO2" in idx: C[j, idx["CO2"]] = 4.9e-4 * 0.01 * n
    if "H" in idx:   C[j, idx["H"]]   = 1e-8 * n
    for s in species:
        if s not in ["H2", "He", "H2O", "CO", "CH4", "N2", "NH3", "CO2", "H"]:
            if s in idx: C[j, idx[s]] = max(C[j, idx[s]], 1e-30 * n)

bulk = {"H2", "He"}
active_idx = [i for i, s in enumerate(species) if s not in bulk]

y_dum = torch.zeros(nz, 1, dtype=torch.float64)
y_dum[:, 0] = torch.from_numpy(n_molm3.copy())
A, B, Cd = kt.diffusion_coefficients(y_dum,
    torch.from_numpy(Kzz_intf * 1e-4).double(),
    torch.from_numpy(dzi * 0.01).double())
diff_A, diff_B, diff_C = A.numpy(), B.numpy(), Cd.numpy()

T_t = torch.from_numpy(T_K.copy()).double()
P_t = torch.from_numpy(P_Pa.copy()).double()
eye_ns = torch.eye(nspecies, dtype=torch.float64)
dt = 1e-8

t0 = time.time()
for step in range(max_steps):
    C_t = torch.from_numpy(C.copy()).double()
    rate, rc_ddC, _ = kinet.forward(T_t, P_t, C_t)
    chem_tend = torch.matmul(stoich, rate.unsqueeze(-1)).squeeze(-1).numpy()
    cvol = torch.ones(nz, dtype=torch.float64)
    jac_rxn = kinet.jacobian(T_t, C_t, cvol, rate, rc_ddC)
    chem_jac = torch.matmul(stoich, jac_rxn).numpy()

    diff_tend_act = diff_A[:, None] * C[:, active_idx].copy()
    diff_tend_act[:-1] += diff_B[:-1, None] * C[1:, :][:, active_idx]
    diff_tend_act[1:] += diff_C[1:, None] * C[:-1, :][:, active_idx]
    diff_tend = np.zeros_like(C)
    diff_tend[:, active_idx] = diff_tend_act
    diff_jac = np.zeros((nz, nspecies, nspecies))
    for ai in active_idx:
        diff_jac[:, ai, ai] = diff_A.ravel()

    C_old = C.copy()
    try:
        delta = kt.evolve_implicit(
            torch.from_numpy(np.ascontiguousarray(chem_tend + diff_tend)).double(),
            eye_ns,
            torch.from_numpy(np.ascontiguousarray(chem_jac + diff_jac)).double(),
            dt).numpy()
    except RuntimeError:
        dt = max(dt * 0.5, 1e-14); continue

    C = np.maximum(C_old + delta, 1e-50)
    rc = np.max(np.abs(C[:, active_idx] - C_old[:, active_idx]) /
                (np.abs(C_old[:, active_idx]) + 1e-50))
    if rc < 0.3: dt = min(dt * 1.5, 1e8)
    elif rc > 2.0: dt = max(dt * 0.5, 1e-14)

    if (step + 1) % 100 == 0:
        print(f"  step {step+1}/{max_steps}, dt={dt:.2e}")

elapsed = time.time() - t0
nsteps = step + 1
ymix = C / C.sum(axis=1, keepdims=True)

out_path = sys.argv[1]
with open(out_path, "wb") as f:
    pickle.dump({"ymix": ymix, "P_cgs": P_cgs, "T_K": T_K,
                 "species": species, "nsteps": nsteps, "elapsed": elapsed}, f)
print(f"HD189: {nsteps} steps, {elapsed:.1f}s")
'''
    out_pkl = os.path.join(SCRIPT_DIR, "_hd189_kt.pkl")
    script_path = os.path.join(SCRIPT_DIR, "_run_hd189.py")
    with open(script_path, "w") as f:
        f.write(script)

    result = subprocess.run(
        [sys.executable, script_path, out_pkl],
        capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=600,
        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    print(result.stdout.strip())
    if result.returncode != 0:
        print("STDERR:", result.stderr[-500:])
        raise RuntimeError("HD189 kintera solver failed")

    with open(out_pkl, "rb") as f:
        data = pickle.load(f)
    os.remove(script_path)
    os.remove(out_pkl)
    return data


def run_vulcan_chapman(nz=20):
    """Run VULCAN Chapman for comparison."""
    cfg_template = os.path.join(VULCAN_DIR, "vulcan_cfg_chapman.py")
    if not os.path.exists(cfg_template):
        raise FileNotFoundError("vulcan_cfg_chapman.py not found")

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
        f.write("# Pressure(dyne/cm2)  Temp(K)  Kzz(cm2/s)\n"
                "Pressure\tTemp\tKzz\n")
        for p in P:
            f.write(f"{p:.6e}\t250.0\t1.0e+06\n")

    out = os.path.join(VULCAN_DIR, "output", "chapman_plot_cmp.vul")
    if os.path.exists(out):
        os.remove(out)

    t0 = __import__("time").time()
    res = subprocess.run(
        [sys.executable, "vulcan.py"],
        capture_output=True, text=True, cwd=VULCAN_DIR, timeout=120)
    v_elapsed = __import__("time").time() - t0

    if res.returncode != 0:
        raise RuntimeError(f"VULCAN failed:\n{res.stderr[-300:]}")

    m = re.search(r'with (\d+) steps', res.stdout)
    v_steps = int(m.group(1)) if m else -1

    with open(out, "rb") as f:
        data = pickle.load(f)
    return data["variable"], data["atm"], v_steps, v_elapsed


def load_vulcan_hd189():
    """Try to load existing VULCAN HD189 output."""
    for vname in ["HD189-kintera.vul", "HD189.vul", "HD189-native-cmp.vul"]:
        vpath = os.path.join(VULCAN_DIR, "output", vname)
        if os.path.exists(vpath):
            with open(vpath, "rb") as f:
                vdata = pickle.load(f)
            return vdata["variable"], vdata["atm"], vname
    return None, None, None


# ======================================================================
#  Plotting
# ======================================================================

def plot_chapman():
    print("  Running kintera Chapman solver...")
    kt_data = run_kintera_chapman()

    ymix_kt = kt_data["ymix"]
    P_cgs = kt_data["P_cgs"]
    species = kt_data["species"]
    idx = {sp: i for i, sp in enumerate(species)}
    nsteps = kt_data["nsteps"]
    elapsed = kt_data["elapsed"]

    has_vulcan = False
    try:
        print("  Running VULCAN Chapman...")
        v_var, v_atm, v_steps, v_elapsed = run_vulcan_chapman(nz=20)
        v_species = v_var["species"]
        v_idx = {sp: i for i, sp in enumerate(v_species)}
        v_ymix = v_var["ymix"]
        v_pco = v_atm["pco"]
        has_vulcan = True
        print(f"  VULCAN: {v_steps} steps, {v_elapsed:.1f}s")
    except Exception as e:
        print(f"  VULCAN skipped: {e}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle("Chapman Cycle: Kintera-Native vs VULCAN", fontsize=14, fontweight="bold")

    for ax, sp, color in zip(axes, ["O", "O2", "O3"], ["#e74c3c", "#3498db", "#2ecc71"]):
        ax.semilogx(ymix_kt[:, idx[sp]], P_cgs, "-o", color=color,
                     markersize=4, label="kintera", linewidth=2)
        if has_vulcan and sp in v_idx:
            ax.semilogx(v_ymix[:, v_idx[sp]], v_pco, "--s", color="gray",
                         markersize=4, label="VULCAN", linewidth=1.5, alpha=0.8)
        ax.set_xlabel(f"{sp} mixing ratio", fontsize=12)
        ax.invert_yaxis()
        ax.set_yscale("log")
        if ax == axes[0]:
            ax.set_ylabel("Pressure (dyn/cm²)", fontsize=12)
        ax.set_title(sp, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    info = f"kintera: {nsteps} steps, {elapsed:.1f}s"
    if has_vulcan:
        info += f"  |  VULCAN: {v_steps} steps, {v_elapsed:.1f}s"
        info += f"  |  kintera {v_elapsed/elapsed:.1f}x faster"
    fig.text(0.5, -0.02, info, ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "chapman_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)
    return out


def plot_hd189():
    print("  Running kintera HD189 solver (500 steps)...")
    kt_data = run_kintera_hd189()

    ymix_kt = kt_data["ymix"]
    P_cgs = kt_data["P_cgs"]
    species = kt_data["species"]
    idx = {sp: i for i, sp in enumerate(species)}
    nsteps = kt_data["nsteps"]
    elapsed = kt_data["elapsed"]

    has_vulcan = False
    print("  Loading VULCAN HD189 reference...")
    v_var, v_atm, vname = load_vulcan_hd189()
    if v_var is not None:
        v_species = v_var["species"]
        v_idx = {sp: i for i, sp in enumerate(v_species)}
        v_ymix = v_var["ymix"]
        v_pco = v_atm["pco"]
        has_vulcan = True
        print(f"  VULCAN loaded from {vname}")
    else:
        print("  No VULCAN reference available")

    plot_species = [s for s in ["H2O", "CH4", "CO", "CO2", "H", "NH3", "HCN", "C2H2"]
                    if s in idx]

    ncols = 4
    nrows = max(1, (len(plot_species) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), sharey=True)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    fig.suptitle("HD 189733b: Kintera-Native vs VULCAN\n"
                 "(kintera: forward reactions only; VULCAN: forward + thermodynamic reverse)",
                 fontsize=13, fontweight="bold")

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12",
              "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]

    P_bar = P_cgs / 1e6

    for i, sp in enumerate(plot_species):
        ax = axes_flat[i]
        kt_mix = ymix_kt[:, idx[sp]]
        valid = kt_mix > 1e-45
        if valid.any():
            ax.semilogx(kt_mix[valid], P_bar[valid], "-o", color=colors[i % len(colors)],
                         markersize=3, label="kintera", linewidth=2)
        if has_vulcan and sp in v_idx:
            v_mix = v_ymix[:, v_idx[sp]]
            v_valid = v_mix > 1e-30
            if v_valid.any():
                ax.semilogx(v_mix[v_valid], v_pco[v_valid] / 1e6, "--", color="gray",
                             label="VULCAN", linewidth=1.5, alpha=0.8)
        ax.set_xlabel(f"{sp} mixing ratio", fontsize=11)
        ax.invert_yaxis()
        ax.set_yscale("log")
        if i % ncols == 0:
            ax.set_ylabel("Pressure (bar)", fontsize=12)
        ax.set_title(sp, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(len(plot_species), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "hd189_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)
    return out


# ======================================================================
#  Main
# ======================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Vertical Abundance Comparison Plots")
    print("=" * 70)

    print("\n--- Chapman Cycle ---")
    p1 = plot_chapman()

    print("\n--- HD 189733b ---")
    p2 = plot_hd189()

    print("\n" + "=" * 70)
    print(f"Chapman: {p1}")
    print(f"HD189:   {p2}")
    print("=" * 70)
