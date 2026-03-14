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
    """Run Chapman solver using kintera-native functions:
    kt.Kinetics.forward() for rates, kt.evolve_implicit() for chemistry,
    kt.diffusion_tendency() for transport, with IMEX operator splitting."""
    script = r'''
import os, sys, pickle, time, yaml
import numpy as np
import torch
import kintera as kt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "VULCAN")
AVOGADRO = 6.02214076e23
RGAS = 8.314462

# ── Build YAML with VULCAN's cross-sections ──────────────────────────
def build_matched_yaml():
    O2 = np.loadtxt(os.path.join(VULCAN_DIR, "thermo/photo_cross/O2/O2_cross.csv"),
                    skiprows=2, delimiter=",")
    O3 = np.loadtxt(os.path.join(VULCAN_DIR, "thermo/photo_cross/O3/O3_cross.csv"),
                    skiprows=2, delimiter=",")

    O2_br1 = np.where(O2[:, 0] >= 176, 1.0, 0.0)
    O3_br1 = np.where(O3[:, 0] <= 310, 0.0,
             np.where(O3[:, 0] <= 320, (O3[:, 0] - 310) / (320 - 310) * 0.1,
             np.where(O3[:, 0] <= 340, 0.1 + (O3[:, 0] - 320) / (340 - 320) * 0.9,
                      1.0)))

    O2_eff = O2.copy(); O3_eff = O3.copy()
    O2_eff[:, 2] *= O2_br1; O3_eff[:, 2] *= O3_br1

    def xsec_to_yaml_list(data):
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
             "cross-section": [{"format": "YAML",
                                "data": xsec_to_yaml_list(O2_eff)}]},
            {"equation": "O3 => O2 + O", "type": "photolysis",
             "cross-section": [{"format": "YAML",
                                "data": xsec_to_yaml_list(O3_eff)}]},
        ],
    }
    out_path = os.path.join(SCRIPT_DIR, "_chapman_matched.yaml")
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
    return out_path

# ── Beer-Lambert RT ──────────────────────────────────────────────────
R_SUN = 6.957e10; AU_CM = 1.4959787e13; HC = 1.98644582e-9

def load_vulcan_stellar_flux(wl_target):
    sflux = np.loadtxt(os.path.join(VULCAN_DIR, "atm/stellar_flux/sflux_chapman.txt"),
                       skiprows=1)
    wl_src, flux_ergs = sflux[:, 0], sflux[:, 1]
    r_star, orbit_radius = 1.0, 4.6524e-3
    flux_ergs = flux_ergs * (r_star * R_SUN / (AU_CM * orbit_radius)) ** 2
    photon_flux = flux_ergs * wl_src / HC
    return np.interp(wl_target, wl_src, photon_flux, left=0.0, right=0.0)

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

# ── Build YAML and load kintera ──────────────────────────────────────
yaml_path = build_matched_yaml()
opts = kt.KineticsOptions.from_yaml(yaml_path)
kinet = kt.Kinetics(opts)
species = opts.species()
nspecies = len(species)
stoich = kinet.stoich
idx = {sp: i for i, sp in enumerate(species)}

wl_np = kinet.buffer("photolysis.wavelength").numpy().copy()
xs_O2 = kinet.buffer("photolysis.cross_section_0").numpy()[:, 0]
xs_O3 = kinet.buffer("photolysis.cross_section_1").numpy()[:, 0]
xs_N2_scat = 4.577e-21 / wl_np**4
xs_O2_scat = 4.12e-21 / wl_np**4

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

T_t = torch.full((nz,), T_K, dtype=torch.float64)
P_t = torch.from_numpy(P_Pa.copy()).double()
wl_t = torch.from_numpy(wl_np).double()

Kzz_si = torch.from_numpy(Kzz * 1e-4).double()
dzi_si = torch.from_numpy(dzi * 0.01).double()

dt, dt_max = 1e-10, 1e6
cos_zen = 1.0
max_steps = 8000
rt_update_frq = 10
rc_target = 0.3
dt_ceil = float('inf')

C_cgs = C * AVOGADRO * 1e-6
aflux = beer_lambert(
    [C_cgs[:, idx["O2"]], C_cgs[:, idx["O3"]],
     C_cgs[:, idx["N2"]], C_cgs[:, idx["O2"]]],
    [xs_O2, xs_O3, xs_N2_scat, xs_O2_scat],
    stellar_flux, dz, cos_zen)

t0 = time.time()
rejects = 0
for step in range(max_steps):
    if step > 0 and step % rt_update_frq == 0:
        C_cgs = C * AVOGADRO * 1e-6
        sc = n_cgs / C_cgs.sum(axis=1)
        aflux = beer_lambert(
            [C_cgs[:, idx["O2"]] * sc, C_cgs[:, idx["O3"]] * sc,
             C_cgs[:, idx["N2"]] * sc, C_cgs[:, idx["O2"]] * sc],
            [xs_O2, xs_O3, xs_N2_scat, xs_O2_scat],
            stellar_flux, dz, cos_zen)

    C_t = torch.from_numpy(np.maximum(C, 1e-50)).double()
    aflux_t = torch.from_numpy(aflux.T.copy()).double()

    rate, rc_ddC, _ = kinet.forward(T_t, P_t, C_t,
        {"wavelength": wl_t, "actinic_flux": aflux_t})
    cvol = torch.ones(nz, dtype=torch.float64)
    jac_rxn = kinet.jacobian(T_t, C_t, cvol, rate, rc_ddC)

    # Chemistry step: kintera-native implicit Euler
    delta_chem = kt.evolve_implicit(rate, stoich, jac_rxn, dt)

    # Diffusion step: kintera-native explicit tendency
    diff_tend = kt.diffusion_tendency(C_t, Kzz_si, dzi_si)
    delta_diff = dt * diff_tend

    delta = (delta_chem + delta_diff).numpy()
    C_old = C.copy()
    C_trial = np.maximum(C_old + delta, 1e-50)
    rc = np.max(np.abs(C_trial - C_old) / (np.abs(C_old) + 1e-50))

    if rc > 2.0:
        dt_ceil = dt * 0.8
        dt = max(dt * 0.5, 1e-14)
        rejects += 1
        continue

    C = C_trial

    grow = min(0.9 * rc_target / max(rc, 1e-10), 2.0)
    grow = max(grow, 0.3)
    dt = min(dt * grow, dt_max, dt_ceil)
    dt_ceil = min(dt_ceil * 1.05, dt_max)

    if step > 500 and step % 20 == 0:
        C_cgs2 = C * AVOGADRO * 1e-6
        sc2 = n_cgs / C_cgs2.sum(axis=1)
        af2 = beer_lambert(
            [C_cgs2[:, idx["O2"]] * sc2, C_cgs2[:, idx["O3"]] * sc2],
            [xs_O2, xs_O3], stellar_flux, dz, cos_zen)
        C_t2 = torch.from_numpy(np.maximum(C, 1e-50)).double()
        r2, _, _ = kinet.forward(T_t, P_t, C_t2,
            {"wavelength": wl_t, "actinic_flux": torch.from_numpy(af2.T.copy()).double()})
        t2 = torch.matmul(stoich, r2.unsqueeze(-1)).squeeze(-1)
        t2 = t2 + kt.diffusion_tendency(C_t2, Kzz_si, dzi_si)
        if torch.max(torch.abs(t2) / (torch.abs(C_t2) + 1e-50)).item() < 1e-6:
            break

elapsed = time.time() - t0
nsteps = step + 1
ymix = C / C.sum(axis=1, keepdims=True)

os.remove(yaml_path)
out_path = sys.argv[1]
with open(out_path, "wb") as f:
    pickle.dump({"ymix": ymix, "P_cgs": P_cgs, "species": species,
                 "nsteps": nsteps, "elapsed": elapsed}, f)
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


def run_kintera_hd189():
    """Run HD189 thermal-chemistry + photolysis solver in a subprocess.

    Uses implicit Euler with exponentially growing timesteps, coupled
    sparse chemistry-diffusion solve, and Beer-Lambert photolysis RT
    using VULCAN's cross-section and stellar flux data.
    """
    script = r'''
import os, sys, pickle, time
import numpy as np
import torch
import kintera as kt
from scipy.sparse import bsr_matrix
from scipy.sparse.linalg import spsolve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VULCAN_DIR = os.path.join(SCRIPT_DIR, "VULCAN")
AVO = 6.02214076e23
RGAS = 8.314462
HC_NM = 1.98644582e-9   # Planck*c in erg*nm
R_SUN = 6.957e10         # cm
AU_CM = 1.4959787e13     # cm

yaml_path = os.path.join(SCRIPT_DIR, "ncho_thermal.yaml")
opts = kt.KineticsOptions.from_yaml(yaml_path)
kinet = kt.Kinetics(opts)
species = opts.species()
nspecies = len(species)
stoich = kinet.stoich
idx = {sp: i for i, sp in enumerate(species)}

with open(os.path.join(VULCAN_DIR, "output", "HD189-kintera.vul"), "rb") as f:
    vul = pickle.load(f)
v = vul["variable"]; atm = vul["atm"]
vul_species = v["species"]
y_ini_cgs = np.array(v["y_ini"])
P_cgs = np.array(atm["pco"])
T_K = np.array(atm["Tco"])
Kzz_cgs = np.array(atm["Kzz"])
dzi_cgs = np.array(atm["dzi"])
dz_cgs = np.array(atm["dz"])
M_cgs = np.array(atm["M"])
nz = len(P_cgs)

P_Pa = P_cgs * 0.1
n_molm3 = P_Pa / (RGAS * T_K)

vul_idx = {sp: i for i, sp in enumerate(vul_species)}
C = np.ones((nz, nspecies)) * 1e-50
for sp in species:
    if sp in vul_idx:
        C[:, idx[sp]] = np.maximum(y_ini_cgs[:, vul_idx[sp]] * 1e6 / AVO, 1e-50)

# Diffusion coefficients
y_dum = torch.zeros(nz, 1, dtype=torch.float64)
y_dum[:, 0] = torch.from_numpy(n_molm3.copy())
A, B, Cd = kt.diffusion_coefficients(y_dum,
    torch.from_numpy(Kzz_cgs * 1e-4).double(),
    torch.from_numpy(dzi_cgs * 0.01).double())
dA_np = A.numpy().ravel()
dB_np = B.numpy().ravel()
dC_np = Cd.numpy().ravel()

T_t = torch.from_numpy(T_K.copy()).double()
P_t = torch.from_numpy(P_Pa.copy()).double()

# ── Photolysis setup ─────────────────────────────────────────────────
bins = np.array(v["bins"])       # wavelength grid (nm)
nwav = len(bins)
cross_abs = v["cross"]           # {species: sigma_abs(nwav)} in cm²
cross_J = v["cross_J"]           # {(species, branch_idx): sigma_branch(nwav)}
n_branch = v["n_branch"]         # {species: num_branches}

# Build photolysis reaction table:
#   photo_parent[r] = kintera species index of the parent
#   photo_stoich[r, s] = net stoichiometric coefficient for species s
#   photo_sigma[r, w] = branching cross-section in cm² at wavelength w
# VULCAN J_sp contains (species, branch_index) tuples.
# We get the products from cross_J branch compositions stored in VULCAN.

# Parse VULCAN's NCHO network file to get photolysis products
import re
photo_rxns = []  # list of (parent, products_dict, branch_sigma)
network_file = os.path.join(VULCAN_DIR, "thermo", "NCHO_photo_network.txt")
in_photo = False
with open(network_file) as nf:
    for raw in nf:
        line = raw.strip()
        if "# reverse stops" in line or "# photo disscoiation" in line:
            in_photo = True; continue
        if not in_photo or not line or line.startswith("#") or "[" not in line:
            continue
        id_m = re.match(r"(\d+)\s+\[", line)
        if not id_m: continue
        rxn_id = int(id_m.group(1))
        if rxn_id % 2 == 0: continue
        eq_m = re.search(r"\[\s*(.*?)\s*\]", line)
        if not eq_m: continue
        eq = eq_m.group(1).strip()
        parts = eq.split("->")
        if len(parts) != 2: continue
        reactants = [s.strip() for s in parts[0].split("+") if s.strip()]
        products = [s.strip() for s in parts[1].split("+") if s.strip()]
        after = line[eq_m.end():].strip().split()
        if len(after) < 2: continue
        parent = after[0]
        try:
            br_idx = int(after[1])
        except ValueError:
            continue
        # Get the branching cross-section
        key = (parent, br_idx)
        if key not in cross_J: continue
        if parent not in idx: continue
        sigma_br = np.array(cross_J[key])
        prod_dict = {}
        for p in products:
            if p == "M": continue
            cm = re.match(r"(\d+)\s+(\S+)", p)
            sp_name = cm.group(2) if cm else p
            coeff = int(cm.group(1)) if cm else 1
            if sp_name in idx:
                prod_dict[idx[sp_name]] = prod_dict.get(idx[sp_name], 0) + coeff
        photo_rxns.append((idx[parent], prod_dict, sigma_br))

n_photo = len(photo_rxns)
print(f"Loaded {n_photo} photolysis reactions")

photo_parent = np.array([r[0] for r in photo_rxns], dtype=int)
photo_stoich = np.zeros((n_photo, nspecies))
photo_sigma = np.zeros((n_photo, nwav))
for r, (par, prod, sig) in enumerate(photo_rxns):
    photo_stoich[r, par] = -1.0
    for si, coeff in prod.items():
        photo_stoich[r, si] += coeff
    photo_sigma[r] = sig

# Absorption cross-sections for RT (all photo-active species)
absorbers = []
for sp_name, sigma in cross_abs.items():
    if sp_name in vul_idx:
        absorbers.append((vul_idx[sp_name], np.array(sigma)))
n_abs = len(absorbers)
print(f"RT absorbers: {n_abs} species, {nwav} wavelength bins")

# Stellar flux: load and scale to planet
sflux_file = os.path.join(VULCAN_DIR, "atm/stellar_flux/sflux-HD189_Moses11.txt")
sflux_raw = np.loadtxt(sflux_file, skiprows=1)
wl_star, flux_star = sflux_raw[:, 0], sflux_raw[:, 1]
r_star = 0.756   # HD 189733 radius in solar radii
orbit_r = 0.031  # AU
flux_star *= (r_star * R_SUN / (AU_CM * orbit_r))**2
photon_flux_star = flux_star * wl_star / HC_NM
stellar_photon = np.interp(bins, wl_star, photon_flux_star, left=0, right=0)

def compute_actinic_flux(y_cgs):
    """Beer-Lambert RT: compute actinic flux at each layer center."""
    alpha = np.zeros((nz, nwav))
    for vi, sigma in absorbers:
        alpha += y_cgs[:, vi:vi+1] * sigma[None, :]
    dtau = alpha * dz_cgs[:, None]
    tau = np.zeros((nz, nwav))
    tau[-1] = dtau[-1] / 2.0
    for j in range(nz - 2, -1, -1):
        tau[j] = tau[j + 1] + (dtau[j + 1] + dtau[j]) / 2.0
    return stellar_photon[None, :] * np.exp(-tau)

def compute_photo_tend(aflux, C_np):
    """Compute photolysis tendency (mol/m³/s) and diagonal Jacobian."""
    C_cgs = C_np * AVO * 1e-6
    J = np.zeros((n_photo, nz))
    dbin = np.diff(np.concatenate([[bins[0]], 0.5*(bins[:-1]+bins[1:]), [bins[-1]]]))
    for r in range(n_photo):
        J[r] = np.sum(photo_sigma[r][None, :] * aflux * dbin[None, :], axis=1)
    photo_tend = np.zeros_like(C_np)
    photo_jac_diag = np.zeros_like(C_np)
    for r in range(n_photo):
        par = photo_parent[r]
        rate_r = J[r] * C_cgs[:, par]
        rate_r_si = rate_r * 1e6 / AVO
        for s in range(nspecies):
            if photo_stoich[r, s] != 0:
                photo_tend[:, s] += photo_stoich[r, s] * rate_r_si
        photo_jac_diag[:, par] -= J[r] * 1e6 / AVO
    return photo_tend, photo_jac_diag

# ── Solver ───────────────────────────────────────────────────────────
max_steps = 2000
dt0 = 1e-6
growth = 1.05
dt_max = 1e10
ns = nspecies
N_full = nz * ns
rt_update_frq = 20

nnz_blocks = nz + 2 * (nz - 1)
indptr_bsr = np.zeros(nz + 1, dtype=np.int32)
indices_bsr = np.zeros(nnz_blocks, dtype=np.int32)
base_data = np.zeros((nnz_blocks, ns, ns))
diag_slot = np.zeros(nz, dtype=np.int32)

blk = 0
for j in range(nz):
    indptr_bsr[j] = blk
    if j > 0:
        indices_bsr[blk] = j - 1
        base_data[blk] = -dC_np[j] * np.eye(ns)
        blk += 1
    diag_slot[j] = blk
    indices_bsr[blk] = j
    blk += 1
    if j < nz - 1:
        indices_bsr[blk] = j + 1
        base_data[blk] = -dB_np[j] * np.eye(ns)
        blk += 1
indptr_bsr[nz] = blk

y_cgs_cur = y_ini_cgs.copy()
aflux = compute_actinic_flux(y_cgs_cur)

t0 = time.time()
t_phys = 0.0

for step in range(max_steps):
    dt = min(dt0 * growth ** step, dt_max)

    if step > 0 and step % rt_update_frq == 0:
        y_cgs_cur = np.zeros_like(y_ini_cgs)
        for sp in species:
            if sp in vul_idx:
                y_cgs_cur[:, vul_idx[sp]] = C[:, idx[sp]] * AVO * 1e-6
        aflux = compute_actinic_flux(y_cgs_cur)

    C_t = torch.from_numpy(np.maximum(C, 1e-50)).double()
    rate, rc_ddC, _ = kinet.forward(T_t, P_t, C_t)
    chem_tend = torch.matmul(stoich, rate.unsqueeze(-1)).squeeze(-1).numpy()

    photo_tend, photo_jac_diag = compute_photo_tend(aflux, C)

    diff_tend = dA_np[:, None] * C
    diff_tend[:-1] += dB_np[:-1, None] * C[1:]
    diff_tend[1:] += dC_np[1:, None] * C[:-1]
    rhs = chem_tend + photo_tend + diff_tend

    cvol = torch.ones(nz, dtype=torch.float64)
    jac_rxn = kinet.jacobian(T_t, C_t, cvol, rate, rc_ddC)
    chem_jac = torch.matmul(stoich, jac_rxn).numpy()
    np.nan_to_num(chem_jac, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    eye_np = np.eye(ns)
    D_blocks = eye_np[None, :, :] / dt - chem_jac - \
        dA_np[:, None, None] * eye_np[None, :, :]
    for j in range(nz):
        D_blocks[j] -= np.diag(photo_jac_diag[j])

    data = base_data.copy()
    data[diag_slot] = D_blocks
    W = bsr_matrix((data, indices_bsr.copy(), indptr_bsr.copy()),
                   shape=(N_full, N_full))

    delta = spsolve(W.tocsc(), rhs.ravel()).reshape(nz, ns)
    np.nan_to_num(delta, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.clip(C + delta, 1e-50, n_molm3[:, None])
    t_phys += dt

    if (step + 1) % 200 == 0:
        print(f"  step {step+1}/{max_steps}, dt={dt:.2e}, t={t_phys:.2e}")

elapsed = time.time() - t0
ymix = C / n_molm3[:, None]

out_path = sys.argv[1]
with open(out_path, "wb") as f:
    pickle.dump({"ymix": ymix, "P_cgs": P_cgs, "T_K": T_K,
                 "species": species, "nsteps": max_steps,
                 "elapsed": elapsed, "rejects": 0,
                 "t_phys": t_phys}, f)
print(f"HD189 photo+chem: {max_steps} steps, t={t_phys:.2e}, {elapsed:.1f}s")
'''
    out_pkl = os.path.join(SCRIPT_DIR, "_hd189_kt.pkl")
    script_path = os.path.join(SCRIPT_DIR, "_run_hd189.py")
    with open(script_path, "w") as f:
        f.write(script)

    result = subprocess.run(
        [sys.executable, script_path, out_pkl],
        capture_output=True, text=True, cwd=SCRIPT_DIR, timeout=1800,
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
    print("  Running kintera HD189 solver...")
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
    subtitle = f"kintera: {nsteps} steps ({kt_data.get('rejects',0)} rejected), {elapsed:.0f}s"
    fig.suptitle(f"HD 189733b: Kintera-Native vs VULCAN\n({subtitle})",
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
