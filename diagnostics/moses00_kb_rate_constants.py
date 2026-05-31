"""Recover KB's rate constants k_KB by dividing KB's reported rates by KB's
reported concentrations (parsed from MIXING RATIOS section), then compare to
kintera's computed k at the same (T, ρ).

Eliminates concentration dependence so we compare model-level rate constants
directly.
"""
from __future__ import annotations
import sys, re
sys.path.insert(0, '/home/sam2/dev/kintera/diagnostics')
import numpy as np
from pathlib import Path

KB_LOG = Path("/tmp/kb_m00/kinetics.out-revert")
KB_RATES_RAW = np.load("/tmp/kb_moses00_rates.npz")["rates"]  # (521, 91), indexed by KB row 1..91 → 0..90
NLYR = 91
# KB row N (1-indexed in log) → kintera 0-indexed layer (N + 8). NALT1=10 means
# KB simulates atm levels 10..91 (1-indexed). Atm level 10 = kintera 0-indexed L9.
# So the rate at KB row N is at kintera L(N+8).
KB_TO_KT_OFFSET = 8
# Build aligned arrays of shape (521, 91) where index k = kintera 0-indexed L_k.
KB_RATES = np.zeros_like(KB_RATES_RAW)
for kb_row_minus_1 in range(NLYR):
    kt_layer = kb_row_minus_1 + KB_TO_KT_OFFSET + 1  # KB row (1-indexed) is kb_row_minus_1+1
    if 0 <= kt_layer < NLYR:
        KB_RATES[:, kt_layer] = KB_RATES_RAW[:, kb_row_minus_1]

# Parse the LAST MIXING RATIOS section before REACTION RATES at line 119241
# Format: PRESSURE  H  H2  C  CH  (1)CH2  (3)CH2  CH3  CH4  C2  C2H ...
# 10 species per block, multiple blocks for full species list.
with KB_LOG.open() as f:
    lines = f.readlines()

# Find last MIXING RATIOS before REACTION RATES
mr_start = None
for i, ln in enumerate(lines):
    if "REACTION RATES:" in ln:
        break
    if "MIXING RATIOS" in ln and "RATIOS :" in ln:
        mr_start = i

print(f"Last MIXING RATIOS section starts at line {mr_start+1}")

# Parse: blocks of 10 species. Each block: header line with names, then 91 data rows.
# Stop at PRODUCTION AND LOSS RATES (which has its own PRESSURE+species headers).
kb_mr: dict[str, np.ndarray] = {}
i = mr_start + 1
data_re = re.compile(r'^\s*(\d+)\)\s+([-+0-9.eE]+)\s+(.*)$')
while i < len(lines):
    ln = lines[i]
    stop_markers = ("REACTION RATES", "PRODUCTION AND LOSS",
                    "REACTION CONTRIBUTIONS", "FLUX:",
                    "WIND VELOCITY", "DIFFUSION COEFFICIENT",
                    "SCALE HEIGHT")
    if any(m in ln for m in stop_markers):
        break
    # Header line lists species names
    if "PRESSURE" in ln and "RATE(" not in ln and "PRODUCT" not in ln:
        # Extract species names after PRESSURE
        idx = ln.find("PRESSURE")
        names_part = ln[idx + len("PRESSURE"):].strip()
        names = names_part.split()
        if not names:
            i += 1
            continue
        # Skip blank line if any
        i += 1
        while i < len(lines) and not data_re.match(lines[i]):
            i += 1
        # Read up to 91 rows
        block_data = {name: np.zeros(NLYR) for name in names}
        for layer in range(NLYR):
            if i >= len(lines):
                break
            m = data_re.match(lines[i])
            if m is None:
                break
            row_idx = int(m.group(1)) - 1
            rest = m.group(3)
            vals = []
            for tok in rest.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    break
            if len(vals) >= len(names) and 0 <= row_idx < NLYR:
                for j, name in enumerate(names):
                    block_data[name][row_idx] = vals[j]
            i += 1
        for name, arr in block_data.items():
            kb_mr[name] = arr
    else:
        i += 1

print(f"Parsed KB mixing ratios for {len(kb_mr)} species")
print(f"  sample: H_L20={kb_mr.get('H',[0])[20]:.3e}, CH3_L20={kb_mr.get('CH3',[0])[20]:.3e}, "
      f"CH4_L20={kb_mr.get('CH4',[0])[20]:.3e}")

# Convert MR to concentration (KB MR × density), applying the same NALT1=10 offset.
# KB MR table row N (1-indexed) → kintera 0-indexed L(N+8). Re-align the array.
import kintera as kt
atm = kt.parse_kinetics_base_atmosphere(
    "/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare/examples/titan_moses00/atm/atm.titan.moses00.kt.inp"
)
density = np.array(atm.density)
kb_conc: dict[str, np.ndarray] = {}
for sp, mr_raw in kb_mr.items():
    mr_aligned = np.zeros_like(mr_raw)
    for kb_row_minus_1 in range(NLYR):
        kt_layer = kb_row_minus_1 + KB_TO_KT_OFFSET + 1
        if 0 <= kt_layer < NLYR:
            mr_aligned[kt_layer] = mr_raw[kb_row_minus_1]
    kb_conc[sp] = mr_aligned * density

print(f"\nKB output-state concentrations at L20 (z=142km):")
for sp in ["H", "H2", "CH4", "CH3", "C2H2", "C2H4", "C2H6", "M"]:
    if sp in kb_conc:
        print(f"  {sp}: MR={kb_mr[sp][20]:.3e}, conc={kb_conc[sp][20]:.3e}")

# Now load moses00 PUN to know reactant indices per reaction
pun = kt.parse_kinetics_base_pun(
    "/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare/examples/titan_moses00/kindata/kindata.titan.moses00.pun"
)
sp_by_id = {s.id: s.name for s in pun.species}

# For each reaction, compute k_KB = SRATE_KB / (∏ c_KB^coef_i)
NREACT_KB = KB_RATES.shape[0]
k_KB = np.zeros((NREACT_KB, NLYR), dtype=np.float64)
reaction_info = {}
for rxn in pun.reactions:
    rid = rxn.id
    if rid > NREACT_KB:
        continue
    reactants = []
    coefs = []
    for j, part in enumerate(rxn.participants[:rxn.n_reactants]):
        sp_name = sp_by_id.get(part.species_id, None)
        if sp_name is None:
            continue
        reactants.append(sp_name)
        coefs.append(int(part.coefficient) or 1)
    reaction_info[rid] = (reactants, coefs)
    # Compute product of concentrations per layer
    denom = np.ones(NLYR)
    for sp_name, coef in zip(reactants, coefs):
        if sp_name == "M":
            # M = density
            c = density
        elif sp_name in kb_conc:
            c = kb_conc[sp_name]
        else:
            # Species not in KB output (e.g. fixed at zero like H2O)
            denom = None
            break
        denom *= c ** coef
    if denom is None:
        continue
    nz = denom > 1e-100
    k_KB[rid - 1, nz] = KB_RATES[rid - 1, nz] / denom[nz]

# Save
np.savez("/tmp/moses00_kb_k.npz", k_KB=k_KB, kb_conc=np.array([(sp, conc) for sp, conc in kb_conc.items()], dtype=object))

# Print top 15 mismatch reactions with both k values
kt_data = np.load("/tmp/moses00_rate_compare.npz", allow_pickle=True)
kt_rates = kt_data["kt_rates"]
labels = kt_data["reaction_labels"]
altitudes = kt_data["altitudes"]

# Compute kintera's k at the SAME altitudes by dividing kt_rates by injected concentrations
# Injected conc = KB fort.7 state. Already have it... easier: just compute k_kt from formulas.
# For the comparison: compare kt_rates(at fort.7) to KB rates at output, then check k_KB vs k_kt.

# Better: load the kintera per-reaction prod-per-cm3-s from comparison, divide by fort.7 conc.
ref_atm = kt.parse_kinetics_base_atmosphere(
    "/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare/examples/titan_moses00/case/fort.7.kt"
)
fort7_density = np.array(ref_atm.density)
fort7_conc: dict[str, np.ndarray] = {}
for sp in pun.species:
    if sp.name in ref_atm.species_profiles and sp.name not in ("M", "JDUST"):
        fort7_conc[sp.name] = np.array(ref_atm.species_profiles[sp.name]) * fort7_density
fort7_conc["M"] = fort7_density
fort7_conc["JDUST"] = np.zeros(NLYR)

# k_kt = kt_rates / ∏ fort7_conc^coef
k_kt = np.zeros((NREACT_KB, NLYR))
for rid in range(1, NREACT_KB + 1):
    if rid not in reaction_info:
        continue
    reactants, coefs = reaction_info[rid]
    denom = np.ones(NLYR)
    skip = False
    for sp_name, coef in zip(reactants, coefs):
        if sp_name not in fort7_conc:
            skip = True
            break
        denom *= fort7_conc[sp_name] ** coef
    if skip:
        continue
    nz = denom > 1e-100
    k_kt[rid - 1, nz] = kt_rates[rid - 1, nz] / denom[nz]

# Show comparison for key reactions across several altitudes (kintera 0-indexed)
print(f"\n=== Rate-constant comparison (kintera 0-indexed layers) ===")
TEST_LAYERS = [10, 15, 20, 25, 29, 35, 40, 50, 60, 70]
for L in TEST_LAYERS:
    print(f"\n--- L{L} (z={atm.altitude[L]:.0f}km, T={atm.temperature[L]:.1f}K, ρ={density[L]:.2e}) ---")
print(f"{'rxn':<5} {'reaction':<40} {'k_kintera':<13} {'k_KB':<13} {'ratio':<10}")
key_rxns = [(157, "CH3+CH3+M -> C2H6+M"),
            (10, "C2H2 -> C2H + H"),
            (9, "CH4 -> CH+H+H2"),
            (5, "CH4 -> CH3+H"),
            (166, "CH3+C3H5+M -> C4H8+M"),
            (161, "CH3+C2H5+M -> C3H8+M"),
            (147, "(1)CH2+CH4 -> 2CH3"),
            (158, "CH3+C2H3 -> CH4+C2H2"),
            (177, "C2H+CH4 -> C2H2+CH3"),
            (134, "H+C6H3 -> C6H2+H2"),
            (135, "H+C6H2+M -> C6H3+M"),
            ]
for rid, lab in key_rxns:
    if rid - 1 >= NREACT_KB:
        continue
    print(f"\n  {rid:<3} {lab[:55]}")
    print(f"  {'L':<4} {'k_kintera':<13} {'k_KB':<13} {'ratio':<10}")
    for L in TEST_LAYERS:
        kkt = k_kt[rid - 1, L]
        kkb = k_KB[rid - 1, L]
        r = kkt / kkb if kkb > 1e-100 else float('inf')
        print(f"  L{L:<3d} {kkt:<13.2e} {kkb:<13.2e} {r:<10.3f}")
