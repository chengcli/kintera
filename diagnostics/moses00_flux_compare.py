"""Compare kintera's per-face vertical flux to KB's at fort.7 SS for HCN
(the sign-flip species) and other stable hydrocarbons. Localizes the
MR-form discretization mismatch.

KB log line 114842+ has "FLUX:" section with values at face N.5 (between
cell N and N+1, 1-indexed). For moses00, face label N.5 = kintera face
between L(N+8) and L(N+9).

For kintera's mr_diffusion form, flux at face L+0.5 should be:
    flux = -K(face) × n_tot(face) × ∂(c/n)/∂z

We compare per face to find where kintera diverges from KB."""
from __future__ import annotations
import sys, re
sys.path.insert(0, '/home/sam2/dev/kintera/diagnostics')
exec(open('/home/sam2/dev/kintera/diagnostics/moses00_perturb.py').read().split('def main()')[0])

import os
os.environ.setdefault('KINTERA_DISABLE_CHEMB_OVERRIDES', '1')

import numpy as np
import torch

KB_LOG = "/tmp/kb_m00/kinetics.out-revert"
NALT_OFFSET = 8

# Parse KB FLUX section
with open(KB_LOG) as f:
    lines = f.readlines()

flux_start = None
for i, ln in enumerate(lines):
    if "FLUX:" in ln and not "EFFECTIVE" in ln:
        flux_start = i
        break
print(f"FLUX section at line {flux_start+1}")

kb_flux: dict[str, np.ndarray] = {}
i = flux_start + 1
face_re = re.compile(r'^\s*(\d+)\.5\)\s+([-+0-9.eE]+)\s+(.*)$')
while i < len(lines):
    ln = lines[i]
    if any(s in ln for s in ("EFFECTIVE WIND VELOCITY", "PRODUCTION AND LOSS",
                              "REACTION RATES", "GENERALIZED")):
        break
    if "PRESSURE" in ln and "RATE(" not in ln and "PRODUCT" not in ln:
        names = ln[ln.find("PRESSURE") + len("PRESSURE"):].split()
        if not names:
            i += 1; continue
        for name in names:
            kb_flux.setdefault(name, np.zeros(91))
        i += 1
        # Read face rows
        while i < len(lines):
            m = face_re.match(lines[i])
            if not m:
                if "PRESSURE" in lines[i]:
                    break
                i += 1; continue
            face_idx = int(m.group(1))  # 1-indexed face N (means face N.5)
            kt_face_idx = face_idx + NALT_OFFSET  # face between kintera L(N+8) and L(N+9)
            rest = m.group(3)
            vals = []
            for tok in rest.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    break
            if len(vals) >= len(names) and 0 <= kt_face_idx < 91:
                for j, name in enumerate(names):
                    kb_flux[name][kt_face_idx] = vals[j]
            i += 1
    else:
        i += 1

print(f"Parsed KB flux for {len(kb_flux)} species")

# Build kintera state
ts, filtered, species = build_state_and_sources()
ref = kt.parse_kinetics_base_atmosphere(str(REF_PATH))
ts.state.concentration = inject_state(ts, ref)
ts.concentration = ts.state.concentration
species_index = {s: i for i, s in enumerate(species)}
altitude = ts.state.x1v.numpy() / 1.0e5
density = ts.density[0].numpy()
kzz = ts.kzz[0].numpy()  # eddy diffusion at cell centers

# Build kintera face flux for each species using mr_diffusion convention:
#   flux(L+0.5) = -K(L+0.5) × n_tot(L+0.5) × [c(L+1)/n_tot(L+1) - c(L)/n_tot(L)] / dz(L+0.5)
# where K, n_tot at face are computed from cell-center values (need to know how kintera does this)
conc = ts.state.concentration[0].numpy()  # (NLYR, nspecies)
nlyr = 91
# Face spacing: dz between cell centers
dz = ts.state.dx1f.numpy()  # (NLYR,) cell widths in cm
# Face altitude midpoint
alt_face = 0.5 * (altitude[:-1] + altitude[1:])  # NLYR-1 faces

# Kintera's face values: face L between cell L and L+1.
# Use arithmetic mean for K and n_tot at face.
K_face = 0.5 * (kzz[:-1] + kzz[1:])
n_face = 0.5 * (density[:-1] + density[1:])
dz_face = 0.5 * (dz[:-1] + dz[1:])  # face spacing

# MR gradient
mr = np.where(density[:, None] > 0, conc / density[:, None], 0.0)  # (NLYR, nspecies)
grad_mr_face = (mr[1:, :] - mr[:-1, :]) / dz_face[:, None]
# Flux at face: cm⁻²/s = K (cm²/s) × n (cm⁻³) × dimensionless / cm  → wait units check
# flux [cm⁻²/s] = K [cm²/s] × n [cm⁻³] × (∆MR / ∆z [1/cm])
#               = K × n × ∆MR / ∆z. Units: cm²/s × cm⁻³ × 1/cm = cm⁻⁴/s. WRONG
# Actually mr_diffusion flux convention: F = -K × n_tot × ∂χ/∂z where χ = c/n_tot
# F has units cm⁻²/s ✓ when K has cm²/s, n cm⁻³, ∂χ/∂z 1/cm
# K × n × 1/cm = cm²/s × cm⁻³ × cm⁻¹ = cm⁻²/s ✓

kt_flux = -K_face[:, None] * n_face[:, None] * grad_mr_face  # (NLYR-1, nspecies)

# Compare at key faces for stable species
REPORT_SPECIES = ["HCN", "C2H6", "C2H4", "C2H2", "HC3N", "CH4", "H2", "C2N2"]
REPORT_FACES = [20, 25, 29, 35, 40, 50, 60, 70]  # kintera 0-indexed face L (between L and L+1)

print()
print(f"{'Species':<8} {'face L':<7} {'z (km)':<8} {'kt flux':<12} {'KB flux':<12} {'ratio':<9} {'sign':<8}")
for sp in REPORT_SPECIES:
    if sp not in species_index or sp not in kb_flux:
        print(f"  {sp}: not found in both")
        continue
    i = species_index[sp]
    for face in REPORT_FACES:
        if face >= nlyr - 1:
            continue
        kt_f = kt_flux[face, i]
        kb_f = kb_flux[sp][face]  # kb_flux indexed by kintera face L
        ratio = kt_f / kb_f if abs(kb_f) > 1e-30 else float('nan')
        sign_match = "match" if (kt_f * kb_f >= 0) else "FLIP"
        print(f"{sp:<8} L{face:<5d} {alt_face[face]:<8.0f} "
              f"{kt_f:<+12.3e} {kb_f:<+12.3e} {ratio:<+9.3f} {sign_match:<8}")
    print()

np.savez("/tmp/moses00_flux_compare.npz",
         species=np.array(species, dtype=object),
         altitude_face=alt_face,
         kt_flux=kt_flux,
         kb_flux=np.array([(sp, arr) for sp, arr in kb_flux.items()], dtype=object))
print("Saved /tmp/moses00_flux_compare.npz")
