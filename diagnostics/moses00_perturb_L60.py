"""Focused perturbation study on L60+ remaining gap (after moses00-validated
milestone). At kintera SS, low-altitude species match KB within 10% but
upper-atmosphere (L60+) species drift (C2H4=0.08, HC3N=0.27, C2=0.04 of KB).

Hypothesis: transport-form discrepancy (kintera concentration-form vs KB
mixing-ratio form, see project-kintera-transport-form). Confirm via the
subtraction matrix at multiple dt scales:
  - small dt (1e-2 s): captures chemistry + RT instantaneous tendencies
  - large dt (1e+5 s): captures transport divergence integrated
  - very large dt (1e+7 s): SS-ish, captures all processes including
    accumulated transport effect

Compare A−B (transport contribution) at each scale for upper-atmosphere
species. If A−B at large dt grows ≫ small dt, transport is the gap.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, '/home/sam2/dev/kintera/diagnostics')
exec(open('/home/sam2/dev/kintera/diagnostics/moses00_perturb.py').read().split('def main()')[0])

import torch
import numpy as np

DTS = [1.0e-2, 1.0e+2, 1.0e+4, 1.0e+5, 1.0e+6]
SPECIES_TO_REPORT = ["H", "CH3", "C2H4", "C2H6", "C2H2", "HCN", "HC3N",
                     "C2H", "C2", "CN", "C2N2", "C"]
LAYERS = [20, 30, 40, 50, 60, 65, 70, 75, 80]

ts, filtered, species = build_state_and_sources()
species_index = {s: i for i, s in enumerate(species)}
ref = kt.parse_kinetics_base_atmosphere(str(REF_PATH))

# Build atm sources once
atm_with = kt.build_kinetics_base_titan_source_terms(
    str(PUN_PATH),
    boundary_path=str(BC_PATH),
    run_input_path=str(RUN_INPUT),
    photo_catalog_path=str(CHENG_DIR / "Cheng_catalog_v4.dat"),
    cross_dir=str(CHENG_DIR / "Cheng_cross"),
    flux_path=str(EX_DIR / "flux2atmos.inp"),
)
species_set = set(species)
filtered = [t for t in atm_with
            if all(r in species_set for r in (t.reactants or []) + (t.products or []))]
for t in filtered:
    t.parameters["freeze_actinic_flux"] = False

altitudes = ts.state.x1v.numpy() / 1.0e5

print(f"{'Species':<8} {'L':<4} {'z (km)':<8} ", end="")
for dt_v in DTS:
    print(f"{'A-B@dt='+str(dt_v):<14}", end="")
print()
print(f"  (A−B is transport contribution: kintera mr_diffusion vs KB conc form)")
print()

for sp in SPECIES_TO_REPORT:
    if sp not in species_index:
        continue
    i = species_index[sp]
    for L in LAYERS:
        print(f"{sp:<8} L{L:<3d} {altitudes[L]:<8.0f} ", end="")
        for dt_v in DTS:
            # Inject fresh state
            ts.state.concentration = inject_state(ts, ref)
            ts.concentration = ts.state.concentration
            # Experiment A: full
            atm_full = kt.build_kinetics_base_titan_atm2d_source_terms(ts, filtered)
            d_a = take_step(ts, atm_full, dt_v, use_kzz=True)[L, i].item()
            # Re-inject
            ts.state.concentration = inject_state(ts, ref)
            ts.concentration = ts.state.concentration
            d_b = take_step(ts, atm_full, dt_v, use_kzz=False)[L, i].item()
            transport = (d_a - d_b) / dt_v
            print(f"{transport:<+14.2e}", end="")
        print()
    print()
