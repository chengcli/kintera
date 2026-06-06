#!/usr/bin/env python3.10
"""Perturbation / physics-isolation harness for moses00 at a fixed (KB SS) state.

Injects KB fort.50 (KB's converged SS) as the initial state and computes the
per-physics-process tendency for target species/layers, isolating:

  net        = chem_full + transport          (kintera full pipeline tendency)
  chem_full  = all chemical source terms
  chem_therm = thermal reactions only (photo OFF == actinic flux = 0)
  chem_photo = chem_full - chem_therm          (photochemistry contribution)
  transport  = mr_hybrid eddy + Cheng molecular diffusion divergence
  trans_eddy / trans_mol  (eddy vs molecular split)

At KB's SS, KB's net == 0 per species, so kintera's net is the pipeline
disagreement. The split says which process owns it; chem_photo says whether
the chemistry piece is photochemistry or thermal.

Usage:
  python3.10 diagnostics/perturbation_isolate.py [SP1 SP2 ...]
env: KINTERA_DISABLE_CHEMB_OVERRIDES=1 KINTERA_TITAN_NETWORK_MODE=full
     KINTERA_TITAN_PHOTO_ALLOW_RADICALS=1  (the validated baseline)
"""
import os, sys
os.environ.setdefault("KINTERA_DISABLE_CHEMB_OVERRIDES", "1")
os.environ.setdefault("KINTERA_TITAN_NETWORK_MODE", "full")
os.environ.setdefault("KINTERA_TITAN_PHOTO_ALLOW_RADICALS", "1")
from pathlib import Path
import torch, numpy as np
torch.set_default_dtype(torch.float64)
import kintera as kt
from kintera.atm2d.transport import build_transport_matrix
from kintera.kinetics_base.titan.transport_diffusion import (
    kinetics_base_titan_cheng_diffusion, kinetics_base_titan_species_masses)

EX = Path("diagnostics/KINETICS-base-compare/examples/titan_moses00")
CHENG = Path("diagnostics/KINETICS-base-compare/examples/titan")
TARGETS = sys.argv[1:] or ["HC3N", "C6H6", "C4H2", "C2H2", "C3H3"]
LEVELS = [50, 60, 65, 70, 75, 80]

pun = kt.parse_kinetics_base_pun(str(EX/"kindata"/"kindata.titan.moses00.pun"))
floating = [s.name for s in pun.species if s.id in [1]+list(range(3,46))+list(range(68,84))]
fixed = [s.name for s in pun.species if s.id in [67,2]+list(range(46,67))+list(range(84,87))+[87]]
species = floating + fixed
raw = kt.parse_kinetics_base_atmosphere(str(EX/"atm"/"atm.titan.moses00.kt.inp"))
missing = [s for s in species if s not in raw.species_profiles]
zero = [0.0]*len(raw.altitude)
class Atm:
    altitude=list(raw.altitude); density=list(raw.density); temperature=list(raw.temperature)
    pressure=list(raw.pressure); eddy_diffusion=list(raw.eddy_diffusion)
    species_profiles=dict(raw.species_profiles)
    for s in missing: species_profiles[s]=list(raw.density) if s in ("N2","M") else list(zero)
    if "JDUST" in species_profiles: species_profiles["JDUST"]=list(zero)
ts = kt.build_kinetics_base_titan_state(
    Atm(), species=species, fixed_species=fixed,
    boundary_path=str(EX/"boundary"/"boundary.moses00.inp"),
    pun_path=str(EX/"kindata"/"kindata.titan.moses00.pun"))
sterms = kt.build_kinetics_base_titan_source_terms(
    str(EX/"kindata"/"kindata.titan.moses00.pun"),
    boundary_path=str(EX/"boundary"/"boundary.moses00.inp"),
    run_input_path=str(EX/"case"/"kinetics.inp"),
    photo_catalog_path=str(CHENG/"Cheng_catalog_v4.dat"),
    cross_dir=str(CHENG/"Cheng_cross"), flux_path=str(EX/"flux2atmos.inp"))
for t in sterms:
    if "freeze_actinic_flux" in getattr(t, "parameters", {}):
        t.parameters["freeze_actinic_flux"] = False
sp_set = set(species)
filtered = [t for t in sterms if all(r in sp_set for r in (t.reactants or [])+(t.products or []))]
thermal_only = [t for t in filtered if t.kind != "pun_photo_rate_reaction"]

# inject fort.50 (KB SS)
ref = kt.parse_kinetics_base_atmosphere(str(EX/"case"/"fort.50.kt.v2"))
c = ts.state.concentration.clone(); density = ts.density[0]
for name in ts.species:
    i = ts.species.index(name)
    if name in {"M", "JDUST"} or name not in ref.species_profiles: continue
    prof = torch.as_tensor(np.array(ref.species_profiles[name]), dtype=c.dtype)
    if prof.numel() == c.shape[1]: c[0, :, i] = prof*density
ts.state.concentration = c

# --- chemistry tendencies (transport OFF) ---
chem_full = kt.build_source_linearization(ts.state, kt.build_kinetics_base_titan_atm2d_source_terms(ts, filtered)).tendency
chem_therm = kt.build_source_linearization(ts.state, kt.build_kinetics_base_titan_atm2d_source_terms(ts, thermal_only)).tendency
chem_photo = chem_full - chem_therm

# --- transport tendencies (chemistry OFF) ---
masses = kinetics_base_titan_species_masses(list(ts.species), {sp.name: sp for sp in pun.species})
bdiff = kinetics_base_titan_cheng_diffusion(ts.state, masses, density=ts.density)
trans_full = build_transport_matrix(ts.state, ts.kzz, binary_diffusion=bdiff, molecular_weights=masses, density=ts.density, form="mr_hybrid").matvec(c)
trans_eddy = build_transport_matrix(ts.state, ts.kzz, density=ts.density, form="mr_diffusion").matvec(c)
trans_mol = trans_full - trans_eddy
net = chem_full + trans_full

alt = np.array(raw.altitude)
print(f"Physics isolation @ KB SS (fort.50). Targets: {TARGETS}")
print("All tendencies in cm-3 s-1. At KB SS, KB net=0; kintera net = pipeline disagreement.")
for s in TARGETS:
    if s not in species:
        print(f"  {s}: not in species"); continue
    i = species.index(s)
    print(f"\n{s}:  c, then chem(full/thermal/photo) | transport(full/eddy/mol) | NET, net/c")
    print(f"  {'L':>3}{'alt':>6}{'c':>10}{'chem_f':>10}{'chem_th':>10}{'chem_ph':>10}{'tr_f':>10}{'tr_eddy':>10}{'tr_mol':>10}{'NET':>10}{'net/c':>9}")
    for L in LEVELS:
        cc = c[0, L, i].item()
        row = [chem_full, chem_therm, chem_photo, trans_full, trans_eddy, trans_mol, net]
        vals = "".join(f"{a[0,L,i].item():>10.2e}" for a in row)
        print(f"  {L:>3}{alt[L]:>6.0f}{cc:>10.2e}{vals}{net[0,L,i].item()/cc if cc>0 else float('nan'):>9.1e}")
