"""Dump kintera's photolysis rate (k, cm⁻³/s) per altitude for major sources
at KB fort.7 SS state. Use to spot unphysically high rates (no aerosol
attenuation → low-altitude UV penetration too strong)."""
from __future__ import annotations
import sys
sys.path.insert(0, '/home/sam2/dev/kintera/diagnostics')
# Reuse setup code from moses00_perturb.py
exec(open('/home/sam2/dev/kintera/diagnostics/moses00_perturb.py').read().split('def main()')[0])

import torch
from kintera.kinetics_base.titan.source_integration import _photo_rate_profile
from kintera.kinetics_base.titan.models import KBTitanState

ts, filtered, species = build_state_and_sources()
ref = kt.parse_kinetics_base_atmosphere(str(REF_PATH))
ts.state.concentration = inject_state(ts, ref)
ts.concentration = ts.state.concentration

species_index = {s: i for i, s in enumerate(species)}
dz = ts.state.dx1f.view(1, -1)
src_state = KBTitanState(
    species=ts.species, fixed_species=ts.fixed_species,
    varying_species=ts.varying_species, conversion=ts.conversion,
    concentration=ts.concentration, density=ts.density,
    kzz=ts.kzz, state=ts.state,
)

photo_terms = [t for t in filtered if t.kind == "pun_photo_rate_reaction"]
print(f"Total photo terms: {len(photo_terms)}")

altitudes = ts.state.x1v / 1.0e5  # km
LAYERS = [10, 20, 30, 40, 50, 60, 70, 80, 85]

# For each photo term, compute rate constant k(z), and total reaction rate
# k × [parent] at each altitude.
print()
print(f"{'Reaction':<35} | " + " | ".join(f"L{L:<2d}(z={altitudes[L].item():>4.0f})" for L in LAYERS))
print(f"{'':<35} | " + " | ".join(f"{'k (s⁻¹)':<12}" for L in LAYERS))
print("-" * (35 + 14 * len(LAYERS)))

# Sort by impact (max rate × concentration at any layer)
ranked = []
for t in photo_terms:
    parent = t.reactants[0]
    if parent not in species_index:
        continue
    pi = species_index[parent]
    rp = _photo_rate_profile(t, src_state, ts.state.concentration, species_index, pi, dz)
    if rp is None:
        continue
    parent_conc = ts.state.concentration[0, :, pi]
    rate = rp[0] * parent_conc  # photon-induced reaction rate
    max_rate = rate.max().item()
    ranked.append((max_rate, t, rp[0], rate, parent))

ranked.sort(key=lambda x: -x[0])
print(f"\nTop 15 photo reactions by max(k × [parent]):\n")
for max_r, t, rp, rate, parent in ranked[:15]:
    lhs = " + ".join(t.reactants)
    rhs = " + ".join(t.products)
    label = f"{lhs} -> {rhs}"[:35]
    row = " | ".join(f"{rp[L].item():.2e}" for L in LAYERS)
    print(f"{label:<35} | {row}")

# Also dump total reaction rate (k × [parent])
print(f"\nSame reactions, rate = k × [parent] (cm⁻³/s):\n")
for max_r, t, rp, rate, parent in ranked[:15]:
    lhs = " + ".join(t.reactants)
    rhs = " + ".join(t.products)
    label = f"{lhs} -> {rhs}"[:35]
    row = " | ".join(f"{rate[L].item():.2e}" for L in LAYERS)
    print(f"{label:<35} | {row}")
