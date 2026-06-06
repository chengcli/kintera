"""Stage 5.2a: validate the core-engine THERMAL tendency against the
hand-rolled baseline at the moses00 fort.50 reference state.

Builds the moses00 Titan state exactly as `moses00_match.py` does, then compares:
  - baseline: sum of `linearize` over the hand-rolled THERMAL source terms.
  - core:     tendency from core Kinetics rate constants (Arrhenius + KBFalloff)
              + ChembOverrideLayer, frozen-k mass-action, on the same state.

Photolysis is excluded here (its activation policy is validated separately).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import kintera as kt

torch.set_default_dtype(torch.float64)

sys.path.insert(0, str(Path(__file__).parent))
import moses00_match as mm  # noqa: E402

from kintera.kinetics_base.titan.core_chemb import build_chemb_override_layer  # noqa: E402


def build_state():
    pun = kt.parse_kinetics_base_pun(str(mm.PUN_PATH))
    floating = mm.species_names(pun, mm.FLOATING_IDS)
    fixed = mm.species_names(pun, mm.FIXED_IDS)
    species = floating + fixed
    raw_atm = kt.parse_kinetics_base_atmosphere(str(mm.ATM_PATH))
    missing = [s for s in species if s not in raw_atm.species_profiles]
    zero_profile = [0.0] * len(raw_atm.altitude)
    density_like = ("N2", "M")
    force_zero = ("JDUST",)

    class _AtmShim:
        altitude = list(raw_atm.altitude)
        density = list(raw_atm.density)
        temperature = list(raw_atm.temperature)
        pressure = list(raw_atm.pressure)
        eddy_diffusion = list(raw_atm.eddy_diffusion)
        species_profiles = dict(raw_atm.species_profiles)
        for s in missing:
            species_profiles[s] = list(raw_atm.density) if s in density_like else list(zero_profile)
        for s in force_zero:
            if s in species_profiles:
                species_profiles[s] = list(zero_profile)

    ts = kt.build_kinetics_base_titan_state(
        _AtmShim(), species=species, fixed_species=fixed,
        boundary_path=str(mm.BC_PATH), pun_path=str(mm.PUN_PATH),
    )
    sterms = kt.build_kinetics_base_titan_source_terms(
        str(mm.PUN_PATH), boundary_path=str(mm.BC_PATH), run_input_path=str(mm.RUN_INPUT),
        photo_catalog_path=str(mm.PHOTO_CATALOG), cross_dir=str(mm.CROSS_DIR),
        flux_path=str(mm.FLUX_PATH),
    )
    species_set = set(species)
    filtered = [t for t in sterms
                if all(r in species_set for r in (t.reactants or []) + (t.products or []))]
    # chemistry only: drop boundary-condition terms (upper/lower_boundary_*)
    thermal_terms = [t for t in filtered if t.kind == "pun_thermal_reaction"]
    ref = kt.parse_kinetics_base_atmosphere(str(mm.REF_PATH))
    ts.state.concentration = mm.inject_reference_state(ts, ref)
    ts.concentration = ts.state.concentration
    atm_sources = kt.build_kinetics_base_titan_atm2d_source_terms(ts, thermal_terms)
    return ts, atm_sources


def baseline_thermal_tendency(ts, atm_sources):
    """Sum linearize() over the (thermal-only) source terms."""
    lin = kt.build_source_linearization(ts.state, atm_sources)
    return lin.tendency, len(atm_sources)


def core_species_order():
    pun = kt.parse_kinetics_base_pun(str(mm.PUN_PATH))
    return [s.name for s in sorted(pun.species, key=lambda s: s.id)]


def core_thermal_tendency(ts):
    """Core-engine thermal tendency on ts.state, returned in ts.species order."""
    op = kt.KineticsOptions.from_kinetics_base_pun(str(mm.PUN_PATH))
    core_sp = core_species_order()
    nsp = len(core_sp)
    reactions = op.reactions()  # arrhenius(291) + kb_falloff(50), core column order
    nrxn = len(reactions)

    # permutation: ts.species index -> core species index
    core_idx = {name: i for i, name in enumerate(core_sp)}
    ts_to_core = [core_idx[name] for name in ts.species]

    # embed ts concentration into core species order
    conc_ts = ts.state.concentration  # (ncol, nlyr, nsp_ts)
    ncol, nlyr, _ = conc_ts.shape
    conc = torch.zeros((ncol, nlyr, nsp), dtype=conc_ts.dtype)
    conc[..., ts_to_core] = conc_ts
    T = ts.state.temperature
    P = ts.state.pressure
    density = ts.density  # (ncol, nlyr) total number density

    # rate constants from core modules (reactions() order)
    k_arr = kt.Arrhenius(op.arrhenius()).forward(T, P, conc, {})
    k_kbf = kt.KBFalloff(op.kb_falloff()).forward(T, P, conc, {"number_density": density})
    k = torch.cat([k_arr, k_kbf], dim=-1)  # (ncol, nlyr, nrxn)

    # chemb override (frozen-density k) on the matched columns
    layer = build_chemb_override_layer(reactions)
    k = layer.apply(k, T, density)

    # reactant/net stoichiometry on core species order
    react_st = torch.zeros((nsp, nrxn), dtype=conc.dtype)
    net_st = torch.zeros((nsp, nrxn), dtype=conc.dtype)
    for j, r in enumerate(reactions):
        for name, c in r.reactants().items():
            i = core_idx[name]
            react_st[i, j] += c
            net_st[i, j] -= c
        for name, c in r.products().items():
            i = core_idx[name]
            net_st[i, j] += c

    # frozen-k mass action: rate_f = k * prod(C^react_stoich)
    csafe = conc.clamp_min(1e-300)
    prod = torch.exp(torch.einsum("ij,blj->bli", react_st.T, torch.log(csafe)))  # (b,l,nrxn)
    # zero-out reactions whose reactant conc is genuinely zero
    has_zero = torch.einsum("ij,blj->bli", react_st.T, (conc <= 0).double()) > 0
    prod = torch.where(has_zero, torch.zeros_like(prod), prod)
    rate_f = k * prod
    tendency = torch.einsum("ij,blj->bli", net_st, rate_f)  # (b,l,nsp) core order

    # map back to ts.species order
    tend_ts = torch.zeros_like(conc_ts)
    tend_ts[...] = tendency[..., ts_to_core]
    return tend_ts, nrxn


def main():
    print("[setup] building moses00 state + baseline source terms")
    ts, atm_sources = build_state()
    base, n_thermal = baseline_thermal_tendency(ts, atm_sources)
    print(f"  baseline thermal source terms: {n_thermal}")
    core, nrxn = core_thermal_tendency(ts)
    print(f"  core thermal reactions: {nrxn}")

    c = ts.state.concentration
    # compare where the baseline tendency is non-trivial
    mask = base.abs() > 0
    denom = torch.maximum(base.abs(), torch.full_like(base, 1e-300))
    rel = (core - base).abs() / denom
    print(f"  cells with nonzero baseline tendency: {int(mask.sum())}")
    print(f"  max rel diff (nonzero cells):    {rel[mask].max().item():.3e}")
    print(f"  median rel diff (nonzero cells): {rel[mask].median().item():.3e}")
    frac = (rel[mask] < 1e-6).double().mean().item()
    print(f"  fraction within 1e-6: {frac:.4f}")
    # worst offenders
    flat = rel.flatten()
    worst = torch.topk(flat, 5).indices
    sp = ts.species
    for w in worst.tolist():
        b, l, s = np.unravel_index(w, rel.shape)
        print(f"    worst: {sp[s]} L{l}: core={core[b,l,s]:.3e} base={base[b,l,s]:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
