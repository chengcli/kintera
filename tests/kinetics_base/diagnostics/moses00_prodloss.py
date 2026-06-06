"""Frozen-state per-reaction prod/loss probe at KB's converged SS.

Step 1 of kt-kb-matching protocol: inject KB's fort.7 SS, compute kintera's
chemistry tendency at that state, and break down the dominant production
and loss reactions for a target species at a target layer.

Discrepancies at SS that are >> transport tendency imply kintera's rate
constants disagree with what KB used to reach that SS.

Usage:
  KT_SPECIES=C3H8 KT_LAYERS="20,25,30,40,60" python diagnostics/moses00_prodloss.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

import kintera as kt

torch.set_default_dtype(torch.float64)

EX_DIR = Path(
    "/home/sam2/dev/kintera/tests/kinetics_base/diagnostics/KINETICS-base-compare/examples/titan_moses00"
)
PUN_PATH = EX_DIR / "kindata" / "kindata.titan.moses00.pun"
ATM_PATH = EX_DIR / "atm" / "atm.titan.moses00.kt.inp"
BC_PATH = EX_DIR / "boundary" / "boundary.moses00.inp"
RUN_INPUT = EX_DIR / "case" / "kinetics.inp"
REF_PATH = EX_DIR / "case" / "fort.7.kt"  # KB SS (not fort.50)

CHENG_DIR = Path(
    "/home/sam2/dev/kintera/tests/kinetics_base/diagnostics/KINETICS-base-compare/examples/titan"
)
PHOTO_CATALOG = CHENG_DIR / "Cheng_catalog_v4.dat"
CROSS_DIR = CHENG_DIR / "Cheng_cross"
FLUX_PATH = EX_DIR / "flux2atmos.inp"

FLOATING_IDS = [1] + list(range(3, 46)) + list(range(68, 84))
# M is PUN id=87 in moses00 (id=150 in moses05.C2)
FIXED_IDS = [67, 2] + list(range(46, 67)) + list(range(84, 87)) + [87]
# moses00 fort.7 stores ALL species as mixing ratios. M and JDUST appear with
# placeholder values (M=1.0, JDUST=0.0); skip them in inject and keep build-
# time values (M=density, JDUST=density via the atm shim density_like list).
CONCENTRATION_SPECIES = set()
SKIP_INJECT = {"M", "JDUST"}  # fort.7 has JDUST as aerosol MR; force to 0 here since moses00 has no aerosol chem

TARGET_SPECIES = os.environ.get("KT_SPECIES", "C3H8").split(",")
LAYERS = [int(x) for x in os.environ.get("KT_LAYERS", "20,30,40,60").split(",")]
TOP_N = int(os.environ.get("KT_TOPN", "8"))


def species_names(pun, ids):
    sp_by_id = {s.id: s.name for s in pun.species}
    return [sp_by_id[i] for i in ids if i in sp_by_id]


def inject_state(ts, ref):
    c = ts.concentration.clone()
    density = ts.density[0]
    for name in ts.species:
        i = ts.species.index(name)
        if name in SKIP_INJECT:
            continue
        if name not in ref.species_profiles:
            continue
        prof = torch.as_tensor(np.array(ref.species_profiles[name]), dtype=c.dtype)
        if prof.numel() != c.shape[1]:
            continue
        if name in CONCENTRATION_SPECIES:
            c[0, :, i] = prof
        else:
            c[0, :, i] = prof * density
    return c


def label_term(t):
    if t.kind == "pun_photo_rate_reaction":
        prefix = "hv"
    elif t.kind == "pun_thermal_reaction":
        prefix = "T"
    elif t.kind == "pun_electron_impact_reaction":
        prefix = "e-"
    else:
        prefix = t.kind[:4]
    rhs = " + ".join(t.products) if t.products else "(none)"
    lhs = " + ".join(t.reactants) if t.reactants else "(hv)"
    return f"[{prefix}] {lhs} -> {rhs}"


def reaction_rate_constant(term, T, density):
    """Compute kintera's rate constant for `term` at scalar T and density.

    Returns (k_value, source_label) where source_label indicates which
    code path (chemb override / pun_catalog / photo / other) produced k.
    """
    from kintera.kinetics_base.titan.physics import _pun_rate_constant
    from kintera.kinetics_base.titan.chemb_overrides import (
        has_titan_chemb_override,
        has_titan_chemb_override_by_signature,
        titan_chemb_rate_constant,
        titan_chemb_rate_constant_by_signature,
    )
    if term.kind != "pun_thermal_reaction":
        return None, term.kind
    T_t = torch.as_tensor(float(T), dtype=torch.float64)
    d_t = torch.as_tensor(float(density), dtype=torch.float64)
    rxn_id = getattr(term, "reaction_id", None)
    if rxn_id is not None and has_titan_chemb_override(rxn_id):
        k = titan_chemb_rate_constant(rxn_id, T_t, d_t)
        if k is not None:
            return float(k), f"chemb(id={rxn_id})"
    if has_titan_chemb_override_by_signature(term.reactants, term.products):
        k = titan_chemb_rate_constant_by_signature(term.reactants, term.products, T_t, d_t)
        if k is not None:
            return float(k), "chemb(sig)"
    k = _pun_rate_constant(term.parameters, T_t.unsqueeze(0), d_t.unsqueeze(0))
    return float(k[0]), "pun_cat"


def format_term_params(term):
    """Compact summary of rate parameters for `term`."""
    p = term.parameters
    parts = []
    for k in ("A", "b", "C", "D", "E", "F"):
        v = p.get(k, 0.0)
        if v != 0.0:
            parts.append(f"{k}={v:.3g}")
    return ",".join(parts) if parts else "(none)"


def main() -> int:
    print("[setup] loading moses00 PUN")
    pun = kt.parse_kinetics_base_pun(str(PUN_PATH))
    floating = species_names(pun, FLOATING_IDS)
    fixed = species_names(pun, FIXED_IDS)
    species = floating + fixed

    raw_atm = kt.parse_kinetics_base_atmosphere(str(ATM_PATH))
    missing = [s for s in species if s not in raw_atm.species_profiles]
    zero_profile = [0.0] * len(raw_atm.altitude)
    density_like = ("N2", "M")
    # JDUST: moses00 atm ships nonzero JDUST that with kintera's σ=1e-8 cm²
    # would give column τ~2800 at L60, killing photolysis entirely. moses00
    # has no aerosol chemistry — force JDUST=0 in the shim so the diagnostic
    # sees the same photolysis rates as the SS run.
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
    atm = _AtmShim()
    ts = kt.build_kinetics_base_titan_state(
        atm, species=species, fixed_species=fixed,
        boundary_path=str(BC_PATH), pun_path=str(PUN_PATH),
    )

    print(f"  state: {ts.state.nspecies} species x {ts.state.nlyr} levels")

    sterms = kt.build_kinetics_base_titan_source_terms(
        str(PUN_PATH),
        boundary_path=str(BC_PATH),
        run_input_path=str(RUN_INPUT),
        photo_catalog_path=str(PHOTO_CATALOG),
        cross_dir=str(CROSS_DIR),
        flux_path=str(FLUX_PATH),
    )
    species_set = set(species)
    filtered = [t for t in sterms
                if all(r in species_set for r in (t.reactants or []) + (t.products or []))]
    for t in filtered:
        t.parameters["freeze_actinic_flux"] = False
    print(f"  filtered source terms: {len(filtered)}")

    ref = kt.parse_kinetics_base_atmosphere(str(REF_PATH))
    print(f"  reference KB SS: {len(ref.species_profiles)} species")
    ts.state.concentration = inject_state(ts, ref)
    ts.concentration = ts.state.concentration

    # Per-term individual sources for 1:1 mapping
    individual_sources = []
    matched_terms = []
    for t in filtered:
        atm_one = kt.build_kinetics_base_titan_atm2d_source_terms(ts, [t])
        for s in atm_one:
            individual_sources.append(s)
            matched_terms.append(t)
    print(f"  individual sources: {len(individual_sources)}")

    alts = ts.state.x1v.numpy() / 1.0e5
    density = ts.density[0].numpy()

    print()
    print(f"=== Frozen prod/loss at KB fort.7 SS, layers {LAYERS} ===")
    for sp_name in TARGET_SPECIES:
        if sp_name not in species:
            print(f"\n[skip] {sp_name} not in subset")
            continue
        sp_i = species.index(sp_name)
        c_vals = ts.state.concentration[0, :, sp_i].numpy()

        print(f"\n-- {sp_name} ----------------------------------------")
        for L in LAYERS:
            c = c_vals[L]
            den = density[L]
            alt = alts[L]
            mr = c / den if den > 0 else 0.0
            print(f"  L{L:2d} (z={alt:6.1f} km, rho={den:.2e}): c={c:.3e} cm-3, MR={mr:.2e}")

            contributions = []
            for src, term in zip(individual_sources, matched_terms):
                try:
                    lin = src.linearize(ts.state)
                except Exception:
                    continue
                tend = lin.tendency
                if tend.shape[-1] != ts.state.nspecies:
                    continue
                v = tend[0, L, sp_i].item()
                if abs(v) > 1e-30:
                    contributions.append((term, v))

            contributions.sort(key=lambda x: -abs(x[1]))
            prod_sum = sum(v for _, v in contributions if v > 0)
            loss_sum = sum(-v for _, v in contributions if v < 0)
            net = prod_sum - loss_sum
            tau = abs(c / net) if abs(net) > 1e-30 and c > 1e-30 else float('inf')
            print(f"     prod={prod_sum:.2e}  loss={loss_sum:.2e}  net={net:+.2e}  "
                  f"tau_net={tau:.1e}s  (c/net inverse)")
            T_layer = ts.state.temperature[0, L].item()
            for term, v in contributions[:TOP_N]:
                sign = "+" if v > 0 else "-"
                k_val, k_src = reaction_rate_constant(term, T_layer, den)
                k_str = f"k={k_val:.2e}({k_src})" if k_val is not None else f"({k_src})"
                params = format_term_params(term)
                print(f"       {sign} {abs(v):.2e}  {label_term(term):<48s} {k_str}  [{params}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
