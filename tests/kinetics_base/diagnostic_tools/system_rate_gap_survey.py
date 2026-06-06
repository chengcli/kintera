"""System-wide reaction rate-gap survey vs KB at the converged snapshot.

Inject KB's converged concentrations into kintera. For every KB reaction
that appears in any species's prod+loss/*.dat (top-3 per species per
altitude, unioned), compute kintera's reaction rate and compare against
KB's. Rank by max |log10(kt/kb)| across all altitudes.

Reactions missed by the top-3 dump are by definition not dominant for
any species at any altitude, so they aren't included here.

Output: ranked list of reactions with the worst rate gap, with each
reaction's equation, the altitude where the gap peaks, and the
kintera/KB rate values at that altitude.

Usage:
    python diagnostic_tools/system_rate_gap_survey.py [TOP_N]
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import torch

import kintera as kt

os.environ.setdefault("KINTERA_TITAN_NETWORK_MODE", "full")

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from diagnostic_tools.rate_diff import (
    parse_kb_prodloss,
    parse_kb_reactions_dat,
    signature,
    kintera_rate_for_term,
)


ROOT = pathlib.Path("/home/sam2/dev/kintera/tests/kinetics_base/diagnostics/KINETICS-base-compare")
TITAN = ROOT / "examples/titan"
KB_RUN = pathlib.Path("/tmp/kb_run_xport")


def main(top_n: int = 60) -> None:
    print("=== System-wide reaction rate-gap survey (KB state injected) ===\n")

    initial = kt.parse_kinetics_base_atmosphere(
        str(TITAN / "kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz")
    )
    species = list(initial.species_profiles.keys())
    titan_state = kt.build_kinetics_base_titan_state(
        initial,
        species=species,
        boundary_path=str(TITAN / "titan_Cheng_N_ions_H2CN.bc_save"),
        pun_path=str(TITAN / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun"),
    )
    kb_atm = kt.parse_kinetics_base_atmosphere(str(KB_RUN / "fort.7"))
    kb_conc = torch.zeros_like(titan_state.concentration)
    for j, name in enumerate(species):
        if name in kb_atm.species_profiles:
            kb_conc[0, :, j] = torch.tensor(
                kb_atm.species_profiles[name],
                dtype=titan_state.concentration.dtype,
            )
    titan_state.concentration[:] = kb_conc
    titan_state.state.concentration = titan_state.concentration

    source_terms, pun_meta = (
        kt.build_kinetics_base_titan_source_terms(
            str(TITAN / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun"),
            special_path=str(TITAN / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.special"),
            boundary_path=str(TITAN / "titan_Cheng_N_ions_H2CN.bc_save"),
            run_input_path=str(TITAN / "ions_c6h7+_H2CN.inp-1"),
            photo_catalog_path=str(TITAN / "Cheng_catalog_v4.dat"),
            cross_dir=str(TITAN / "Cheng_cross"),
            flux_path=str(TITAN / "flare_kin_oct2003.inp"),
        ),
        kt.kinetics_base_species_metadata_from_pun(
            str(TITAN / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun")
        ),
    )

    rxn_dat = parse_kb_reactions_dat(KB_RUN / "Reactions.dat")

    # Gather all KB reaction rates from every prod+loss file
    print("Scanning prod+loss/*.dat to collect KB reaction rates ...")
    kb_rate_by_rid: dict[int, np.ndarray] = {}
    for f in (KB_RUN / "prod+loss").glob("*_prod.dat"):
        try:
            ids, _, rates = parse_kb_prodloss(f)
        except Exception:
            continue
        for j, rid in enumerate(ids):
            if rid not in kb_rate_by_rid:
                kb_rate_by_rid[rid] = np.array(rates[:, j])
            else:
                # Take the max-magnitude across all reporting species since
                # KB stores reaction rate (same value across species; we keep
                # the largest non-zero to handle parser quirks).
                arr = kb_rate_by_rid[rid]
                kb_rate_by_rid[rid] = np.where(
                    np.abs(rates[:, j]) > np.abs(arr), np.array(rates[:, j]), arr
                )
    for f in (KB_RUN / "prod+loss").glob("*_loss.dat"):
        try:
            ids, _, rates = parse_kb_prodloss(f)
        except Exception:
            continue
        for j, rid in enumerate(ids):
            if rid not in kb_rate_by_rid:
                kb_rate_by_rid[rid] = np.array(rates[:, j])
            else:
                arr = kb_rate_by_rid[rid]
                kb_rate_by_rid[rid] = np.where(
                    np.abs(rates[:, j]) > np.abs(arr), np.array(rates[:, j]), arr
                )

    print(f"Collected {len(kb_rate_by_rid)} KB reactions from prod+loss/")

    # Build kintera-by-signature lookup
    kintera_by_sig: dict[tuple, list] = {}
    for t in source_terms:
        sig = (signature(t.reactants), signature(t.products))
        kintera_by_sig.setdefault(sig, []).append(t)

    # Evaluate every matched kintera term once
    species_to_idx = {s: i for i, s in enumerate(species)}
    print("Evaluating kintera source terms on KB state ...")
    rows = []
    kb_only = 0
    matched = 0
    for rid, kb_arr in kb_rate_by_rid.items():
        if rid not in rxn_dat:
            continue
        reactants, products = rxn_dat[rid]
        sig = (signature(reactants), signature(products))
        if sig not in kintera_by_sig:
            kb_only += 1
            continue
        terms = kintera_by_sig[sig]
        # Use the first matching term (signature is unique per reaction id
        # in kintera too, modulo duplicates)
        first_term = terms[0]
        # Aggregate kintera tendency for this signature
        single_atm = kt.build_kinetics_base_titan_atm2d_source_terms(
            titan_state, terms, pun_metadata=pun_meta
        )
        agg = np.zeros(titan_state.state.nlyr, dtype=np.float64)
        # Pick a reference species to extract reaction rate from tendency.
        # We use the first product that appears with stoichiometry 1.
        ref_species = None
        ref_coef = 1
        for prod in first_term.products:
            if prod in species_to_idx:
                ref_species = prod
                ref_coef = sum(1 for p in first_term.products if p == prod) - sum(
                    1 for r in first_term.reactants if r == prod
                )
                if ref_coef != 0:
                    break
        if ref_species is None:
            continue
        ref_idx = species_to_idx[ref_species]
        for indexed, term in zip(single_atm, terms):
            tend = kintera_rate_for_term(titan_state.state, indexed)
            agg += tend[:, ref_idx] / float(abs(ref_coef))
        kt_arr = agg
        matched += 1

        # Score = max |log10(|kt|/|kb|)| across altitudes where both are above
        # a noise floor; cap when one is zero
        # Also track absolute gap to flag big rate disagreements
        nlyr = min(len(kt_arr), len(kb_arr), 40)
        max_logdev = 0.0
        worst_L = 0
        worst_kt = 0.0
        worst_kb = 0.0
        peak_kb = 0.0
        for L in range(nlyr):
            kt_v = float(kt_arr[L])
            kb_v = float(kb_arr[L])
            if abs(kb_v) > peak_kb:
                peak_kb = abs(kb_v)
            if abs(kt_v) < 1e-30 and abs(kb_v) < 1e-30:
                continue
            if abs(kt_v) < 1e-30:
                if abs(kb_v) > 1e-25:
                    score = 30.0
                else:
                    score = 0.0
            elif abs(kb_v) < 1e-30:
                if abs(kt_v) > 1e-25:
                    score = 30.0
                else:
                    score = 0.0
            else:
                score = abs(np.log10(abs(kt_v) / abs(kb_v)))
            if score > max_logdev:
                max_logdev = score
                worst_L = L
                worst_kt = kt_v
                worst_kb = kb_v
        if peak_kb < 1e-25:
            continue
        eqn = " + ".join(reactants) + " -> " + " + ".join(products)
        rows.append((max_logdev, peak_kb, rid, eqn, worst_L, worst_kt, worst_kb))

    print(f"matched: {matched}, KB-only (no kintera signature match): {kb_only}\n")

    # Rank by max_logdev * weight (peak_kb to emphasize important reactions)
    # First show top by log-deviation
    rows.sort(key=lambda r: (-r[0], -r[1]))

    print(f"=== Top {top_n} reactions by max |log10(kt/kb)| ===")
    print(f"  {'rank':>4s} {'rid':>5s} {'logdev':>7s} {'peak_kb':>11s} {'L*':>3s}"
          f" {'kt*':>11s} {'kb*':>11s}  reaction")
    for r, (logdev, peak_kb, rid, eqn, L, kt_v, kb_v) in enumerate(rows[:top_n]):
        print(f"  {r+1:>4d} {rid:>5d} {logdev:>7.2f} {peak_kb:>11.3e} {L:>3d}"
              f" {kt_v:>11.3e} {kb_v:>11.3e}  {eqn}")

    # Now show top by raw absolute rate gap (different ranking)
    rows.sort(key=lambda r: -abs(r[5] - r[6]) if r[0] > 0 else 0)
    print(f"\n=== Top {top_n} reactions by absolute |kt - kb| at worst-L cell ===")
    print(f"  {'rank':>4s} {'rid':>5s} {'|gap|':>11s} {'logdev':>7s} {'L*':>3s}"
          f" {'kt*':>11s} {'kb*':>11s}  reaction")
    for r, (logdev, peak_kb, rid, eqn, L, kt_v, kb_v) in enumerate(rows[:top_n]):
        gap = abs(kt_v - kb_v)
        print(f"  {r+1:>4d} {rid:>5d} {gap:>11.3e} {logdev:>7.2f} {L:>3d}"
              f" {kt_v:>11.3e} {kb_v:>11.3e}  {eqn}")


if __name__ == "__main__":
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    main(top_n)
