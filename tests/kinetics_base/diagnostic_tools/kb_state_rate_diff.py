"""KB-state prod/loss diagnostic: inject KB's converged abundances into kintera
and diff the per-reaction tendencies against KB's own prod+loss/*.dat at the
same snapshot.

The point: when kintera and KB diverge after a long integration, it's hard to
tell which reaction is wrong from the drifted state alone — feedback loops
mask the upstream culprit. By INJECTING KB's state into kintera and computing
tendencies WITHOUT integrating, every reaction's rate becomes a pure function
of (rate-constant × KB-faithful concentrations). A divergence vs KB's
prod+loss is then a direct fingerprint of:
  - A missing reaction (KB has it, kintera doesn't)
  - A wrong rate constant
  - A wrong stoichiometry (different products in kintera vs KB)
  - A wrong species name (matched-by-signature would miss it)

Run across multiple snapshots (start / mid / converged) to see how the gap
emerges as KB integrates — gives an immediate ordering of "which reactions
were already wrong at t=0 vs which only matter once cation pool builds up".

Usage::

    python -m diagnostic_tools.kb_state_rate_diff <species> [NT [NT ...]]
    python diagnostic_tools/kb_state_rate_diff.py NH4+ 1 50 500
    python diagnostic_tools/kb_state_rate_diff.py H 50  # single snapshot
"""

from __future__ import annotations

import pathlib
import sys
import os

import numpy as np
import torch

# Quiet pyright in this scratch tool
os.environ.setdefault("PYTHONHASHSEED", "0")

import kintera as kt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from diagnostic_tools.rate_diff import (
    parse_kb_prodloss,
    parse_kb_reactions_dat,
    signature,
    kintera_rate_for_term,
)


ROOT = pathlib.Path("/home/sam2/dev/kintera/tests/kinetics_base/diagnostics/KINETICS-base-compare")
TITAN = ROOT / "examples/titan"


def build_titan_state_with_kb_concentration(kb_run: pathlib.Path):
    """Build a titan_state whose concentration is KB's state at the given
    snapshot (NTIME). KB writes its converged profiles into ``fort.7``; we
    parse that as the initial atmosphere so the resulting titan_state holds
    KB-faithful abundances at every level.

    NOTE: ``build_kinetics_base_titan_state`` applies conversions designed for
    the INITIAL atmosphere file (mixing ratios in some entries, e.g.
    ``kinetics_base_cheng_cold_trap_mixing_ratio`` for CH4) — these corrupt
    a number-density fort.7. We OVERWRITE the concentration tensor with the
    raw species_profiles after building, ensuring every species is the exact
    KB number-density value at every level.
    """
    kb_atm = kt.parse_kinetics_base_atmosphere(str(kb_run / "fort.7"))
    species = list(kb_atm.species_profiles.keys())
    titan_state = kt.build_kinetics_base_titan_state(
        kb_atm,
        species=species,
        boundary_path=str(TITAN / "titan_Cheng_N_ions_H2CN.bc_save"),
        pun_path=str(TITAN / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun"),
    )
    # Force the concentration to the raw KB profiles, bypassing conversions.
    nlyr = titan_state.state.concentration.shape[1]
    for i, sp in enumerate(species):
        prof = kb_atm.species_profiles[sp]
        for L in range(min(nlyr, len(prof))):
            titan_state.state.concentration[0, L, i] = float(prof[L])
    # Refresh the cached snapshot stored on the model (used internally as
    # the initial-state reference).
    titan_state.concentration = titan_state.state.concentration.clone()
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
    return titan_state, source_terms, pun_meta, species


def compute_kt_rates_at_kb_state(
    target_species: str,
    kb_run: pathlib.Path,
    levels: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict, list[str]]:
    """Return (kt_rates_by_kb_rid, kb_rates_by_rid, reaction_signatures,
    species_list)."""
    titan_state, source_terms, pun_meta, species = (
        build_titan_state_with_kb_concentration(kb_run)
    )

    # KB rates for target_species at this snapshot
    prod_path = kb_run / "prod+loss" / f"{target_species}_prod.dat"
    loss_path = kb_run / "prod+loss" / f"{target_species}_loss.dat"
    kb_rates: dict[int, np.ndarray] = {}
    if prod_path.exists():
        ids, _, rates = parse_kb_prodloss(prod_path)
        for j, rid in enumerate(ids):
            kb_rates[rid] = kb_rates.get(rid, 0.0) + rates[:, j]
    if loss_path.exists():
        ids, _, rates = parse_kb_prodloss(loss_path)
        for j, rid in enumerate(ids):
            kb_rates[rid] = kb_rates.get(rid, 0.0) - rates[:, j]

    rxn_dat = parse_kb_reactions_dat(kb_run / "Reactions.dat")
    kb_signatures = {
        rid: (signature(rxn_dat[rid][0]), signature(rxn_dat[rid][1]))
        for rid in kb_rates if rid in rxn_dat
    }

    # Group kintera source terms by signature
    kintera_by_signature: dict[tuple, list] = {}
    for t in source_terms:
        if target_species not in t.products and target_species not in t.reactants:
            continue
        sig = (signature(t.reactants), signature(t.products))
        kintera_by_signature.setdefault(sig, []).append(t)

    target_idx = species.index(target_species)
    matched_kt_rates: dict[int, np.ndarray] = {}
    for kb_rid, sig in kb_signatures.items():
        if sig not in kintera_by_signature:
            continue
        terms = kintera_by_signature[sig]
        single_atm = kt.build_kinetics_base_titan_atm2d_source_terms(
            titan_state, terms, pun_metadata=pun_meta
        )
        agg = np.zeros(titan_state.state.nlyr, dtype=np.float64)
        for indexed, term in zip(single_atm, terms):
            tend = kintera_rate_for_term(titan_state.state, indexed)
            # KB's prod+loss output stores REACTION rate (not species
            # tendency); kintera's `.linearize()` returns species tendency =
            # net_stoichiometry × reaction_rate. Divide by |net_coef| (keeping
            # tend's sign — KB stores loss as negative, production as positive
            # in the same file, signed by whether the species appears as
            # reactant or product).
            net_coef = abs(
                sum(1 for p in term.products if p == target_species)
                - sum(1 for r in term.reactants if r == target_species)
            )
            if net_coef == 0:
                continue
            agg += tend[:, target_idx] / float(net_coef)
        matched_kt_rates[kb_rid] = agg

    return matched_kt_rates, kb_rates, rxn_dat, species


def rank_divergences(
    kt_rates: dict[int, np.ndarray],
    kb_rates: dict[int, np.ndarray],
    rxn_dat: dict,
    levels: tuple[int, ...],
) -> list[tuple[float, int, str, list[tuple[float, float, float]]]]:
    """Return ranked list ``(score, rid, equation, [(L, kt, kb)...])`` with
    score = max abs log10(kt/kb) over the requested levels."""
    rows = []
    for rid in sorted(kt_rates):
        kt_arr = kt_rates[rid]
        kb_arr = kb_rates.get(rid, np.zeros_like(kt_arr))
        max_log = 0.0
        per_lev = []
        for L in levels:
            if L >= len(kt_arr) or L >= len(kb_arr):
                continue
            kt_v = float(kt_arr[L]); kb_v = float(kb_arr[L])
            per_lev.append((L, kt_v, kb_v))
            if abs(kt_v) > 1e-30 and abs(kb_v) > 1e-30:
                lr = abs(np.log10(abs(kt_v) / abs(kb_v)))
                if np.isfinite(lr):
                    max_log = max(max_log, lr)
            elif abs(kt_v) > 1e-25 and abs(kb_v) < 1e-30:
                max_log = max(max_log, 30.0)
            elif abs(kb_v) > 1e-25 and abs(kt_v) < 1e-30:
                max_log = max(max_log, 30.0)
        sig = rxn_dat.get(rid)
        eqn = (" + ".join(sig[0]) + " -> " + " + ".join(sig[1])) if sig else "?"
        rows.append((max_log, rid, eqn, per_lev))
    rows.sort(reverse=True)
    return rows


def run_snapshot(target: str, nt: int, levels: tuple[int, ...]) -> None:
    kb_run = pathlib.Path(f"/tmp/kb_run_{nt}")
    if not kb_run.exists():
        print(f"  [skip] {kb_run} not found")
        return
    print(f"\n=== {target} @ NT={nt} (KB state injected) ===")
    kt_rates, kb_rates, rxn_dat, species = compute_kt_rates_at_kb_state(
        target, kb_run, levels
    )
    n_match = len(set(kt_rates.keys()) & set(kb_rates.keys()))
    n_kb_only = len(set(kb_rates.keys()) - set(kt_rates.keys()))
    print(f"  KB rxns touching {target}: {len(kb_rates)}, kt-matched: {n_match}, KB-only: {n_kb_only}")

    if n_kb_only > 0:
        print(f"\n  KB-only reactions (no signature-match in kintera):")
        for rid in sorted(set(kb_rates.keys()) - set(kt_rates.keys())):
            sig = rxn_dat.get(rid)
            if sig:
                kb_arr = kb_rates[rid]
                peak = max((abs(kb_arr[L]) for L in levels if L < len(kb_arr)), default=0.0)
                if peak > 1e-25:
                    print(f"    rxn {rid:5d} (peak |rate|={peak:.2e}): "
                          f"{' + '.join(sig[0])} -> {' + '.join(sig[1])}")

    ranked = rank_divergences(kt_rates, kb_rates, rxn_dat, levels)
    print(f"\n  Top divergences (max |log10(kt/kb)| over levels {list(levels)}):")
    header = "  rank rxn   max-logdev  " + "  ".join(
        f"L{L:<3d}".center(28) for L in levels
    ) + "  reaction"
    print(header)
    sub = " " * 21 + "  " + "  ".join(
        "kt           kb        ratio".center(28) for _ in levels
    )
    print(sub)
    for k, (score, rid, eqn, per_lev) in enumerate(ranked[:20]):
        row = f"  {k+1:>3d}  {rid:>5d}  {score:>10.2f}  "
        for L, kt_v, kb_v in per_lev:
            if abs(kt_v) < 1e-30 and abs(kb_v) < 1e-30:
                cell = "          zero            "
            elif abs(kb_v) < 1e-30:
                cell = f" {kt_v:>10.2e} {kb_v:>10.2e}    inf "
            elif abs(kt_v) < 1e-30:
                cell = f" {kt_v:>10.2e} {kb_v:>10.2e}    0   "
            else:
                r = kt_v / kb_v
                cell = f" {kt_v:>10.2e} {kb_v:>10.2e} {r:>+7.1e}"
            row += " " + cell
        row += "  " + eqn
        print(row)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    target = args[0]
    nts = [int(a) for a in args[1:]] if len(args) > 1 else [50, 500]
    levels = (5, 10, 20, 30)
    print(f"Target species: {target}")
    print(f"Snapshots: {nts}")
    print(f"Levels: {list(levels)}")
    for nt in nts:
        run_snapshot(target, nt, levels)


if __name__ == "__main__":
    main()
