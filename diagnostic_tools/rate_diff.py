"""G5: per-reaction rate diff kintera vs KINETICS-base.

For a target species (e.g. NH4+), compare:
- KB's per-reaction production / loss rate (cm^-3/s per altitude per
  reaction id), read from ``kb_run/prod+loss/<species>_prod.dat`` and
  ``_loss.dat``.
- kintera's contribution from each ``KBTitanSourceTerm`` whose reactants
  and products match a KB reaction, evaluated at the same atmospheric
  state.

The .pun file gives kintera one set of reaction IDs and KB's Fortran
gives a different (re-numbered) set, so the only reliable way to match
is by reactants/products signature. We use ``Reactions.dat`` from the
KB output directory to build that mapping.

Usage::

    python -m diagnostic_tools.rate_diff NH4+ 25
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Iterable

import numpy as np
import torch

import kintera as kt


def parse_kb_prodloss(path: pathlib.Path) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Parse a KB ``<species>_<prod|loss>.dat`` file.

    Returns ``(kb_reaction_ids, altitudes_km, rates[alt, rxn])``.

    .. note::

       KB Fortran has a known off-by-one bug in the loss-file write
       (``kinetgen1X.F:10995`` ``write(89,...) srate(_, 1, 2, kk)`` where
       the 3rd index should be 1). For NLONX=1, column-major layout
       means ``srate(_, 1, 2, kk)`` reads memory at ``srate(_, 1, 1, kk+1)``
       — the SRATE at the next altitude. Verified by dumping SRATE in
       OUTPUT_IMP and matching row N file value to dumped value at alt
       N+1. The prod file uses the correct ``srate(_, 1, 1, kk)``.

       This parser detects the loss-file pathway by filename and SHIFTS
       the rates DOWN by one row so ``rates[L]`` corresponds to the
       altitude in column 0 of row L. Without this shift, the
       KB-state-injected diagnostic systematically compares kt J at
       altitude L against KB loss data from altitude L+1 — accounting
       for ~half of the high-altitude J overshoot in earlier runs.
    """
    text = path.read_text().splitlines()
    ids_line = text[1].split()
    reaction_ids = [int(x) for x in ids_line]
    rows = []
    for line in text[2:]:
        parts = line.split()
        if not parts:
            continue
        try:
            row = [float(x) for x in parts]
        except ValueError:
            continue
        rows.append(row)
    arr = np.array(rows, dtype=np.float64)
    altitudes_km = arr[:, 0]
    rates = arr[:, 1:]

    if path.name.endswith("_loss.dat") and "_legacy_buggy" in str(path):
        # Legacy compensation for KB runs produced before
        # diagnostics/kb_patches/01-loss-file-altitude-shift.patch.
        # The pre-patch binary wrote srate(_, 1, 2, kk), which aliased
        # to next-altitude data. We only apply this shift if the path
        # contains "_legacy_buggy" as a sentinel — fresh KB runs
        # generated with the patch should NOT be shifted.
        shifted = np.zeros_like(rates)
        shifted[1:] = rates[:-1]
        rates = shifted

    return reaction_ids, altitudes_km, rates


def _normalize_species_label(name: str) -> str:
    """Map the slight differences between KB Reactions.dat species labels and
    kintera species names. Most names match exactly; a few use brackets vs
    parentheses or short forms."""
    return name.strip()


def parse_kb_reactions_dat(
    path: pathlib.Path,
) -> dict[int, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Parse KB's ``Reactions.dat``. Each row looks like::

           474  &    CH4         +    NH3+        =    NH4+        +    CH3                                                               &   3.90E-10  &   0.00  &        0.0  \\

    Returns ``{rid: (reactant_tuple, product_tuple)}``.
    """
    result: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    coeff_re = re.compile(r"^(\d+)([A-Z(].*)$")
    for line in path.read_text().splitlines():
        if "&" not in line or "=" not in line:
            continue
        # Split into 4 fields: id, equation, A, n, Ea (last fields after &).
        fields = [f.strip() for f in line.split("&")]
        if len(fields) < 2:
            continue
        try:
            rid = int(fields[0])
        except ValueError:
            continue
        eqn = fields[1]
        if "=" not in eqn:
            continue
        lhs, rhs = eqn.split("=", 1)
        reactants = _split_eqn_side(lhs, coeff_re)
        products = _split_eqn_side(rhs, coeff_re)
        if not reactants or not products:
            continue
        result[rid] = (tuple(reactants), tuple(products))
    return result


def _split_eqn_side(side: str, coeff_re: re.Pattern[str]) -> list[str]:
    """Split a side of a reaction (LHS or RHS) into species, expanding
    coefficients (e.g. ``2H`` becomes ``["H", "H"]``).

    Tokens are separated by ``+`` with surrounding whitespace. We rejoin
    a trailing ``+`` back onto the previous species so that cations like
    ``NH3+`` survive (``NH3 + +`` after a naive split → ``NH3+``).
    """
    raw_tokens = [t.strip() for t in side.split("+")]
    # Glue empty tokens (which signal a "+" charge marker after the previous
    # token) back onto their predecessor.
    tokens: list[str] = []
    for t in raw_tokens:
        if not t and tokens:
            tokens[-1] = tokens[-1] + "+"
        elif t:
            tokens.append(t)
    out: list[str] = []
    for token in tokens:
        if not token:
            continue
        m = coeff_re.match(token)
        if m:
            n = int(m.group(1))
            sp = m.group(2).strip()
        else:
            n = 1
            sp = token
        sp = _normalize_species_label(sp)
        for _ in range(n):
            out.append(sp)
    return out


def signature(species_list: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(species_list))


def build_titan_state(species_override: list[str] | None = None):
    ROOT = pathlib.Path("/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare")
    TITAN = ROOT / "examples/titan"
    initial = kt.parse_kinetics_base_atmosphere(
        str(TITAN / "kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz")
    )
    species = species_override or list(initial.species_profiles.keys())
    titan_state = kt.build_kinetics_base_titan_state(
        initial,
        species=species,
        boundary_path=str(TITAN / "titan_Cheng_N_ions_H2CN.bc_save"),
        pun_path=str(TITAN / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun"),
    )
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
    return titan_state, source_terms, pun_meta


def kintera_rate_for_term(state, atm_source_for_term) -> np.ndarray:
    """Evaluate one indexed source at the current state and return the full
    (nlyr, nspecies) tendency contribution."""
    lin = atm_source_for_term.linearize(state)
    return lin.tendency.detach().cpu().numpy().squeeze(0)


def diff_rates(target_species: str, nt: int, levels: tuple[int, ...] = (5, 10, 20, 30)) -> None:
    kb_run = pathlib.Path(f"/tmp/kb_run_{nt}")
    kt_dump = pathlib.Path(f"/tmp/kt_traj_{nt}.npz")
    if not kb_run.exists():
        print(f"missing KB dir {kb_run}")
        return
    if not kt_dump.exists():
        print(f"missing kintera dump {kt_dump}")
        return

    print(f"=== Rate diff: {target_species} at NT={nt} ===\n")

    # KB rates: {kb_rid -> rate-profile [alt]}
    prod_path = kb_run / "prod+loss" / f"{target_species}_prod.dat"
    loss_path = kb_run / "prod+loss" / f"{target_species}_loss.dat"
    kb_rates: dict[int, np.ndarray] = {}
    if prod_path.exists():
        prod_ids, _, prod_rates = parse_kb_prodloss(prod_path)
        for j, rid in enumerate(prod_ids):
            kb_rates[rid] = kb_rates.get(rid, 0.0) + prod_rates[:, j]
    if loss_path.exists():
        loss_ids, _, loss_rates = parse_kb_prodloss(loss_path)
        for j, rid in enumerate(loss_ids):
            kb_rates[rid] = kb_rates.get(rid, 0.0) - loss_rates[:, j]
    print(f"KB reactions touching {target_species}: {len(kb_rates)}")

    # KB rid -> reactants/products signature
    rxn_dat = parse_kb_reactions_dat(kb_run / "Reactions.dat")
    kb_signatures = {
        rid: (signature(rxn_dat[rid][0]), signature(rxn_dat[rid][1]))
        for rid in kb_rates
        if rid in rxn_dat
    }

    # kintera state + sources
    d = np.load(kt_dump, allow_pickle=True)
    species_dump = [str(x) for x in d["species"]]
    c_dump = np.asarray(d["concentration"])
    titan_state, source_terms, pun_meta = build_titan_state(species_override=species_dump)
    c = (
        torch.from_numpy(c_dump)
        .to(dtype=titan_state.state.dtype, device=titan_state.state.device)
        .unsqueeze(0)
    )
    titan_state.state.concentration = c
    target_idx = species_dump.index(target_species)

    # kintera signature -> list of source_terms (often 1)
    kintera_by_signature: dict[
        tuple[tuple[str, ...], tuple[str, ...]], list
    ] = {}
    for t in source_terms:
        if target_species not in t.products and target_species not in t.reactants:
            continue
        sig = (signature(t.reactants), signature(t.products))
        kintera_by_signature.setdefault(sig, []).append(t)
    print(f"kintera reactions touching {target_species}: {sum(len(v) for v in kintera_by_signature.values())}")

    matched_kt_rates: dict[int, np.ndarray] = {}
    only_kb: list[int] = []
    for kb_rid, sig in kb_signatures.items():
        if sig in kintera_by_signature:
            # Aggregate kintera rate for all terms matching this signature.
            terms = kintera_by_signature[sig]
            single_atm = kt.build_kinetics_base_titan_atm2d_source_terms(
                titan_state, terms, pun_metadata=pun_meta
            )
            agg = np.zeros(titan_state.state.nlyr, dtype=np.float64)
            for indexed in single_atm:
                tend = kintera_rate_for_term(titan_state.state, indexed)
                agg += tend[:, target_idx]
            matched_kt_rates[kb_rid] = agg
        else:
            only_kb.append(kb_rid)

    print(f"matched: {len(matched_kt_rates)}, KB-only: {len(only_kb)}\n")

    if only_kb:
        print("KB reactions with no kintera signature match:")
        for r in only_kb[:10]:
            sig = rxn_dat.get(r)
            if sig:
                print(f"  {r}: {' + '.join(sig[0])} -> {' + '.join(sig[1])}")
        print()

    # Print top reactions by max rate magnitude, kt vs KB at requested levels
    rids_sorted = sorted(
        matched_kt_rates,
        key=lambda r: -max(np.max(np.abs(kb_rates[r])), np.max(np.abs(matched_kt_rates[r])))
    )
    if not rids_sorted:
        print("(no matched reactions to compare)")
        return
    header = f"{'rid':>5s}  " + "  ".join(f"L{L:<3d}".center(22) for L in levels) + "  reaction"
    print(header)
    sub_header = " " * 5 + "  " + "  ".join("kt        KB       ratio".center(22) for _ in levels)
    print(sub_header)
    for rid in rids_sorted[:25]:
        row = f"{rid:>5d}  "
        for L in levels:
            kt_v = matched_kt_rates[rid][L] if L < len(matched_kt_rates[rid]) else 0.0
            kb_v = kb_rates[rid][L] if L < len(kb_rates[rid]) else 0.0
            if abs(kt_v) < 1e-30 and abs(kb_v) < 1e-30:
                cell = "       zero          "
            elif abs(kb_v) < 1e-30:
                cell = f"{kt_v:8.1e} {kb_v:8.1e}  inf "
            else:
                r = kt_v / kb_v
                cell = f"{kt_v:8.1e} {kb_v:8.1e} {r:>5.2f}"
            row += " " + cell
        sig = rxn_dat[rid]
        row += "  " + " + ".join(sig[0]) + " -> " + " + ".join(sig[1])
        print(row)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("usage: rate_diff.py <species> [NT]")
        sys.exit(1)
    target = args[0]
    nt = int(args[1]) if len(args) > 1 else 25
    diff_rates(target, nt)


if __name__ == "__main__":
    main()
