"""Compare kintera's actinic-flux profile to KB's implied profile.

KB's prod+loss/<sp>_loss.dat lists per-altitude reaction rates
(cm^-3 s^-1). For a photolysis reaction A -> products, the listed
rate equals J × [A], so J = rate / [A] is the per-molecule photo
rate at each altitude.

This script:
1. Picks a target photo reaction (default: rxn 10, C2H2 -> C2H + H)
2. Loads KB state (concentrations) and KB's loss profile
3. Backs out KB's J(z) = loss_rate(z) / [C2H2](z)
4. Builds kintera source terms, finds the matching photo term, runs the
   direct-actinic-flux integrator on KB state, integrates σ(λ) × F(z,λ)
   to get kintera's J(z)
5. Prints kintera/KB ratio per level and per-wavelength flux at L5 and L30

Usage: python disort_vs_kb_actinic.py [reaction_id]
       reaction_id defaults to 10 (pun rxn id for C2H2 -> C2H + H).
"""
from __future__ import annotations

import pathlib
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/sam2/dev/kintera")

import kintera as kt
from kintera.kinetics_base.titan.radiation import (
    _kinetics_base_direct_actinic_flux,
)
from kintera.kinetics_base.titan.models import KBTitanSourceTerm


ROOT = pathlib.Path("/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare")
TITAN = ROOT / "examples/titan"
KB_RUN = pathlib.Path("/tmp/kb_run_500")


def parse_kb_prodloss(path: pathlib.Path) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Parse a KB prod+loss/<species>_<prod|loss>.dat file.

    Each file starts with a header listing reaction ids, then per-altitude
    rates: altitude, rate_rxn_1, rate_rxn_2, ...
    """
    text = path.read_text().splitlines()
    # KB prod+loss format:
    #  alt         reaction number       <-- header
    #                  519  10  664 ...  <-- reaction ids
    #  0.000e+00  4.07e-01  8.5e-04 ...  <-- data
    rxn_ids: list[int] = []
    data_start = 0
    for i, line in enumerate(text):
        parts = line.split()
        if not parts:
            continue
        if parts[0].lower() in ("alt", "altitude") and "reaction" in line.lower():
            # next non-empty line lists the reaction ids
            for jj in range(i + 1, len(text)):
                id_parts = text[jj].split()
                if id_parts:
                    for p in id_parts:
                        try:
                            rxn_ids.append(int(p))
                        except ValueError:
                            pass
                    data_start = jj + 1
                    break
            break
    rows = []
    for line in text[data_start:]:
        parts = line.split()
        if len(parts) < 1 + len(rxn_ids):
            continue
        try:
            row = [float(x) for x in parts[: 1 + len(rxn_ids)]]
        except ValueError:
            continue
        rows.append(row)
    arr = np.array(rows)
    if arr.size == 0:
        return rxn_ids, np.array([]), np.array([])
    altitudes = arr[:, 0]
    rates = arr[:, 1:]
    return rxn_ids, altitudes, rates


def main(target_pun_id: int = 10) -> None:
    print(f"=== Actinic-flux diagnostic for pun rxn {target_pun_id} ===\n")

    # Load KB state to inject
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
    # Inject KB converged state
    kb_atm = kt.parse_kinetics_base_atmosphere(str(KB_RUN / "fort.7"))
    kb_concentration = torch.zeros_like(titan_state.concentration)
    for j, name in enumerate(species):
        if name in kb_atm.species_profiles:
            kb_concentration[0, :, j] = torch.tensor(
                kb_atm.species_profiles[name],
                dtype=titan_state.concentration.dtype,
                device=titan_state.concentration.device,
            )
    titan_state.concentration[:] = kb_concentration

    # Build source terms
    source_terms, _pun_meta = kt.build_kinetics_base_titan_source_terms(
        str(TITAN / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun"),
        special_path=str(TITAN / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.special"),
        boundary_path=str(TITAN / "titan_Cheng_N_ions_H2CN.bc_save"),
        run_input_path=str(TITAN / "ions_c6h7+_H2CN.inp-1"),
        photo_catalog_path=str(TITAN / "Cheng_catalog_v4.dat"),
        cross_dir=str(TITAN / "Cheng_cross"),
        flux_path=str(TITAN / "flare_kin_oct2003.inp"),
    ), kt.kinetics_base_species_metadata_from_pun(
        str(TITAN / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun")
    )

    target_term: KBTitanSourceTerm | None = None
    for t in source_terms:
        if t.reaction_id == target_pun_id and t.kind == "pun_photo_rate_reaction":
            target_term = t
            break
    if target_term is None:
        print(f"!! pun rxn {target_pun_id} not found as pun_photo_rate_reaction")
        return

    reactant = target_term.reactants[0]
    products = target_term.products
    print(f"Target reaction: {reactant} -> {' + '.join(products)}")
    print(f"  source tag: {target_term.parameters.get('source')}")
    print(f"  top-of-atm rate constant: {target_term.parameters.get('rate'):.3e} s^-1")

    # Compute kintera's actinic flux on KB state
    species_index = {n: i for i, n in enumerate(species)}
    wavelengths = target_term.parameters.get("wavelengths") or []
    flux_top = target_term.parameters.get("flux") or []
    cross = target_term.parameters.get("cross_section") or []
    nwave = len(wavelengths)
    print(f"  nwave: {nwave}")

    top_flux = torch.tensor(flux_top, dtype=torch.float64)
    sigma = torch.tensor(cross, dtype=torch.float64)
    actinic = _kinetics_base_direct_actinic_flux(
        target_term,
        titan_state,
        titan_state.concentration.to(dtype=torch.float64),
        species_index,
        top_flux,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    # actinic: (ncol=1, nlyr, nwave)
    j_kt_per_lev = (actinic[0] * sigma.view(1, -1)).sum(dim=-1).cpu().numpy()

    # Get KB's photo rate for the relevant reaction
    # Reaction id in KB Reactions.dat numbering may differ from pun id;
    # back out via the products' loss file (e.g. C2H2 loss file lists rxn 10 entry)
    kb_loss_file = KB_RUN / "prod+loss" / f"{reactant}_loss.dat"
    if not kb_loss_file.exists():
        print(f"  KB loss file not found: {kb_loss_file}")
        return
    rxn_ids, kb_alt, kb_rates = parse_kb_prodloss(kb_loss_file)
    # Find which KB rxn matches our pun rxn 10 by signature - use product set
    # Easier: look at signatures in Reactions.dat
    # For C2H2 -> C2H + H, the KB rxn is whichever produces C2H from C2H2 photolysis
    # In KB this is rxn 10 (Cheng), per Reactions.dat
    # We'll just trust target_pun_id matches KB rxn id for these short Cheng IDs
    if target_pun_id not in rxn_ids:
        # try the canonical signature match instead
        print(f"  rxn {target_pun_id} not in {kb_loss_file.name}; "
              f"available: {rxn_ids[:10]}...")
        return
    col = rxn_ids.index(target_pun_id)
    kb_rate_per_lev = kb_rates[:, col]  # cm^-3 s^-1

    # KB rate / [reactant] = J
    reactant_idx = species_index[reactant]
    n_reactant_per_lev = titan_state.concentration[0, :, reactant_idx].cpu().numpy()
    # Map KB alt grid to kintera alt grid (assume same — 50 levels)
    print(f"\n  KB alt[0]={kb_alt[0]}, kt alt[0]={initial.altitude[0]:.1f}")
    print(f"  KB alt[-1]={kb_alt[-1]}, kt alt[-1]={initial.altitude[-1]:.1f}")

    print(f"\nPer-level J(z) [s^-1]:")
    print(f"  {'lev':>3s}  {'alt(km)':>7s}  {'kt_J':>12s}  {'kb_J':>12s}  {'kt/kb':>8s}  {'kt_n':>10s}")
    for L in [0, 5, 10, 15, 20, 25, 30, 35, 39]:
        if L >= len(j_kt_per_lev):
            continue
        kt_J = j_kt_per_lev[L]
        kb_J = kb_rate_per_lev[L] / max(n_reactant_per_lev[L], 1e-30)
        ratio = kt_J / kb_J if kb_J > 0 else float("nan")
        print(f"  L{L:<2d}  {initial.altitude[L]:>7.1f}  "
              f"{kt_J:>12.3e}  {kb_J:>12.3e}  {ratio:>8.3f}  {n_reactant_per_lev[L]:>10.2e}")


if __name__ == "__main__":
    pun_id = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(pun_id)
