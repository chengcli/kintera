"""Snapshot transport-balance diagnostic.

At KB's converged steady state:
    chem_net + transport_div + escape ≈ 0   (per species, per altitude)

This script injects KB's converged state into kintera, evaluates kintera's
chemistry net rate and transport divergence on that state, and reports
the residual. A non-trivial residual at any altitude for any species
points to a kintera-vs-KB disagreement somewhere in the chem / transport /
escape stack.

Boundary-pinned species (fixed mixing ratios, lower-BC, upper-BC) are
NOT expected to satisfy the SS condition — they're held to a value
that overrides the physical balance. Flag those rows but don't ring
the alarm.

Usage:
    python diagnostic_tools/snapshot_transport_balance.py [TOP_N]

Where TOP_N (default 25) is the number of worst (species, level) cells
to print.
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import torch

import kintera as kt

os.environ.setdefault("KINTERA_TITAN_NETWORK_MODE", "full")

ROOT = pathlib.Path("/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare")
TITAN = ROOT / "examples/titan"
KB_RUN = pathlib.Path("/tmp/kb_run_500")


def main(top_n: int = 25) -> None:
    print("=== Snapshot transport-balance diagnostic ===\n")

    # 1. Build kintera state with KB's atmosphere
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

    # 2. Inject KB's converged concentrations
    kb_atm = kt.parse_kinetics_base_atmosphere(str(KB_RUN / "fort.7"))
    kb_conc = torch.zeros_like(titan_state.concentration)
    for j, name in enumerate(species):
        if name in kb_atm.species_profiles:
            kb_conc[0, :, j] = torch.tensor(
                kb_atm.species_profiles[name],
                dtype=titan_state.concentration.dtype,
                device=titan_state.concentration.device,
            )
    titan_state.concentration[:] = kb_conc
    titan_state.state.concentration = titan_state.concentration

    # 3. Build all source terms and the transport matrix
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

    atm_sources = kt.build_kinetics_base_titan_atm2d_source_terms(
        titan_state, source_terms, pun_metadata=pun_meta
    )

    species_diffusion_scale = kt.kinetics_base_titan_species_diffusion_scale(
        titan_state.species,
        dtype=titan_state.state.dtype,
        device=titan_state.state.device,
    )

    transport = kt.build_transport_matrix(
        titan_state.state,
        titan_state.kzz,
        species_diffusion_scale=species_diffusion_scale,
    )

    # 4. Apply both operators to the KB state
    #    transport_div  has units cm^-3 s^-1, same as chem_net
    transport_div = transport.matvec(titan_state.concentration)
    chem_lin = kt.build_source_linearization(titan_state.state, atm_sources)
    chem_net = chem_lin.tendency

    total = chem_net + transport_div  # at SS this should be ~0

    chem_arr = chem_net[0].detach().cpu().numpy()        # (nlyr, nspecies)
    trans_arr = transport_div[0].detach().cpu().numpy()
    total_arr = total[0].detach().cpu().numpy()
    conc_arr = titan_state.concentration[0].detach().cpu().numpy()

    nlyr, nspc = chem_arr.shape
    species_names = list(titan_state.species)

    # 5. Boundary-pinned set: species at L=0 or last real level that
    #    are pinned have non-physical residuals — don't ring alarm.
    # For brevity we'll just flag levels 0 and last-real-level as pinned;
    # interior levels are expected to balance.
    last_real = 39  # 40 real levels, L0..L39

    # Build a "denominator" for relative residual: max(|chem|, |transport|)
    denom = np.maximum(np.abs(chem_arr), np.abs(trans_arr))
    # Avoid divide-by-zero
    denom = np.where(denom > 1e-30, denom, 1e-30)
    rel = np.abs(total_arr) / denom

    # 6. Print summary stats per level (interior only)
    print(f"{'lev':>3s} {'alt(km)':>8s} {'maxAbs_chem':>13s} {'maxAbs_trans':>13s}"
          f" {'maxAbs_resid':>13s} {'max_rel_resid':>13s} {'argmax_species':>14s}")
    for L in range(nlyr):
        if L > last_real:
            break
        # Only consider non-pinned interior
        # (we still print the table — pinning is per species, not per level)
        argmax = int(np.argmax(rel[L]))
        print(f"{L:>3d} {kb_atm.altitude[L]:>8.1f} {np.max(np.abs(chem_arr[L])):>13.3e}"
              f" {np.max(np.abs(trans_arr[L])):>13.3e} {np.max(np.abs(total_arr[L])):>13.3e}"
              f" {rel[L, argmax]:>13.3f} {species_names[argmax]:>14s}")

    # 7. Top worst (species, level) cells across interior
    print(f"\n=== Top {top_n} (species, level) residuals (interior, L1..L{last_real-1}) ===")
    # mask boundary levels
    interior = np.zeros_like(rel, dtype=bool)
    interior[1:last_real, :] = True
    flat_idx = np.argsort(-(rel * interior).ravel())[:top_n]
    print(f"  {'rank':>4s} {'species':>14s} {'lev':>4s} {'alt':>7s}"
          f" {'conc':>11s} {'chem':>13s} {'trans':>13s} {'resid':>13s} {'rel':>8s}")
    for r, fi in enumerate(flat_idx):
        L, S = divmod(int(fi), nspc)
        print(f"  {r+1:>4d} {species_names[S]:>14s} {L:>4d} {kb_atm.altitude[L]:>7.1f}"
              f" {conc_arr[L, S]:>11.3e} {chem_arr[L, S]:>13.3e}"
              f" {trans_arr[L, S]:>13.3e} {total_arr[L, S]:>13.3e}"
              f" {rel[L, S]:>8.2f}")


if __name__ == "__main__":
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    main(top_n)
