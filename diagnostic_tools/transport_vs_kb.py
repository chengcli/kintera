"""Compare kintera's transport divergence against KB's at the converged state.

KB instrumentation dumps full PSTOR/DSTOR (production/loss summed over all
reactions) to ``prod+loss/_full_pstor_dstor.dat``. At steady state:

    transport_div_KB[species, alt] = -(PSTOR + DSTOR)[species, alt]

Kintera's transport divergence on the same KB state is obtained by applying
the transport matrix to the concentration tensor.

Usage:
    python diagnostic_tools/transport_vs_kb.py [TOP_N]
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
KB_RUN = pathlib.Path("/tmp/kb_run_xport")


def load_kb_pstor_dstor(path: pathlib.Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return {species_id: array(NALT)} for PSTOR and DSTOR."""
    pstor: dict[int, np.ndarray] = {}
    dstor: dict[int, np.ndarray] = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            iv = int(parts[0])
            k = int(parts[1])
            p = float(parts[2])
            d = float(parts[3])
            arr_p = pstor.setdefault(iv, np.zeros(64))
            arr_d = dstor.setdefault(iv, np.zeros(64))
            arr_p[k - 1] = p
            arr_d[k - 1] = d
    return pstor, dstor


def build_species_name_map() -> dict[int, str]:
    """Map KB species id (1..NMOL) -> species name from fort.15."""
    # fort.15 (boundary conditions) has one row per varying species with the
    # name in column 5 (or 7 depending on layout). It's only the NVARYF
    # subset though, so we also fall back to fort.50 atm-file species blocks.
    names: dict[int, str] = {}
    with open(KB_RUN / "fort.15") as f:
        lines = f.readlines()
    i = 5  # first species row
    species_id = 1
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 5:
            i += 1
            continue
        name = parts[-1]
        names[species_id] = name
        species_id += 1
        i += 1
    return names


def main(top_n: int = 30) -> None:
    print("=== Kintera vs KB transport divergence (snapshot at converged state) ===\n")

    if not (KB_RUN / "prod+loss" / "_full_pstor_dstor.dat").exists():
        print("KB PSTOR/DSTOR dump missing — re-run instrumented KB first.")
        return

    pstor, dstor = load_kb_pstor_dstor(KB_RUN / "prod+loss" / "_full_pstor_dstor.dat")
    print(f"Loaded KB chem tendencies for {len(pstor)} species\n")

    # 1. Build kintera state with KB converged atmosphere
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
                device=titan_state.concentration.device,
            )
    titan_state.concentration[:] = kb_conc
    titan_state.state.concentration = titan_state.concentration

    # 2. Build the transport matrix and apply to the state
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
    kintera_trans = transport.matvec(titan_state.concentration)
    kt_trans = kintera_trans[0].detach().cpu().numpy()  # (nlyr, nspecies)

    # 3. Map KB species id -> kintera species index
    # The Cheng pun file orders species in the same way as kintera's species list.
    # KB uses internal numbering via the MAPPING ARRAY FOR SPECIES in fort.3.
    # The atmosphere file (kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz) has
    # species in the order they appear after %ALT/%DEN/%TEMP/%PRE/%EDDY/%WIND,
    # which is the same order as kb_atm.species_profiles keys.
    # KB's IV index = position in NMOL list. We need a name-keyed dict.
    # Quickest: try kt_index iteratively; for each kt species name, see if KB has
    # the same name in fort.50 atm species blocks → that's IV.
    species_block_order = list(initial.species_profiles.keys())  # same as kb_atm
    # KB internal IV numbering matches the NMOL species list, which is
    # different from the post-FIX kintera ordering. Walk through fort.50's
    # %SPECIES_NAME blocks and assign sequential IVs starting at 1.
    with open(KB_RUN / "fort.50") as f:
        f50_lines = f.readlines()
    iv_to_name: dict[int, str] = {}
    next_iv = 1
    for line in f50_lines:
        st = line.strip()
        if not st.startswith("%"):
            continue
        tag = st[1:].split()[0] if " " in st else st[1:]
        # Skip the header tags
        if tag in {"ALT", "DEN", "TEMP", "PRE", "EDDY", "WIND"}:
            continue
        iv_to_name[next_iv] = tag
        next_iv += 1

    print(f"Mapped {len(iv_to_name)} KB IV → name slots\n")

    # 4. Build per-species comparison
    nlyr = kt_trans.shape[0]
    last_real = 39
    species_to_kt = {s: i for i, s in enumerate(species)}
    rows: list[tuple] = []
    for iv, name in iv_to_name.items():
        if iv not in pstor:
            continue
        if name not in species_to_kt:
            continue
        sp_idx = species_to_kt[name]
        chem_net_kb = pstor[iv][:nlyr] + dstor[iv][:nlyr]  # (PSTOR + DSTOR)
        trans_kb = -chem_net_kb
        trans_kt = kt_trans[:, sp_idx]
        for L in range(1, last_real):
            tk = trans_kt[L]
            tb = trans_kb[L]
            if abs(tk) < 1e-20 and abs(tb) < 1e-20:
                continue
            denom = max(abs(tk), abs(tb), 1e-30)
            rel = abs(tk - tb) / denom
            rows.append((rel, name, L, tk, tb, kb_atm.altitude[L]))

    rows.sort(key=lambda r: -r[0])

    # 5. Summary stats
    rel_arr = np.array([r[0] for r in rows])
    print(f"Total (species, level) cells with non-trivial transport: {len(rel_arr)}")
    if len(rel_arr) == 0:
        return
    print(f"  median rel diff: {np.median(rel_arr):.3f}")
    print(f"  90th pctile:     {np.percentile(rel_arr, 90):.3f}")
    print(f"  max rel diff:    {rel_arr.max():.3f}")
    print(f"  fraction with rel < 0.1:   {(rel_arr < 0.1).mean():.3f}")
    print(f"  fraction with rel < 0.01:  {(rel_arr < 0.01).mean():.3f}")

    print(f"\n=== Top {top_n} worst (species, level) ===")
    print(f"  {'rank':>4s} {'species':>14s} {'lev':>4s} {'alt':>7s}"
          f" {'kt_trans':>13s} {'kb_trans':>13s} {'rel':>8s}")
    for r, (rel, name, L, tk, tb, alt) in enumerate(rows[:top_n]):
        print(f"  {r+1:>4d} {name:>14s} {L:>4d} {alt:>7.1f}"
              f" {tk:>13.3e} {tb:>13.3e} {rel:>8.2f}")


if __name__ == "__main__":
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(top_n)
