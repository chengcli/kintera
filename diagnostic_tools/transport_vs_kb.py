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

    # 2b. Prototype: Cheng Titan transport = eddy + molecular + gravity sep.
    #
    # KB's per-species effective diffusion coefficient (kinetgen2X.F:COEFF1):
    #     D(i, k+1/2) = DIFF_i(k+1/2) + Kzz
    # where DIFF_i = 7.3e16 × T_face^0.75 / n_face × sqrt((1+28/m_i)/(1+28/16))
    # is the Cheng Titan molecular diffusion formula (m_i in amu).
    #
    # The KB flux at face k+1/2 has eddy (mixing-ratio) + molecular (concentration
    # gradient + gravity-separation), combined into the form
    #     F = -D × (∂_z c + c / H_i)
    # with the effective scale height
    #     H_i(k+1/2) = D × SCALE / (Kzz + DIFF_i × ZMM_i × FA)
    # where SCALE = dz / log(n[k]/n[k+1]), ZMM_i = m_i / m_avg, and the
    # thermal-diffusion factor FA = 1 + (1 - ZMM_i)(SCALE/ZMM_i)(2/(T[k]+T[k+1]))
    # ((T[k+1]-T[k])/dz). With α=0 (no thermal diffusion) and roughly isothermal
    # atmosphere, FA ≈ 1.
    #
    # Verification check: for pure-eddy (DIFF=0) and isothermal, the form
    # reduces to ∂_z(K n_tot ∂_z(c/n_tot)) — the mixing-ratio eddy diffusion.
    conc_np = titan_state.concentration[0].detach().cpu().numpy()  # (nlyr, nspecies)
    dx1f = titan_state.state.dx1f.detach().cpu().numpy()           # (nlyr,)
    kzz_np = titan_state.kzz[0].detach().cpu().numpy()             # (nlyr,)
    diff_scale = species_diffusion_scale.detach().cpu().numpy()    # (nspecies,)
    ntot = np.array(kb_atm.density)                                # (nlyr_kb,)
    temp = np.array(kb_atm.temperature)
    nlyr_kt = conc_np.shape[0]
    if ntot.shape[0] < nlyr_kt:
        pad = np.zeros(nlyr_kt - ntot.shape[0])
        ntot = np.concatenate([ntot, pad])
        temp = np.concatenate([temp, pad])
    else:
        ntot = ntot[:nlyr_kt]
        temp = temp[:nlyr_kt]
    last_real_lyr = int((titan_state.density[0] > 0).nonzero(as_tuple=True)[0][-1])

    # Cell-center altitudes from KB
    alt = np.array(kb_atm.altitude)
    if alt.shape[0] < nlyr_kt:
        alt = np.concatenate([alt, np.full(nlyr_kt - alt.shape[0], alt[-1])])
    alt = alt[:nlyr_kt] * 1.0e5  # km -> cm

    # Per-species molecular masses (amu)
    from kintera.kinetics_base.titan.physics import _kinetics_base_species_mass_amu
    pun_metadata_dict = kt.kinetics_base_species_metadata_from_pun(
        str(TITAN / "kindata_yy_clean/Cheng_ions_c6h7+_v3_H2CN.pun")
    )
    masses = np.array(
        [_kinetics_base_species_mass_amu(s, pun_metadata_dict) for s in species]
    )

    # Average atmospheric mass at each cell center.
    # Use mass-weighted avg of all gas species concentrations.
    avg_mass = np.zeros(nlyr_kt)
    for k in range(last_real_lyr + 1):
        if ntot[k] <= 0:
            continue
        avg_mass[k] = np.sum(conc_np[k, :] * masses) / ntot[k]

    # Face-quantities arrays: index k = face between cell k-1 and cell k.
    # We'll fill face k+1 (between cells k and k+1) for k in 0..last_real_lyr-1.
    nspc = conc_np.shape[1]
    d_face = np.zeros((last_real_lyr + 1, nspc))   # D(i, k+1/2)
    h_face = np.zeros((last_real_lyr + 1, nspc))   # H(i, k+1/2)
    del_face = np.zeros(last_real_lyr + 1)         # dz of face
    for k in range(last_real_lyr):
        dz_face = alt[k + 1] - alt[k]
        if dz_face <= 0:
            continue
        del_face[k] = dz_face
        n_face = 0.5 * (ntot[k] + ntot[k + 1])
        t_face = 0.5 * (temp[k] + temp[k + 1])
        k_face = 0.5 * (kzz_np[k] + kzz_np[k + 1])
        ratio = ntot[k] / ntot[k + 1] if ntot[k + 1] > 0 else 1.0
        scale = dz_face / np.log(ratio) if ratio > 1.0 else dz_face
        m_avg_face = 0.5 * (avg_mass[k] + avg_mass[k + 1])
        if m_avg_face <= 0:
            m_avg_face = 28.0
        dT_dz = (temp[k + 1] - temp[k]) / dz_face
        for s in range(nspc):
            m_i = masses[s]
            zmm = m_i / m_avg_face
            diff = (7.3e16 * (t_face ** 0.75) / n_face
                    * np.sqrt((1.0 + 28.0 / m_i) / (1.0 + 28.0 / 16.0)))
            fa = 1.0 + (1.0 - zmm) * (scale / zmm) * (1.0 / t_face) * dT_dz
            d_eff = diff + k_face * diff_scale[s]
            denom = (k_face * diff_scale[s]) + diff * zmm * fa
            if denom <= 0:
                d_face[k, s] = 0.0
                h_face[k, s] = scale
                continue
            d_face[k, s] = d_eff
            h_face[k, s] = d_eff * scale / denom

    # Scharfetter-Gummel exponential discretization with spherical r²
    # geometry, mirroring kinetgen2X.F:COEFF1 lines 5165-5202.
    # Tridiagonal at interior cell k (k = 1..last_real_lyr-1):
    #   A[k] c[k-1] + B[k] c[k] + C[k] c[k+1] = -transport_div[k]
    # so transport_div[k] = -(A c[k-1] + B c[k] + C c[k+1]).
    PRAD = 2575e5  # Titan radius in cm (placeholder; refine if needed)
    mr_trans = np.zeros_like(conc_np)
    for k in range(1, last_real_lyr):
        del_low = del_face[k - 1]
        del_up = del_face[k]
        if del_low <= 0 or del_up <= 0:
            continue
        smdel = (alt[k + 1] - alt[k - 1]) / 2.0
        # Spherical-geometry factors at lower & upper faces relative to cell k.
        face_low = PRAD + 0.5 * (alt[k - 1] + alt[k])
        face_up = PRAD + 0.5 * (alt[k] + alt[k + 1])
        center = PRAD + alt[k]
        rq = (face_low / center) ** 2  # lower face
        rp = (face_up / center) ** 2   # upper face
        for s in range(nspc):
            d_low = d_face[k - 1, s]
            d_up = d_face[k, s]
            if d_low <= 0 or d_up <= 0:
                continue
            ss_low = 0.5 * del_low / h_face[k - 1, s]
            ss_up = 0.5 * del_up / h_face[k, s]
            # Clamp SS to avoid overflow with exp; should be reasonable
            ss_low = float(np.clip(ss_low, -50, 50))
            ss_up = float(np.clip(ss_up, -50, 50))
            d05p = np.exp(ss_low) / del_low
            d05m = np.exp(-ss_low) / del_low
            d15p = np.exp(ss_up) / del_up
            d15m = np.exp(-ss_up) / del_up
            qq = rq * d_low
            pp = rp * d_up
            A = -qq * d05m / smdel
            B = (pp * d15m + qq * d05p) / smdel
            C = -pp * d15p / smdel
            mr_trans[k, s] = -(A * conc_np[k - 1, s]
                               + B * conc_np[k, s]
                               + C * conc_np[k + 1, s])

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

    # 4. Build per-species comparison (kintera concentration form vs KB)
    nlyr = kt_trans.shape[0]
    last_real = 39
    species_to_kt = {s: i for i, s in enumerate(species)}
    rows: list[tuple] = []
    mr_rows: list[tuple] = []
    for iv, name in iv_to_name.items():
        if iv not in pstor:
            continue
        if name not in species_to_kt:
            continue
        sp_idx = species_to_kt[name]
        chem_net_kb = pstor[iv][:nlyr] + dstor[iv][:nlyr]  # (PSTOR + DSTOR)
        trans_kb = -chem_net_kb
        trans_kt = kt_trans[:, sp_idx]
        trans_mr = mr_trans[:, sp_idx]
        for L in range(1, last_real):
            tk = trans_kt[L]
            tb = trans_kb[L]
            tm = trans_mr[L]
            if abs(tk) < 1e-20 and abs(tb) < 1e-20 and abs(tm) < 1e-20:
                continue
            denom = max(abs(tk), abs(tb), 1e-30)
            rel = abs(tk - tb) / denom
            rows.append((rel, name, L, tk, tb, kb_atm.altitude[L]))
            denom_mr = max(abs(tm), abs(tb), 1e-30)
            rel_mr = abs(tm - tb) / denom_mr
            mr_rows.append((rel_mr, name, L, tm, tb, kb_atm.altitude[L]))

    rows.sort(key=lambda r: -r[0])
    mr_rows.sort(key=lambda r: -r[0])

    # 4b. Mixing-ratio prototype summary
    rel_mr_arr = np.array([r[0] for r in mr_rows])
    print(f"\n=== Prototype: mixing-ratio diffusion form vs KB ===")
    print(f"Total interior cells: {len(rel_mr_arr)}")
    if len(rel_mr_arr) > 0:
        print(f"  median rel diff: {np.median(rel_mr_arr):.3f}")
        print(f"  90th pctile:     {np.percentile(rel_mr_arr, 90):.3f}")
        print(f"  max rel diff:    {rel_mr_arr.max():.3f}")
        print(f"  fraction with rel < 0.1:   {(rel_mr_arr < 0.1).mean():.3f}")
        print(f"  fraction with rel < 0.01:  {(rel_mr_arr < 0.01).mean():.3f}")
        print(f"\n  Top 10 worst (species, level) for MR prototype:")
        for r, (rel, name, L, tm, tb, alt) in enumerate(mr_rows[:10]):
            print(f"    {r+1:>3d} {name:>14s} L{L:<3d} alt={alt:>7.1f}"
                  f"  mr={tm:>11.3e}  kb={tb:>11.3e}  rel={rel:>6.2f}")
    print()

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
