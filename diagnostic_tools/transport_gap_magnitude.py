"""Quantify the magnitude of the kintera-vs-KB transport gap on the KB snapshot.

For each species at each interior altitude:
    abs_gap  = |kt_trans - kb_trans|
    chem_ref = max(|kt_chem|, |kt_trans|, |kb_trans|)   (a natural scale)
    rel_gap  = abs_gap / chem_ref

A species "matters" for SS if rel_gap is large at altitudes where its
chem and transport are comparable. We report:
  (a) per-species: max abs_gap, max rel_gap, the altitude where each
      happens, and a "concentration timescale" tau = conc / abs_gap
      (in seconds — how long before transport error would meaningfully
      move conc if not balanced by chem).
  (b) top-30 (species, altitude) cells ranked by abs_gap (and by
      "chem-relative importance" abs_gap / max(|chem|, eps)).

Uses kintera's concentration-form transport (the current default) — i.e.
the existing build_transport_matrix output — so the numbers reflect what
the actual solver does today.

Usage:
    python diagnostic_tools/transport_gap_magnitude.py
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import torch

import kintera as kt

os.environ.setdefault("KINTERA_TITAN_NETWORK_MODE", "full")

ROOT = pathlib.Path("/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare")
TITAN = ROOT / "examples/titan"
KB_RUN = pathlib.Path("/tmp/kb_run_xport")


def load_kb_pstor_dstor(path: pathlib.Path) -> tuple[dict, dict]:
    pstor: dict[int, list[float]] = {}
    dstor: dict[int, list[float]] = {}
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            iv = int(parts[0])
            k = int(parts[1])
            p = float(parts[2])
            d = float(parts[3])
            pstor.setdefault(iv, [0.0] * 64)[k - 1] = p
            dstor.setdefault(iv, [0.0] * 64)[k - 1] = d
    return pstor, dstor


def main() -> None:
    print("=== Transport KB-vs-KT gap magnitude (KB-converged snapshot) ===\n")

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
    kt_trans = transport.matvec(titan_state.concentration)[0].detach().cpu().numpy()

    pstor, dstor = load_kb_pstor_dstor(KB_RUN / "prod+loss" / "_full_pstor_dstor.dat")

    # Build IV -> name from fort.50 ordering
    iv_to_name: dict[int, str] = {}
    next_iv = 1
    with open(KB_RUN / "fort.50") as f:
        for line in f:
            st = line.strip()
            if not st.startswith("%"):
                continue
            tag = st[1:].split()[0] if " " in st else st[1:]
            if tag in {"ALT", "DEN", "TEMP", "PRE", "EDDY", "WIND"}:
                continue
            iv_to_name[next_iv] = tag
            next_iv += 1

    species_to_kt = {s: i for i, s in enumerate(species)}
    conc = titan_state.concentration[0].detach().cpu().numpy()
    last_real = 39
    alt = np.array(kb_atm.altitude)

    # Per-species summary
    sp_rows = []
    cell_rows = []
    for iv, name in iv_to_name.items():
        if iv not in pstor or name not in species_to_kt:
            continue
        sp_idx = species_to_kt[name]
        chem_kb = np.array(pstor[iv][:last_real + 1]) + np.array(dstor[iv][:last_real + 1])
        trans_kb = -chem_kb  # at SS
        trans_kt = kt_trans[:last_real + 1, sp_idx]
        gap = trans_kt - trans_kb  # signed
        abs_gap = np.abs(gap)
        c_sp = conc[:last_real + 1, sp_idx]

        # Per-cell rows
        for L in range(1, last_real):
            ag = abs_gap[L]
            chem_scale = max(abs(chem_kb[L]), abs(trans_kt[L]), abs(trans_kb[L]), 1e-30)
            rel = ag / chem_scale
            c_val = c_sp[L]
            tau = c_val / ag if ag > 0 else float("inf")
            cell_rows.append(
                (ag, rel, name, L, alt[L], c_val, chem_kb[L], trans_kt[L], trans_kb[L], tau)
            )

        # Species summary
        if abs_gap[1:last_real].size == 0:
            continue
        kmax = int(np.argmax(abs_gap[1:last_real])) + 1
        sp_rows.append({
            "name": name,
            "kmax": kmax,
            "alt": alt[kmax],
            "max_abs_gap": float(abs_gap[kmax]),
            "max_rel_gap": float(abs_gap[kmax] / max(abs(chem_kb[kmax]), abs(trans_kt[kmax]), abs(trans_kb[kmax]), 1e-30)),
            "conc_at_kmax": float(c_sp[kmax]),
            "tau_yr": (c_sp[kmax] / abs_gap[kmax] / 3.15e7) if abs_gap[kmax] > 0 else float("inf"),
        })

    # 1) Top-30 cells by absolute gap
    cell_rows.sort(key=lambda r: -r[0])
    print(f"=== Top 30 (species, altitude) cells by absolute |kt - kb| transport gap ===")
    print(f"  {'rank':>4s} {'species':>14s} {'L':>3s} {'alt':>7s}"
          f" {'|gap|':>11s} {'conc':>11s} {'tau(yr)':>10s}"
          f" {'kt_trans':>11s} {'kb_trans':>11s} {'chem_kb':>11s} {'rel':>5s}")
    for r, row in enumerate(cell_rows[:30]):
        ag, rel, name, L, a, c, ch, tkt, tkb, tau = row
        tau_yr = tau / 3.15e7
        print(f"  {r+1:>4d} {name:>14s} {L:>3d} {a:>7.1f}"
              f" {ag:>11.3e} {c:>11.3e} {tau_yr:>10.2e}"
              f" {tkt:>11.3e} {tkb:>11.3e} {ch:>11.3e} {rel:>5.2f}")

    # 2) Per-species rankings (max_abs_gap and tau)
    sp_rows.sort(key=lambda r: -r["max_abs_gap"])
    print(f"\n=== Top 20 species by max absolute transport gap ===")
    print(f"  {'rank':>4s} {'species':>14s} {'L':>3s} {'alt':>7s}"
          f" {'|gap|':>11s} {'conc':>11s} {'tau(yr)':>10s} {'rel':>5s}")
    for r, sp in enumerate(sp_rows[:20]):
        print(f"  {r+1:>4d} {sp['name']:>14s} {sp['kmax']:>3d} {sp['alt']:>7.1f}"
              f" {sp['max_abs_gap']:>11.3e} {sp['conc_at_kmax']:>11.3e}"
              f" {sp['tau_yr']:>10.2e} {sp['max_rel_gap']:>5.2f}")

    # 3) Distribution of tau
    taus = sorted(sp["tau_yr"] for sp in sp_rows if np.isfinite(sp["tau_yr"]))
    n = len(taus)
    if n > 0:
        def pct(p):
            return taus[max(0, min(n - 1, int(p / 100 * (n - 1))))]
        print(f"\n=== Per-species transport-error timescale (years) ===")
        print(f"  (tau = conc / |kt_trans - kb_trans| — how long before the gap")
        print(f"   would meaningfully move concentration if uncorrected)")
        print(f"  median:    {pct(50):>10.2e} years")
        print(f"  90th %ile: {pct(90):>10.2e} years")
        print(f"  10th %ile: {pct(10):>10.2e} years")
        print(f"  shortest:  {taus[0]:>10.2e} years")
        print(f"  longest:   {taus[-1]:>10.2e} years")


if __name__ == "__main__":
    main()
