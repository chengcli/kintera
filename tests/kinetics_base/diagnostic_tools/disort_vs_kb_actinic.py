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


ROOT = pathlib.Path("/home/sam2/dev/kintera/tests/kinetics_base/diagnostics/KINETICS-base-compare")
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
    # KB off-by-one bug compensation: pre-patch KB runs wrote
    # srate(_, 1, 2, kk) which aliased to next-altitude data. Fresh
    # runs from the patched KB are correct as-written. We gate the
    # shift on a "_legacy_buggy" sentinel in the path so only the
    # archived buggy run gets corrected; canonical /tmp/kb_run_500/
    # is now patched output.
    if path.name.endswith("_loss.dat") and "_legacy_buggy" in str(path):
        shifted = np.zeros_like(rates)
        shifted[1:] = rates[:-1]
        rates = shifted
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
        if t.reaction_id == target_pun_id and t.kind in (
            "pun_photo_rate_reaction",
            "pun_electron_impact_reaction",
        ):
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
    print(f"  {'lev':>3s}  {'alt(km)':>7s}  {'kt_J':>12s}  {'kb_J':>12s}  {'kt/kb':>8s}  {'tau_sum':>9s}  {'kt_n':>10s}")
    # tau at each level summed over wavelengths (rough scalar tau)
    af = actinic[0].cpu().numpy()  # (nlyr, nwave)
    # Reconstruct column tau as -log(actinic/top_flux), averaged over wavelengths
    # with positive flux (to avoid log of 0)
    f_top = top_flux.cpu().numpy()
    for L in [0, 5, 10, 15, 20, 25, 30, 35, 39]:
        if L >= len(j_kt_per_lev):
            continue
        kt_J = j_kt_per_lev[L]
        kb_J = kb_rate_per_lev[L] / max(n_reactant_per_lev[L], 1e-30)
        ratio = kt_J / kb_J if kb_J > 0 else float("nan")
        # tau per wavelength = -log(actinic/top)/(diurnal_atten_top)
        # but better: tau ≈ -log(actinic/F_top) summed and averaged
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_per_w = -np.log(np.clip(af[L] / np.maximum(f_top, 1.0), 1e-300, 1.0))
        tau_w_avg = tau_per_w[np.isfinite(tau_per_w)].mean() if np.any(np.isfinite(tau_per_w)) else float("nan")
        print(f"  L{L:<2d}  {initial.altitude[L]:>7.1f}  "
              f"{kt_J:>12.3e}  {kb_J:>12.3e}  {ratio:>8.3f}  {tau_w_avg:>9.3f}  {n_reactant_per_lev[L]:>10.2e}")

    # ----- Per-species, per-wavelength dtau dump -----
    # dtau_k(z, lambda) = sigma_k(lambda) * 0.5*(n_k(z) + n_k(z+1)) * dz_layer
    # The total column tau above level L is sum_{i=L}^{top-1} dtau_layer(i)
    # plus a scale-height top contribution.
    opacity_dict = target_term.parameters.get("total_cross_section_by_species", {})
    active_nlyr_full = int(target_term.parameters.get("radiation_active_nlyr") or
                           titan_state.state.nlyr)
    active_nlyr = min(active_nlyr_full, titan_state.state.nlyr)
    alt_km = initial.altitude
    print(f"\n=== Per-species column tau dump (active_nlyr={active_nlyr}) ===")
    print(f"  (cumulative tau from top of radiation, summed over wavelength)")
    species_taus: dict[str, np.ndarray] = {}
    for sp_name, sigma_list in opacity_dict.items():
        if sp_name not in species_index or not isinstance(sigma_list, list):
            continue
        sp_idx = species_index[sp_name]
        sigma_sp = np.array(sigma_list)  # (nwave,)
        n_sp = titan_state.concentration[0, :, sp_idx].cpu().numpy()  # (nlyr,)
        # Build cumulative column above each level (top → bottom)
        col_lambda = np.zeros((active_nlyr, len(sigma_sp)))
        # Top-layer scale-height contribution
        if active_nlyr >= 2:
            c0 = n_sp[active_nlyr - 2]
            c1 = n_sp[active_nlyr - 1]
            dz_top_km = alt_km[active_nlyr - 1] - alt_km[active_nlyr - 2]
            if c0 > 0 and c1 > 0 and c0 != c1:
                H_km = abs(dz_top_km / np.log(c1 / c0))
            else:
                H_km = 10.0
            col_lambda[active_nlyr - 1] = H_km * 1e5 * c1 * sigma_sp
        # Trapezoid going down
        for i in range(active_nlyr - 2, -1, -1):
            dz_km = alt_km[i + 1] - alt_km[i]
            ext_layer = 0.5 * (n_sp[i] + n_sp[i + 1]) * sigma_sp
            col_lambda[i] = col_lambda[i + 1] + ext_layer * dz_km * 1e5
        species_taus[sp_name] = col_lambda.sum(axis=-1)  # collapse wavelengths

    # Show cumulative tau per species at L5, L10, L20, L35
    print(f"  {'species':>8s}  {'τ@L35':>9s}  {'τ@L20':>9s}  {'τ@L10':>9s}  {'τ@L5':>9s}")
    for sp_name, taus in species_taus.items():
        row = f"  {sp_name:>8s}"
        for L in [35, 20, 10, 5]:
            if L < len(taus):
                row += f"  {taus[L]:>9.2e}"
            else:
                row += f"  {'-':>9s}"
        print(row)

    # ----- Per-wavelength dtau dump at L5 and L20 -----
    for audit_L in [20, 5]:
        if audit_L >= active_nlyr:
            continue
        print(f"\n=== Per-species per-wavelength tau breakdown at L{audit_L} "
              f"(alt={alt_km[audit_L]:.1f} km) ===")
        # Identify wavelengths that contribute meaningfully to this reaction's J
        contribs_L = sigma.cpu().numpy() * af[audit_L]
        wave_idx_sorted = np.argsort(-contribs_L)[:8]
        # For each contributing wavelength, show per-species cumulative tau and total
        print(f"  Top 8 contributing wavelengths (by J at L{audit_L}):")
        header = f"    {'wl(A)':>7s}  {'F_z/F_top':>9s}"
        for sp_name in opacity_dict.keys():
            if sp_name in species_index:
                header += f"  {sp_name:>9s}"
        header += f"  {'sum_tau':>9s}"
        print(header)
        for w_i in wave_idx_sorted:
            if contribs_L[w_i] <= 0:
                continue
            row = f"    {wavelengths[w_i]:>7.1f}  {af[audit_L, w_i]/max(f_top[w_i], 1):>9.3f}"
            total_tau = 0.0
            for sp_name, sigma_list in opacity_dict.items():
                if sp_name not in species_index:
                    continue
                sp_idx = species_index[sp_name]
                sigma_w = sigma_list[w_i]
                n_sp = titan_state.concentration[0, :, sp_idx].cpu().numpy()
                tau_sp_w = 0.0
                if active_nlyr >= 2 and audit_L < active_nlyr - 1:
                    # top-layer scale height contribution
                    c0 = n_sp[active_nlyr - 2]
                    c1 = n_sp[active_nlyr - 1]
                    dz_top_km = alt_km[active_nlyr - 1] - alt_km[active_nlyr - 2]
                    if c0 > 0 and c1 > 0 and c0 != c1:
                        H_km = abs(dz_top_km / np.log(c1 / c0))
                    else:
                        H_km = 10.0
                    tau_sp_w += H_km * 1e5 * c1 * sigma_w
                    # layers from active_nlyr-2 down to audit_L
                    for i in range(active_nlyr - 2, audit_L - 1, -1):
                        dz_km = alt_km[i + 1] - alt_km[i]
                        ext_layer = 0.5 * (n_sp[i] + n_sp[i + 1]) * sigma_w
                        tau_sp_w += ext_layer * dz_km * 1e5
                total_tau += tau_sp_w
                row += f"  {tau_sp_w:>9.2e}"
            row += f"  {total_tau:>9.2e}"
            print(row)

    # ----- Snapshot at one level: per-wavelength comparison -----
    audit_level = int(__import__("os").environ.get("AUDIT_LEVEL", "5"))
    print(f"\n=== Per-wavelength audit at L{audit_level} (alt={initial.altitude[audit_level]:.1f} km) ===")
    print(f"  Sum(sigma*F_top) [s^-1, no atten] = {(sigma.cpu().numpy() * f_top).sum():.3e}")
    print(f"  Sum(sigma*actinic) [s^-1, kintera attenuated] = {j_kt_per_lev[audit_level]:.3e}")
    print(f"  KB J at this level = {kb_rate_per_lev[audit_level] / max(n_reactant_per_lev[audit_level], 1e-30):.3e}")
    print()
    print(f"  Top {{n_wave with biggest contribution to j_kt at this level}}:")
    contribs = sigma.cpu().numpy() * af[audit_level]  # per-wavelength contribution to J(z)
    order = np.argsort(-contribs)
    wavelengths_arr = np.array(wavelengths)
    cross_arr = sigma.cpu().numpy()
    print(f"    {'wl(A)':>7s}  {'sigma':>10s}  {'F_top':>10s}  {'F_z':>10s}  {'F_z/F_top':>9s}  {'contribJ':>10s}")
    for i in order[:12]:
        if contribs[i] <= 0:
            break
        print(f"    {wavelengths_arr[i]:>7.1f}  {cross_arr[i]:>10.3e}  {f_top[i]:>10.3e}  "
              f"{af[audit_level, i]:>10.3e}  {af[audit_level, i]/max(f_top[i], 1):>9.3f}  {contribs[i]:>10.3e}")


if __name__ == "__main__":
    pun_id = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(pun_id)
