"""Stage 5.1/5.2: validate the FULL core-engine chemistry tendency
(thermal mass-action + chemb overrides + photolysis) against the hand-rolled
baseline (pun_thermal_reaction + pun_photo_rate_reaction source terms) at the
moses00 fort.50 reference state.

Thermal: core Arrhenius+KBFalloff rate constants -> ChembOverrideLayer ->
frozen-k mass action. Photo: core Photolysis J(z) (fed the same attenuated flux
the baseline computes) assembled as J*parent*stoich with the baseline's
per-term multiplier / min-altitude / suppress-loss handling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

import kintera as kt

torch.set_default_dtype(torch.float64)
sys.path.insert(0, str(Path(__file__).parent))
import moses00_match as mm  # noqa: E402
import stage5_core_thermal_check as s5t  # noqa: E402

from kintera.kinetics_base.titan.models import KBTitanState  # noqa: E402
from kintera.kinetics_base.titan.source_integration import _photo_rate_profile  # noqa: E402
from kintera.kinetics_base.titan.radiation import _kinetics_base_pyharp_actinic_flux  # noqa: E402
from kintera.kinetics_base.titan.source_integration import (  # noqa: E402
    _rate_profile_multiplier_on_state_grid,
)


def _state_for(ts):
    return KBTitanState(
        species=ts.species, fixed_species=ts.fixed_species,
        varying_species=ts.varying_species, conversion=ts.conversion,
        concentration=ts.concentration, density=ts.density, kzz=ts.kzz, state=ts.state)


def build_full_state():
    pun = kt.parse_kinetics_base_pun(str(mm.PUN_PATH))
    floating = mm.species_names(pun, mm.FLOATING_IDS)
    fixed = mm.species_names(pun, mm.FIXED_IDS)
    species = floating + fixed
    species_set = set(species)
    raw_atm = kt.parse_kinetics_base_atmosphere(str(mm.ATM_PATH))
    missing = [sp for sp in species if sp not in raw_atm.species_profiles]
    zero = [0.0] * len(raw_atm.altitude)

    class _AtmShim:
        altitude = list(raw_atm.altitude); density = list(raw_atm.density)
        temperature = list(raw_atm.temperature); pressure = list(raw_atm.pressure)
        eddy_diffusion = list(raw_atm.eddy_diffusion)
        species_profiles = dict(raw_atm.species_profiles)
        for s in missing:
            species_profiles[s] = list(raw_atm.density) if s in ("N2", "M") else list(zero)
        if "JDUST" in species_profiles:
            species_profiles["JDUST"] = list(zero)

    ts = kt.build_kinetics_base_titan_state(
        _AtmShim(), species=species, fixed_species=fixed,
        boundary_path=str(mm.BC_PATH), pun_path=str(mm.PUN_PATH))
    sterms = kt.build_kinetics_base_titan_source_terms(
        str(mm.PUN_PATH), boundary_path=str(mm.BC_PATH), run_input_path=str(mm.RUN_INPUT),
        photo_catalog_path=str(mm.PHOTO_CATALOG), cross_dir=str(mm.CROSS_DIR),
        flux_path=str(mm.FLUX_PATH))
    filtered = [t for t in sterms
                if all(r in species_set for r in (t.reactants or []) + (t.products or []))]
    for t in filtered:
        if "freeze_actinic_flux" in t.parameters:
            t.parameters["freeze_actinic_flux"] = False
    ref = kt.parse_kinetics_base_atmosphere(str(mm.REF_PATH))
    ts.state.concentration = mm.inject_reference_state(ts, ref)
    ts.concentration = ts.state.concentration
    return ts, filtered


def baseline_full_tendency(ts, filtered):
    chem = [t for t in filtered
            if t.kind in ("pun_thermal_reaction", "pun_photo_rate_reaction")]
    atm = kt.build_kinetics_base_titan_atm2d_source_terms(ts, chem)
    lin = kt.build_source_linearization(ts.state, atm)
    return lin.tendency


def core_photo_tendency(ts, filtered):
    """Photo tendency via core Photolysis fed the baseline attenuated flux."""
    photo = [t for t in filtered if t.kind == "pun_photo_rate_reaction"
             and len(t.reactants) == 1]
    conc = ts.state.concentration
    T = ts.state.temperature
    dz = ts.state.dx1f.view(1, -1)
    sp_idx = {n: i for i, n in enumerate(ts.species)}
    tendency = torch.zeros_like(conc)
    tstate = _state_for(ts)
    for t in photo:
        reactant = sp_idx.get(t.reactants[0])
        if reactant is None:
            continue
        prod_coeff = {}
        miss = False
        for n in t.products:
            if n not in sp_idx:
                miss = True; break
            prod_coeff[sp_idx[n]] = prod_coeff.get(sp_idx[n], 0) + 1
        if miss:
            continue
        p = t.parameters
        wl, sig, flux = p.get("wavelengths"), p.get("cross_section"), p.get("flux")
        if not (isinstance(wl, list) and isinstance(sig, list) and isinstance(flux, list) and wl):
            continue
        if isinstance(p.get("secondary_impact"), dict):
            # ion channel; moses00 active set has none, skip defensively
            continue
        Fatt = _kinetics_base_pyharp_actinic_flux(
            t, tstate, conc, sp_idx, torch.tensor(flux), len(sig), conc.dtype, conc.device)
        if Fatt is None:
            continue
        opt = kt.PhotolysisOptions()
        opt.reactions([kt.Reaction(f"{t.reactants[0]} => " + " + ".join(t.products))])
        opt.wavelength(list(wl))
        xs = []
        for w in range(len(sig)):
            xs += [sig[w], sig[w]]
        opt.cross_section(xs); opt.cross_section_nslabs([1])
        diss = {}
        for pr in t.products:
            diss[pr] = diss.get(pr, 0.0) + 1.0
        opt.branches([[{t.reactants[0]: 1.0}, diss]]); opt.branch_names([["a", "b"]])
        opt.quadrature_weights([1.0] * len(sig))
        ph = kt.Photolysis(opt)
        ph.update_xs_diss_stacked(T)
        J = ph.forward(T, Fatt.movedim(-1, 0))[..., 0]  # (ncol, nlyr)

        # baseline per-term assembly: min_altitude, profile multiplier
        min_alt = p.get("min_altitude_km")
        if min_alt is not None:
            alt_km = ts.state.x1v.view(1, -1) / 1.0e5
            J = J * (alt_km >= float(min_alt))
        mult = _rate_profile_multiplier_on_state_grid(t, tstate, conc.dtype, conc.device)
        if mult is not None:
            J = J * mult
        parent = torch.clamp(conc[:, :, reactant], min=0.0)
        rate = J * parent
        if not bool(p.get("suppress_reactant_loss", False)):
            tendency[:, :, reactant] = tendency[:, :, reactant] - rate
        for prod, coeff in prod_coeff.items():
            tendency[:, :, prod] = tendency[:, :, prod] + coeff * rate
    return tendency


def main():
    ts, filtered = build_full_state()
    base = baseline_full_tendency(ts, filtered)
    therm, _ = s5t.core_thermal_tendency(ts)
    photo = core_photo_tendency(ts, filtered)
    core = therm + photo

    mask = base.abs() > 1e-25  # ignore trace float-noise cells
    denom = base.abs().clamp_min(1e-300)
    rel = (core - base).abs() / denom
    print(f"cells with |baseline| > 1e-25: {int(mask.sum())}")
    print(f"FULL chemistry (thermal+photo+chemb) core vs baseline:")
    print(f"  max rel diff:    {rel[mask].max().item():.3e}")
    print(f"  median rel diff: {rel[mask].median().item():.3e}")
    print(f"  fraction within 1e-6: {(rel[mask] < 1e-6).double().mean().item():.4f}")
    import numpy as np
    worst = torch.topk(rel[mask].flatten(), 5).values
    print(f"  top-5 rel (masked): {[f'{x:.2e}' for x in worst.tolist()]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
