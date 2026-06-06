"""Stage 5 (photo): confirm core Photolysis reproduces the validated Titan
attenuated photolysis rate J(z) = Σ_λ σ_r(λ) F_att(z,λ) for the active moses00
photo reactions, fed the SAME attenuated actinic flux the baseline computes.

The baseline (`_photo_rate_profile`) computes J = (F_att * σ_r).sum(-1). Core
`Photolysis.forward` with unit `quadrature_weights` computes the identical sum,
so feeding the same F_att must reproduce J(z) to machine precision. This closes
the photo-routing question (the flux field itself stays a Titan input by design).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

import kintera as kt

torch.set_default_dtype(torch.float64)
sys.path.insert(0, str(Path(__file__).parent))
import stage5_core_thermal_check as s5  # noqa: E402
import moses00_match as mm  # noqa: E402

from kintera.kinetics_base.titan.models import KBTitanState  # noqa: E402
from kintera.kinetics_base.titan.source_integration import _photo_rate_profile  # noqa: E402
from kintera.kinetics_base.titan.radiation import _kinetics_base_pyharp_actinic_flux  # noqa: E402


def main():
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
    photo = [t for t in sterms
             if t.kind == "pun_photo_rate_reaction"
             and all(r in species_set for r in (t.reactants or []) + (t.products or []))]
    for t in photo:
        t.parameters["freeze_actinic_flux"] = False
    ref = kt.parse_kinetics_base_atmosphere(str(mm.REF_PATH))
    ts.state.concentration = mm.inject_reference_state(ts, ref)
    ts.concentration = ts.state.concentration

    species_index = {n: i for i, n in enumerate(ts.species)}
    conc = ts.state.concentration
    dz = ts.state.dx1f.view(1, -1)
    T = ts.state.temperature

    print(f"active photo terms (single-reactant, in subset): "
          f"{sum(1 for t in photo if len(t.reactants)==1)}")

    worst = 0.0
    checked = 0
    skipped_secondary = 0
    for t in photo:
        if len(t.reactants) != 1:
            continue
        reactant = species_index.get(t.reactants[0])
        if reactant is None:
            continue
        params = t.parameters
        wl = params.get("wavelengths"); sig = params.get("cross_section"); flux = params.get("flux")
        if not (isinstance(wl, list) and isinstance(sig, list) and isinstance(flux, list) and wl):
            continue
        if isinstance(params.get("secondary_impact"), dict):
            skipped_secondary += 1
            continue

        # baseline attenuated J(z)
        Jbase = _photo_rate_profile(t, _state_for(ts, ts.state), conc, species_index, reactant, dz)
        if Jbase is None:
            continue

        # extract the SAME attenuated flux F_att(z, lambda)
        flux_tensor = torch.tensor(flux)
        Fatt = _kinetics_base_pyharp_actinic_flux(
            t, _state_for(ts, ts.state), conc, species_index, flux_tensor, len(sig),
            conc.dtype, conc.device)
        if Fatt is None:
            continue

        # build a 1-reaction core Photolysis: branch0=absorb(parent), branch1=dissoc(products)
        opt = kt.PhotolysisOptions()
        eq = f"{t.reactants[0]} => " + " + ".join(t.products)
        opt.reactions([kt.Reaction(eq)])
        opt.wavelength(list(wl))
        xs = []
        for w in range(len(sig)):
            xs.append(sig[w])   # branch0 (absorb, unused in J)
            xs.append(sig[w])   # branch1 (dissociation -> J)
        opt.cross_section(xs); opt.cross_section_nslabs([1])
        absorb = {t.reactants[0]: 1.0}
        diss = {}
        for p in t.products:
            diss[p] = diss.get(p, 0.0) + 1.0
        opt.branches([[absorb, diss]]); opt.branch_names([["a", "b"]])
        opt.quadrature_weights([1.0] * len(sig))
        ph = kt.Photolysis(opt)
        ph.update_xs_diss_stacked(T)
        # Fatt shape (ncol, nlyr, nwave) -> core wants (nwave, ncol, nlyr)
        Fcore = Fatt.movedim(-1, 0)
        Jcore = ph.forward(T, Fcore)[..., 0]  # (ncol, nlyr)

        denom = Jbase.abs().clamp_min(1e-300)
        rel = ((Jcore - Jbase).abs() / denom).max().item()
        worst = max(worst, rel)
        checked += 1

    print(f"checked {checked} photo reactions (skipped {skipped_secondary} secondary-impact)")
    print(f"core Photolysis J(z) vs baseline _photo_rate_profile: max rel diff = {worst:.3e}")
    return 0


def _state_for(ts, state):
    return KBTitanState(
        species=ts.species, fixed_species=ts.fixed_species,
        varying_species=ts.varying_species, conversion=ts.conversion,
        concentration=ts.concentration, density=ts.density, kzz=ts.kzz, state=state)


if __name__ == "__main__":
    raise SystemExit(main())
