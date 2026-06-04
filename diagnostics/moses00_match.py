"""Stage 2: match kintera against the paper-era moses00 neutral network.

60 floating species + 26 fixed species + M (87 total). NPHOTO=20 (real N2,
H2O, CO, CO2, HCN photolysis active). Includes nitrogen chemistry (HCN,
HC3N, CH3CN, NH, ...) and methanol/H2O/CO oxygen chemistry that the
Stage 1 moses05.C2 toy didn't have.

See [[project-titan-simplify-workflow]] for the recipe this follows.

Compared to Stage 1 (moses05.C2 / 15 species):
- N2 is in the fixed species list (id=67), so N2 opacity is native and we
  don't need the wrapper-to-inject-density hack.
- 20 photolysis reactions vs 6.
- 87 species vs 16, so the chemistry network is much richer (more sinks
  for naked-C chain that Stage 1 was missing).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

import kintera as kt

torch.set_default_dtype(torch.float64)

EX_DIR = Path(
    "/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare/examples/titan_moses00"
)
PUN_PATH = EX_DIR / "kindata" / "kindata.titan.moses00.pun"
ATM_PATH = EX_DIR / "atm" / "atm.titan.moses00.kt.inp"
BC_PATH = EX_DIR / "boundary" / "boundary.moses00.inp"
RUN_INPUT = EX_DIR / "case" / "kinetics.inp"
REF_PATH = EX_DIR / "case" / "fort.7.kt"  # KB converged SS (fort.50 was a per-step dump)

# Cheng provides photolysis cross-sections + catalog; moses00 paper uses
# its own flux2atmos.inp (Ly-α differs from Cheng's flare flux).
CHENG_DIR = Path(
    "/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare/examples/titan"
)
PHOTO_CATALOG = CHENG_DIR / "Cheng_catalog_v4.dat"
CROSS_DIR = CHENG_DIR / "Cheng_cross"
FLUX_PATH = EX_DIR / "flux2atmos.inp"

# moses00 kinetics.inp IVARYF (60 floating species, PUN ids):
#   001 003-045 068-083
FLOATING_IDS = [1] + list(range(3, 46)) + list(range(68, 84))
# IFIX (26 fixed): 067 002 046-066 084-086. M is PUN id=87 in moses00
# (id=150 in moses05.C2 — these are network-specific). Add M so the 13
# 3-body reactions (CH3+CH3+M->C2H6+M etc.) survive the subset filter.
FIXED_IDS = [67, 2] + list(range(46, 67)) + list(range(84, 87)) + [87]

# moses00 fort.50/fort.7 stores ALL species as mixing ratios (unlike
# moses05.C2 which had H/H2/aN2 as concentrations). Verified: H@L60=5.52e-4
# in fort.50, * density 2e+11 = 1.1e+8 cm⁻³ — physical. M and JDUST appear
# in KB output with placeholder values (M=1.0, JDUST=0.0); skip them so
# the atm shim's M=density / JDUST=density build-time values survive.
CONCENTRATION_SPECIES = set()
SKIP_INJECT = {"M", "JDUST"}  # JDUST: fort.7 has aerosol MR; kintera's σ=1e-8 cm² inflates
# this to τ≈2800. moses00 has no aerosol chemistry, so force JDUST=0 (atm-shim zero_profile).


def species_names(pun, ids):
    sp_by_id = {s.id: s.name for s in pun.species}
    return [sp_by_id[i] for i in ids if i in sp_by_id]


def inject_reference_state(ts, ref):
    c = ts.concentration.clone()
    density = ts.density[0]
    for name in ts.species:
        i = ts.species.index(name)
        if name in SKIP_INJECT:
            continue
        if name not in ref.species_profiles:
            continue
        prof = torch.as_tensor(
            np.array(ref.species_profiles[name]), dtype=c.dtype, device=c.device
        )
        if prof.numel() != c.shape[1]:
            continue
        if name in CONCENTRATION_SPECIES:
            c[0, :, i] = prof
        else:
            c[0, :, i] = prof * density
    return c


def main() -> int:
    print("[setup] loading moses00 PUN catalog")
    pun = kt.parse_kinetics_base_pun(str(PUN_PATH))
    print(f"  PUN: {len(pun.species)} species, {len(pun.reactions)} reactions")
    floating = species_names(pun, FLOATING_IDS)
    fixed = species_names(pun, FIXED_IDS)
    species = floating + fixed
    print(f"  using {len(species)} species ({len(floating)} floating + {len(fixed)} fixed)")
    print(f"  fixed: {fixed}")

    print()
    print("[setup] building Titan state from atm + boundary + PUN")
    raw_atm = kt.parse_kinetics_base_atmosphere(str(ATM_PATH))
    # Missing fixed species (mostly O/H2O/CO/CO2 chemistry placeholders that
    # moses00 lists in IFIX but doesn't actually populate). Inject zero
    # profiles for any species in the subset that the atm file lacks. We
    # also inject N2 = total density if it's missing (some atms don't
    # have %N2 even when N2 is the bath gas).
    missing = [s for s in species if s not in raw_atm.species_profiles]
    zero_profile = [0.0] * len(raw_atm.altitude)
    # JDUST: moses00 atm file has real aerosol loading but kintera's per-unit
    # σ=1e-8 cm² makes that opaque. moses00 has no aerosol chemistry, force JDUST=0.
    density_like = ("N2", "M")
    force_zero = ("JDUST",)
    class _AtmShim:
        altitude = list(raw_atm.altitude)
        density = list(raw_atm.density)
        temperature = list(raw_atm.temperature)
        pressure = list(raw_atm.pressure)
        eddy_diffusion = list(raw_atm.eddy_diffusion)
        species_profiles = dict(raw_atm.species_profiles)
        for s in missing:
            species_profiles[s] = list(raw_atm.density) if s in density_like else list(zero_profile)
        for s in force_zero:
            if s in species_profiles:
                species_profiles[s] = list(zero_profile)
    if missing:
        print(f"  patched {len(missing)} missing species into atm: {missing[:5]}{'...' if len(missing)>5 else ''}")
    print(f"  forced JDUST=0 in atm shim")
    atm_for_state = _AtmShim()
    ts = kt.build_kinetics_base_titan_state(
        atm_for_state,
        species=species,
        fixed_species=fixed,
        boundary_path=str(BC_PATH),
        pun_path=str(PUN_PATH),
    )
    print(f"  state: ncol={ts.state.ncol}, nlyr={ts.state.nlyr}, nspecies={ts.state.nspecies}")
    print(f"  density L0={ts.density[0,0].item():.2e}, L40={ts.density[0,40].item():.2e}, "
          f"L80={ts.density[0,80].item():.2e}")

    print()
    print("[setup] building source terms (catalog photolysis, no special path)")
    sterms = kt.build_kinetics_base_titan_source_terms(
        str(PUN_PATH),
        boundary_path=str(BC_PATH),
        run_input_path=str(RUN_INPUT),
        photo_catalog_path=str(PHOTO_CATALOG),
        cross_dir=str(CROSS_DIR),
        flux_path=str(FLUX_PATH),
    )
    species_set = set(species)
    filtered = [
        t for t in sterms
        if all(r in species_set for r in (t.reactants or []) + (t.products or []))
    ]
    print(f"  raw source terms: {len(sterms)} → filtered to subset: {len(filtered)}")
    n_photo = sum(1 for t in filtered if t.kind == "pun_photo_rate_reaction")
    print(f"    photo source terms: {n_photo}")

    # Unfreeze actinic flux (KB sets freeze=True via RAD=0; that uses RAW
    # mixing-ratio values for opacity which gives τ≈0).
    for t in filtered:
        t.parameters["freeze_actinic_flux"] = False

    # Pin fixed species. moses00 has 26 fixed species, many initialised to
    # 0 (HE, O, O2, OH, H2O, CO, CO2 — the O-chemistry that the paper
    # didn't actually populate). Without pinning, BE steps let them drift
    # (H2O grew from 0 to 7e+16 at L20 in initial test — Newton-step
    # auto-catalysis from any tiny coupling). Pin them to their initial
    # (fort.50-injected) values.
    def _apply_pins(new_conc):
        return kt.apply_kinetics_base_titan_boundary_pins(new_conc, ts)
    def _apply_dirichlet(system, rhs):
        return kt.apply_kinetics_base_titan_dirichlet_rows(system, rhs, ts)

    # Load reference fort.50
    ref = kt.parse_kinetics_base_atmosphere(str(REF_PATH))
    print()
    print(f"[reference] fort.50.kt.v2: {len(ref.species_profiles)} species, "
          f"{len(ref.altitude)} levels")

    print()
    print("[inject] overwriting concentration with fort.50 reference")
    ts.state.concentration = inject_reference_state(ts, ref)
    ts.concentration = ts.state.concentration

    print("[atm2d] building atm2d source terms")
    # Production default: chemistry through the core engine (CoreChemistrySource).
    # Set KINTERA_TITAN_HANDROLLED_CHEM=1 to fall back to the hand-rolled path.
    if os.environ.get("KINTERA_TITAN_HANDROLLED_CHEM"):
        atm_sources = kt.build_kinetics_base_titan_atm2d_source_terms(ts, filtered)
        print(f"  atm2d sources (hand-rolled chemistry): {len(atm_sources)}")
    else:
        atm_sources = kt.build_kinetics_base_titan_core_source_terms(
            ts, str(PUN_PATH), filtered)
        print(f"  atm2d sources (core-engine chemistry): {len(atm_sources)}")

    # Frozen tendency check
    print()
    print("[tendency] computing dc/dt at fort.50 reference state")
    lin = kt.build_source_linearization(ts.state, atm_sources)
    tend = lin.tendency
    c = ts.state.concentration
    nontrivial = c.abs() > 1e-20
    rel = torch.zeros_like(tend)
    rel[nontrivial] = tend[nontrivial].abs() / c[nontrivial].abs()
    physical = c > 1e+0
    print(f"  median |chem|/|c|: {rel[nontrivial].median().item():.2e}")
    print(f"  median |chem|/|c| (physical cells): {rel[physical].median().item():.2e}")

    # Push to SS
    print()
    NT = int(os.environ.get("KINTERA_TITAN_NTIME", "120"))
    DT_MAX = float(os.environ.get("KINTERA_TITAN_MAX_DT", "1.0e+7"))
    DT_MIN = 1.0e-3
    schedule = np.geomspace(DT_MIN, DT_MAX, NT)
    total_t = schedule.sum()
    print(f"[BE integration to SS]: NT={NT}, dt {DT_MIN:.0e}..{DT_MAX:.0e}, "
          f"total = {total_t:.2e} s ({total_t/86400/365.25:.2f} yr)")
    # Molecular diffusion for matching KB's transport of heavy species
    # and gravitational separation of H2/CH4. The moses00 paper binary
    # ships an overlay `COEFF1.f90` that computes
    #   DIFF = ADIFH2 * T^(SDIFH2-1) * (7.3439e+21/N) * sqrt(2.016/M)
    # and then UNCONDITIONALLY overwrites DIFF with the Cheng-2013
    # formula:
    #   DIFF = 7.3e+16 * T^0.75 / N * sqrt((1+28/M)/(1+28/16))
    # (see paper/moses00/COEFF1.f90:60-63 — no ifdef on the second
    # assignment). So the right physical match is Cheng-2013, even
    # though kinetics.inp ships ADIFH2/SDIFH2 inputs.
    # The "moses" option is preserved for diagnostic comparison.
    from kintera.kinetics_base.titan.transport_diffusion import (
        kinetics_base_titan_cheng_diffusion,
        kinetics_base_titan_moses_diffusion,
        kinetics_base_titan_species_masses,
    )
    moldiff_kind = os.environ.get("KINTERA_TITAN_MOLDIFF", "cheng").lower()
    pun_meta = {sp.name: sp for sp in pun.species}
    moldiff_masses = kinetics_base_titan_species_masses(list(ts.species), pun_meta)
    if moldiff_kind == "moses":
        binary_diffusion = kinetics_base_titan_moses_diffusion(
            ts.state, moldiff_masses, density=ts.density,
        )
        print(f"  moldiff: moses-2005 (ADIFH2=3.81e-5, SDIFH2=1.74)")
    elif moldiff_kind == "cheng":
        binary_diffusion = kinetics_base_titan_cheng_diffusion(
            ts.state, moldiff_masses, density=ts.density,
        )
        print(f"  moldiff: Cheng-2013 (7.3e+16, T^0.75)")
    elif moldiff_kind in ("off", "none", "0"):
        binary_diffusion = None
        moldiff_masses = None
        print(f"  moldiff: OFF")
    else:
        raise ValueError(f"Unknown KINTERA_TITAN_MOLDIFF={moldiff_kind!r}")

    transport_form_env = os.environ.get("KINTERA_TITAN_TRANSPORT", "mr_diffusion").lower()
    if transport_form_env not in ("mr_diffusion", "mr_exp", "mr_hybrid"):
        raise ValueError(f"Unknown KINTERA_TITAN_TRANSPORT={transport_form_env!r}")
    print(f"  transport_form: {transport_form_env}")

    c_start = c.clone()
    c_current = c_start.clone()
    for step, dt in enumerate(schedule):
        ts.state.concentration = c_current
        sys_mat, rhs = kt.build_implicit_step_system(
            ts.state, ts.kzz, float(dt),
            density=ts.density, transport_form=transport_form_env,
            source_terms=atm_sources,
            binary_diffusion=binary_diffusion,
            molecular_weights=moldiff_masses,
        )
        sys_mat, rhs = _apply_dirichlet(sys_mat, rhs)
        c_current = kt.solve_sparse_system(sys_mat, rhs)
        c_current = torch.clamp(c_current, min=0.0)
        c_current = _apply_pins(c_current)
        if (step + 1) % 10 == 0 or step == NT - 1:
            ts.state.concentration = c_current
            tend_cur = kt.build_source_linearization(ts.state, atm_sources).tendency
            tr_cur = kt.build_eddy_diffusion_matrix(
                ts.state, ts.kzz, form="mr_diffusion", density=ts.density,
            ).matvec(c_current)
            total_cur = tend_cur + tr_cur
            phys = c_current > 1e+0
            if phys.any():
                med = (total_cur.abs()[phys] / c_current.abs()[phys]).median().item()
                mx = (total_cur.abs()[phys] / c_current.abs()[phys]).max().item()
                print(f"  step {step+1:3d} dt={dt:.2e}: median |Δ|/|c| = {med:.2e}, max = {mx:.2e}")
    c_after = c_current

    # Compare SS to fort.50 at key layers, for a subset of species
    print()
    REPORT_SPECIES = ["H", "H2", "CH4", "C2H2", "C2H4", "C2H6", "CH3", "C2H5",
                      "C2H", "C2", "C", "CH",
                      "HCN", "HC3N", "CH3CN", "C2N2", "N", "CN", "NH",
                      "H2O", "CO", "CO2", "CH3OH"]
    print(f"{'Species':<10}  fort.50 (KB)              kintera SS              ratio kt/KB")
    print(f"{'':<10}  {'L20':>11} {'L40':>11} {'L60':>11}  "
          f"{'L20':>11} {'L40':>11} {'L60':>11}  {'L20':>6} {'L40':>6} {'L60':>6}")
    for sp in REPORT_SPECIES:
        if sp not in species:
            continue
        i = species.index(sp)
        b20, b40, b60 = c[0, 20, i].item(), c[0, 40, i].item(), c[0, 60, i].item()
        a20, a40, a60 = c_after[0, 20, i].item(), c_after[0, 40, i].item(), c_after[0, 60, i].item()
        r20 = a20 / b20 if abs(b20) > 1e-20 else float('nan')
        r40 = a40 / b40 if abs(b40) > 1e-20 else float('nan')
        r60 = a60 / b60 if abs(b60) > 1e-20 else float('nan')
        print(f"{sp:<10}  {b20:>11.2e} {b40:>11.2e} {b60:>11.2e}  "
              f"{a20:>11.2e} {a40:>11.2e} {a60:>11.2e}  "
              f"{r20:>6.2f} {r40:>6.2f} {r60:>6.2f}")

    np.savez("/tmp/kt_moses00_ss.npz",
             species=np.array(species, dtype=object),
             altitude=ts.state.x1v.numpy() / 1e5,
             density=ts.density[0].numpy(),
             temperature=ts.state.temperature[0].numpy(),
             c_fort50=c[0].numpy(),
             c_kintera_ss=c_after[0].numpy())
    print(f"\nSaved kintera SS to /tmp/kt_moses00_ss.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
