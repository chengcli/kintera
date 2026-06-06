"""Stage 5.1/5.2: 1-step BE-solve check — swap CoreChemistrySource into
build_implicit_step_system and confirm the implicit step matches the
hand-rolled baseline.

Baseline source terms = chemistry (pun_thermal + pun_photo) + boundary BCs.
Core source terms = [CoreChemistrySource] + the same boundary BCs.
A single BE step from fort.50 with each must produce the same concentration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

import kintera as kt

torch.set_default_dtype(torch.float64)
sys.path.insert(0, str(Path(__file__).parent))
import moses00_match as mm  # noqa: E402
import stage5_core_full_check as F  # noqa: E402

from kintera.kinetics_base.titan.core_source import CoreChemistrySource  # noqa: E402
from kintera.kinetics_base.titan.transport_diffusion import (  # noqa: E402
    kinetics_base_titan_cheng_diffusion,
    kinetics_base_titan_species_masses,
)

BOUNDARY_KINDS = ("upper_boundary_flux", "upper_boundary_velocity", "lower_boundary_flux")


def one_step(ts, source_terms, dt, binary_diffusion, masses):
    sys_mat, rhs = kt.build_implicit_step_system(
        ts.state, ts.kzz, float(dt), density=ts.density,
        transport_form="mr_diffusion", source_terms=source_terms,
        binary_diffusion=binary_diffusion, molecular_weights=masses)
    sys_mat, rhs = kt.apply_kinetics_base_titan_dirichlet_rows(sys_mat, rhs, ts)
    c = kt.solve_sparse_system(sys_mat, rhs)
    c = torch.clamp(c, min=0.0)
    return kt.apply_kinetics_base_titan_boundary_pins(c, ts)


def main():
    ts, filtered = F.build_full_state()
    pun = kt.parse_kinetics_base_pun(str(mm.PUN_PATH))
    pun_meta = {sp.name: sp for sp in pun.species}
    masses = kinetics_base_titan_species_masses(list(ts.species), pun_meta)
    bdiff = kinetics_base_titan_cheng_diffusion(ts.state, masses, density=ts.density)

    c0 = ts.state.concentration.clone()

    # baseline: full hand-rolled atm2d source terms (chemistry + boundary)
    base_atm = kt.build_kinetics_base_titan_atm2d_source_terms(ts, filtered)
    # core: CoreChemistrySource + the same boundary terms
    boundary = [t for t in filtered if t.kind in BOUNDARY_KINDS]
    boundary_atm = kt.build_kinetics_base_titan_atm2d_source_terms(ts, boundary)
    core_src = CoreChemistrySource(ts, str(mm.PUN_PATH), filtered)
    core_sources = [core_src] + list(boundary_atm)

    for dt in (1.0, 1.0e3, 1.0e6):
        ts.state.concentration = c0.clone()
        c_base = one_step(ts, base_atm, dt, bdiff, masses)
        ts.state.concentration = c0.clone()
        c_core = one_step(ts, core_sources, dt, bdiff, masses)
        rel = (c_core - c_base).abs() / c_base.abs().clamp_min(1e-300)
        phys = c_base > 1.0  # physically meaningful concentrations (cm^-3)
        print(f"dt={dt:.0e}: 1-step solve core vs baseline  "
              f"max rel(phys)={rel[phys].max().item():.2e}  "
              f"median={rel[phys].median().item():.2e}  "
              f"within1e-6={(rel[phys] < 1e-6).double().mean().item():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
