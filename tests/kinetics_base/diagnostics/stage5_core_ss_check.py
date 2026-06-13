"""Stage 5.5 GATE: run the full moses00 BE integration to steady state with
CoreChemistrySource and confirm the SS matches the hand-rolled baseline (and the
validated moses00 ratios vs the KB fort.7 reference).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

import kintera as kt

torch.set_default_dtype(torch.float64)
sys.path.insert(0, str(Path(__file__).parent))
import moses00_match as mm  # noqa: E402
import stage5_core_full_check as F  # noqa: E402
import stage5_core_step_check as S  # noqa: E402

from kintera.kinetics_base.titan.core_source import CoreChemistrySource  # noqa: E402

REPORT = ["H", "H2", "CH4", "C2H2", "C2H4", "C2H6", "CH3", "C2H5", "C2H",
          "HCN", "C6H6", "C3H3", "CH3C2H", "C4H2"]


def integrate(ts, sources, schedule, bdiff, masses, c0):
    c = c0.clone()
    for dt in schedule:
        ts.state.concentration = c
        smat, rhs = kt.build_implicit_step_system(
            ts.state, ts.kzz, float(dt), density=ts.density,
            transport_form="mr_diffusion", source_terms=sources,
            binary_diffusion=bdiff, molecular_weights=masses)
        smat, rhs = kt.apply_kinetics_base_titan_dirichlet_rows(smat, rhs, ts)
        c = torch.clamp(kt.solve_sparse_system(smat, rhs), min=0.0)
        c = kt.apply_kinetics_base_titan_boundary_pins(c, ts)
    return c


def main():
    ts, filtered = F.build_full_state()
    pun = kt.parse_kinetics_base_pun(str(mm.PUN_PATH))
    pm = {s.name: s for s in pun.species}
    masses = S.kinetics_base_titan_species_masses(list(ts.species), pm)
    bdiff = S.kinetics_base_titan_cheng_diffusion(ts.state, masses, density=ts.density)
    c0 = ts.state.concentration.clone()  # fort.50/fort.7 injected reference

    NT = int(os.environ.get("KINTERA_TITAN_NTIME", "120"))
    schedule = np.geomspace(1.0e-3, 1.0e7, NT)
    print(f"[integrate] NT={NT}, dt 1e-3..1e7")

    base_atm = kt.build_kinetics_base_titan_atm2d_source_terms(ts, filtered)
    boundary = [t for t in filtered if t.kind in S.BOUNDARY_KINDS]
    core_sources = [CoreChemistrySource(ts, str(mm.PUN_PATH), filtered)] + \
        list(kt.build_kinetics_base_titan_atm2d_source_terms(ts, boundary))

    cb = integrate(ts, base_atm, schedule, bdiff, masses, c0)
    cc = integrate(ts, core_sources, schedule, bdiff, masses, c0)

    sp = ts.species
    print("\n  core-vs-baseline SS agreement (physical cells, c>1):")
    phys = cb > 1.0
    rel = (cc - cb).abs() / cb.abs().clamp_min(1e-300)
    print(f"    max rel {rel[phys].max().item():.2e}  median {rel[phys].median().item():.2e}"
          f"  within1e-3 {(rel[phys] < 1e-3).double().mean().item():.4f}")

    print("\n  Species   L40: KB(c0)     base_SS     core_SS    core/KB  base/KB")
    for s in REPORT:
        if s not in sp:
            continue
        i = sp.index(s)
        kb = c0[0, 40, i].item(); b = cb[0, 40, i].item(); c = cc[0, 40, i].item()
        rk = c / kb if abs(kb) > 1e-20 else float('nan')
        rb = b / kb if abs(kb) > 1e-20 else float('nan')
        print(f"  {s:<9} {kb:>11.3e} {b:>11.3e} {c:>11.3e}  {rk:>7.3f} {rb:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
