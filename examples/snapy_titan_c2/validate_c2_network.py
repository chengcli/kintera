#!/usr/bin/env python3
"""Gate A: validate the generated C2 network under snapy's pyenv kintera.

Runs under ``~/pyenv/bin/python`` (old pip kintera -- the runtime that the
snapy GCM case uses). It evaluates the EXACT runtime code path
(:class:`titan_c2_chem.TitanC2Chemistry`) on the moses05 91-level Titan
profile and compares per-reaction rates against the dev-side reference
written by ``make_c2_network.py``:

- Gate A1: every thermal reaction rate matches <= 1e-6 rel at every level.
- Gate A2: total thermal tendency dC/dt per species matches <= 1e-8 rel.
- Gate A3: unattenuated top-of-atmosphere photolysis J matches <= 1e-6 rel.

Exit code 0 = all gates pass.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from titan_c2_chem import (  # noqa: E402
    CM3_TO_SI, SI_TO_CM3, TitanC2Chemistry, normalize_equation,
)

torch.set_default_dtype(torch.float64)
KB_BOLTZ_SI = 1.380649e-23


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="titan_c2_chem.yaml")
    ap.add_argument("--data", default="titan_c2_data.npz")
    ap.add_argument("--ref", default="c2_ref_rates.npz")
    args = ap.parse_args()

    ref = np.load(args.ref, allow_pickle=False)
    chem = TitanC2Chemistry(args.yaml, args.data, device="cpu")

    T = torch.tensor(ref["temperature"])
    dens_cgs = torch.tensor(ref["density"])           # molecule/cm^3
    test_vmr = float(ref["test_vmr"])
    nlev = T.numel()

    # rebuild the reference test state in storage order (SI mol/m^3);
    # N2 absorbs the remainder so sum(conc) == density exactly.
    conc_cgs = torch.zeros(nlev, chem.nsp)
    for i, s in enumerate(chem.species):
        conc_cgs[:, i] = (dens_cgs * (1.0 - 15 * test_vmr) if s == "N2"
                          else test_vmr * dens_cgs)
    conc = conc_cgs * CM3_TO_SI
    pres = (dens_cgs * 1.0e6) * KB_BOLTZ_SI * T       # Pa (n k T)

    rate, _ = chem.rates(T, pres, conc, jrate=None)
    rate_cgs = rate * SI_TO_CM3                        # molecule/cm^3/s

    # map runtime reaction columns -> reference rows by normalized equation
    runtime_eqs = ([normalize_equation(e) for e in chem.kin_equations]
                   + [normalize_equation(
                       " + ".join(c["reactants"]) + " => " + " + ".join(c["products"]))
                      for c in chem.custom])
    ref_eqs = [normalize_equation(str(e)) for e in ref["equations"]]
    ref_rates = ref["rates"]                           # (nref, nlev) cgs
    ref_by_eq = {}
    for i, e in enumerate(ref_eqs):
        assert e not in ref_by_eq, f"duplicate reference equation {ref['equations'][i]}"
        ref_by_eq[e] = i

    n_thermal = chem.nrxn_kin + chem.nrxn_custom
    assert len(runtime_eqs) == n_thermal
    fails = []
    worst = 0.0
    print(f"Gate A1: {n_thermal} thermal reactions x {nlev} levels")
    matched = set()
    for col, e in enumerate(runtime_eqs):
        if e not in ref_by_eq:
            fails.append((col, "NO REFERENCE MATCH", float("nan")))
            continue
        i = ref_by_eq[e]
        matched.add(i)
        r_run = rate_cgs[:, col].numpy()
        r_ref = ref_rates[i]
        floor = max(np.abs(r_ref).max() * 1e-30, 1e-300)
        rel = np.abs(r_run - r_ref) / np.maximum(np.abs(r_ref), floor)
        m = float(rel.max())
        worst = max(worst, m)
        tag = "kin" if col < chem.nrxn_kin else "custom"
        status = "ok" if m <= 1e-6 else "FAIL"
        if m > 1e-6:
            fails.append((col, str(ref["equations"][i]), m))
        print(f"  [{status}] {tag:6} id={ref['ids'][i]:>4} "
              f"{str(ref['equations'][i]):42} max_rel={m:.2e}")
    unmatched = set(range(len(ref_eqs))) - matched
    for i in unmatched:
        fails.append((-1, f"reference rxn not in runtime: {ref['equations'][i]}", float("nan")))

    # Gate A2: total thermal tendency
    stoich_t = chem.stoich[:, :n_thermal]
    tend_run = (stoich_t @ rate[:, :n_thermal].T).T * SI_TO_CM3  # (nlev, nsp) cgs
    tend_ref = np.zeros_like(tend_run.numpy())
    for i, e in enumerate(ref_eqs):
        # reconstruct stoich from the reference equation signature
        (lhs, rhs) = e
        for name, cnt in lhs:
            tend_ref[:, chem.sp[name]] -= cnt * ref_rates[i]
        for name, cnt in rhs:
            tend_ref[:, chem.sp[name]] += cnt * ref_rates[i]
    scale = np.abs(tend_ref).max(axis=0)
    rel2 = np.abs(tend_run.numpy() - tend_ref) / np.maximum(scale, 1e-300)
    a2 = float(rel2.max())
    print(f"Gate A2: max species-tendency rel diff = {a2:.2e} "
          f"({'ok' if a2 <= 1e-8 else 'FAIL'})")

    # Gate A3: unattenuated TOA photolysis J
    jtop_run = chem.photolysis_rates(chem.toa_flux.unsqueeze(0))[0].numpy()
    jtop_ref = ref["photo_jtop"]
    rel3 = np.abs(jtop_run - jtop_ref) / np.maximum(np.abs(jtop_ref), 1e-300)
    a3 = float(rel3.max())
    print(f"Gate A3: {len(jtop_ref)} photolysis branches, "
          f"max J rel diff = {a3:.2e} ({'ok' if a3 <= 1e-6 else 'FAIL'})")
    for p, j in zip(chem.photo_parents, jtop_run):
        print(f"    J_top({p}) = {j:.3e} 1/s")

    ok = (not fails) and a2 <= 1e-8 and a3 <= 1e-6
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print("  ", f)
    print(f"\nGATE A: {'PASS' if ok else 'FAIL'} "
          f"(A1 worst={worst:.2e}, A2={a2:.2e}, A3={a3:.2e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
