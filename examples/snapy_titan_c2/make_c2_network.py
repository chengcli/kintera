#!/usr/bin/env python3
"""Generate the Titan C2 photochemistry network for the snapy GCM case.

Runs under the DEV kintera environment (/opt/anaconda3/bin/python3.10, full
KINETICS-base stack). It translates the validated moses05.C2 15-species
subnetwork (57 thermal reactions + 13 photolysis branches) into artifacts the
OLD pip kintera (snapy's pyenv, no kinetics_base package) can consume:

- ``titan_c2_chem.yaml``  -- kintera-native reactions (arrhenius + falloff
  with synthetic Troe Fc=0.6, which reproduces KB's broadening exactly).
- ``titan_c2_data.npz``   -- photolysis branch cross-sections + TOA flux,
  absorber total cross-sections (for the RT driver), species/molecular
  weights, the moses05 atmosphere profiles (for ICs), and the parameter
  table of the 11 KB UPDATE_CHEMB custom-rate reactions.
- ``c2_ref_rates.npz``    -- dev-side reference rate constants/rates/J for
  the validation gate.

After writing, it invokes ``validate_c2_network.py`` under ``~/pyenv/bin/python``
(Gate A): per-reaction rates from the generated YAML + the vendored custom
layer must match the dev reference to <=1e-6 rel.

KB rate semantics for this subset (verified: all rate blocks have t0=1.0 and
a single block): k = A * T^b * exp(C/T); falloff (D>0): k0=(A,b,C) cm^6,
kinf=(D,E,F) cm^3, F = 0.6^(1/(1+log10(Pr)^2)).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import kintera as kt
from kintera.kinetics_base.titan.chemb_overrides import (
    has_titan_chemb_override_by_signature,
    titan_chemb_rate_constant_by_signature,
)
from kintera.kinetics_base.titan.physics import _pun_rate_constant

REPO = Path(__file__).resolve().parents[2]
EX = REPO / "diagnostics/KINETICS-base-compare/examples/titan_moses05_c2"
CHENG = REPO / "diagnostics/KINETICS-base-compare/examples/titan"
OUT = Path(__file__).resolve().parent
PYENV = os.path.expanduser("~/pyenv/bin/python")

C2_SPECIES = ["H", "H2", "C", "CH", "(1)CH2", "(3)CH2", "CH3", "CH4",
              "C2", "C2H", "C2H2", "C2H3", "C2H4", "C2H5", "C2H6"]
SANITIZE = {"(1)CH2": "CH2_1", "(3)CH2": "CH2_3"}
# storage order for the GCM scalar state: bath gas first, then the C2 set
STORAGE_SPECIES = ["N2"] + [SANITIZE.get(s, s) for s in C2_SPECIES]
PHOTO_PLACEHOLDER_IDS = {845, 846, 847, 848, 849, 870}  # IPHOTO parents (X->X)

# cv_R only matters for thermo bookkeeping (chemistry here is isothermal)
CV_R = {"N2": 2.5, "H": 1.5, "H2": 2.5, "C": 1.5, "CH": 2.5, "CH2_1": 3.0,
        "CH2_3": 3.0, "CH3": 3.0, "CH4": 3.0, "C2": 2.5, "C2H": 3.0,
        "C2H2": 3.0, "C2H3": 3.0, "C2H4": 3.0, "C2H5": 3.0, "C2H6": 3.0}
COMPOSITION = {"N2": {"N": 2}, "H": {"H": 1}, "H2": {"H": 2}, "C": {"C": 1},
               "CH": {"C": 1, "H": 1}, "CH2_1": {"C": 1, "H": 2},
               "CH2_3": {"C": 1, "H": 2}, "CH3": {"C": 1, "H": 3},
               "CH4": {"C": 1, "H": 4}, "C2": {"C": 2},
               "C2H": {"C": 2, "H": 1}, "C2H2": {"C": 2, "H": 2},
               "C2H3": {"C": 2, "H": 3}, "C2H4": {"C": 2, "H": 4},
               "C2H5": {"C": 2, "H": 5}, "C2H6": {"C": 2, "H": 6}}


def sanitize(name: str) -> str:
    return SANITIZE.get(name, name)


def equation(reactants: list[str], products: list[str], *, drop_m: bool) -> str:
    """Build an old-kintera Reaction equation (irreversible, no M)."""
    def side(names):
        out = []
        for n in sorted(set(names), key=names.index):
            if drop_m and n == "M":
                continue
            c = names.count(n)
            out.append(f"{c} {sanitize(n)}" if c > 1 else sanitize(n))
        return " + ".join(out)
    return f"{side(reactants)} => {side(products)}"


def load_c2_thermal(pun):
    """Return (plain, falloff, custom) lists of reaction records."""
    byid = {s.id: s.name for s in pun.species}
    ok = set(C2_SPECIES) | {"M"}
    plain, falloff, custom = [], [], []
    for r in pun.reactions:
        if r.id in PHOTO_PLACEHOLDER_IDS or not r.rate_blocks:
            continue
        b = r.rate_blocks[0]
        if b.A == 0.0:
            continue
        reac = [byid[i] for i in r.reactant_ids]
        prod = [byid[i] for i in r.product_ids]
        if not (reac and all(x in ok for x in reac + prod)):
            continue
        assert len(r.rate_blocks) == 1 and b.Tmin == 1.0, (
            f"rxn {r.id}: unexpected multi-block/t0 (t0={b.Tmin})")
        rec = dict(id=r.id, reactants=reac, products=prod,
                   A=b.A, b=b.b, C=b.C, D=b.D, E=b.E, F=b.F)
        if has_titan_chemb_override_by_signature(reac, prod):
            custom.append(rec)
        elif b.D > 0.0:
            assert "M" in reac, f"rxn {r.id}: falloff without M"
            falloff.append(rec)
        else:
            assert "M" not in reac, f"rxn {r.id}: 3-body without falloff"
            plain.append(rec)
    return plain, falloff, custom


def write_chem_yaml(path: Path, plain, falloff):
    lines = ["# Titan C2 photochemistry -- thermal reactions (moses05.C2 subset)",
             "# GENERATED by make_c2_network.py; validated by validate_c2_network.py.",
             "# KB-native A units (molecule, cm, s); Tref=1 reproduces KB's A*T^b*exp(C/T).",
             "",
             "reference-state:",
             "  Tref: 1.0",
             "  Pref: 1.e5",
             "",
             "species:"]
    for s in STORAGE_SPECIES:
        comp = ", ".join(f"{k}: {v}" for k, v in COMPOSITION[s].items())
        lines.append(f"  - name: {s}")
        lines.append(f"    composition: {{{comp}}}")
        lines.append(f"    cv_R: {CV_R[s]}")
    lines.append("")
    lines.append("reactions:")
    # The installed (old) kintera evaluates k = A*(T/300)^b*exp(-Ea_R/T): the
    # yaml 'reference-state: Tref' is parsed but NOT propagated into the rate
    # sub-options (their compiled default Tref=300 K wins). KB semantics are
    # k = A_kb*T^b*exp(C/T), so we fold the 300^b factor into A here:
    # A_yaml = A_kb * 300^b  ->  A_yaml*(T/300)^b == A_kb*T^b exactly.
    TREF = 300.0
    def fold(a, b):
        return a * TREF ** b
    for r in plain:
        lines.append(f"  - equation: {equation(r['reactants'], r['products'], drop_m=True)}")
        lines.append("    type: arrhenius")
        lines.append(f"    rate-constant: {{A: {fold(r['A'], r['b']):.10e}, b: {r['b']:.6g}, Ea_R: {-r['C']:.6g}}}")
        lines.append(f"    # pun id {r['id']} (KB A = {r['A']:.4e}, Tref=300 folded)")
    for r in falloff:
        lines.append(f"  - equation: {equation(r['reactants'], r['products'], drop_m=True)}")
        lines.append("    type: falloff")
        lines.append(f"    low-P-rate-constant: {{A: {fold(r['A'], r['b']):.10e}, b: {r['b']:.6g}, Ea_R: {-r['C']:.6g}}}")
        lines.append(f"    high-P-rate-constant: {{A: {fold(r['D'], r['E']):.10e}, b: {r['E']:.6g}, Ea_R: {-r['F']:.6g}}}")
        # synthetic Troe params -> F_cent == 0.6 exactly == KB broadening
        lines.append("    Troe: {A: 0.6, T3: 1.0e-10, T1: 1.0e30, T2: 0.}")
        lines.append("    efficiencies: {N2: 1.0}")
        lines.append(f"    # pun id {r['id']} (KB A = {r['A']:.4e}, D = {r['D']:.4e}, Tref=300 folded)")
    path.write_text("\n".join(lines) + "\n")
    return path


def build_photo_terms():
    terms = kt.build_kinetics_base_titan_source_terms(
        str(EX / "kindata/kindata.titan.moses05.pun"),
        boundary_path=str(EX / "boundary/boundary.moses05.C2.inp"),
        run_input_path=str(EX / "case/kinetics.inp"),
        photo_catalog_path=str(CHENG / "Cheng_catalog_v4.dat"),
        cross_dir=str(CHENG / "Cheng_cross"),
        flux_path=str(EX / "flux2atmos.inp"),
    )
    ok = set(C2_SPECIES)
    branches = [
        t for t in terms
        if t.kind == "pun_photo_rate_reaction"
        and t.reaction_id not in PHOTO_PLACEHOLDER_IDS
        and len(t.reactants) == 1 and t.reactants[0] in ok
        and all(p in ok for p in t.products)
        and t.products != t.reactants
        and isinstance(t.parameters.get("cross_section"), list)
    ]
    return branches


def build_npz(path: Path, branches, custom, pun):
    wl = np.asarray(branches[0].parameters["wavelengths"], dtype=np.float64)
    flux = np.asarray(branches[0].parameters["flux"], dtype=np.float64)
    nw = wl.size
    sig = np.zeros((len(branches), nw))
    parents, prods = [], []
    for i, t in enumerate(branches):
        assert list(t.parameters["wavelengths"]) == list(wl)
        assert list(t.parameters["flux"]) == list(flux)
        sig[i] = np.asarray(t.parameters["cross_section"], dtype=np.float64)
        parents.append(sanitize(t.reactants[0]))
        prods.append("|".join(sanitize(p) for p in t.products))
    # absorber total cross-sections for the RT optical depth
    totxs = branches[0].parameters["total_cross_section_by_species"]
    abs_species = sorted(totxs.keys())
    abs_sigma = np.stack([np.asarray(totxs[s], dtype=np.float64) for s in abs_species])

    # the pun molecular_weight field is not a molar mass (it under-counts
    # heavy atoms); build mw from the species composition instead.
    ATOMIC = {"C": 12.011e-3, "H": 1.008e-3, "N": 14.0067e-3}  # kg/mol
    mw_arr = np.array([
        sum(ATOMIC[el] * n for el, n in COMPOSITION[s].items())
        for s in STORAGE_SPECIES])

    # moses05 atmosphere (91 levels) for initial conditions. Most species
    # profiles are mixing ratios, but some (H, H2 in this file) are stored as
    # number densities [cm^-3] -- detect by magnitude and normalize.
    atm = kt.parse_kinetics_base_atmosphere(str(EX / "atm/atm.titan.moses05.kt.inp"))
    dens_prof = np.asarray(atm.density, dtype=np.float64)
    prof_names, prof_vmr = [], []
    for s in C2_SPECIES:
        if s in atm.species_profiles:
            prof = np.asarray(atm.species_profiles[s], dtype=np.float64)
            if prof.max() > 1.0:                      # density-like row
                prof = prof / np.maximum(dens_prof, 1.0)
            prof_names.append(sanitize(s))
            prof_vmr.append(prof)

    cz = [dict(id=r["id"],
               reactants="|".join(sanitize(x) for x in r["reactants"] if x != "M"),
               products="|".join(sanitize(x) for x in r["products"] if x != "M"))
          for r in custom]
    np.savez(
        path,
        storage_species=np.array(STORAGE_SPECIES),
        mw=mw_arr,
        wavelengths=wl,
        toa_flux=flux,
        photo_parents=np.array(parents),
        photo_products=np.array(prods),
        photo_sigma=sig,
        photo_ids=np.array([t.reaction_id for t in branches]),
        abs_species=np.array([sanitize(s) for s in abs_species]),
        abs_sigma=abs_sigma,
        custom_ids=np.array([c["id"] for c in cz]),
        custom_reactants=np.array([c["reactants"] for c in cz]),
        custom_products=np.array([c["products"] for c in cz]),
        atm_altitude_km=np.asarray(atm.altitude, dtype=np.float64),
        atm_temperature=np.asarray(atm.temperature, dtype=np.float64),
        atm_pressure=np.asarray(atm.pressure, dtype=np.float64),
        atm_density=np.asarray(atm.density, dtype=np.float64),
        atm_profile_species=np.array(prof_names),
        atm_profile_vmr=np.stack(prof_vmr) if prof_vmr else np.zeros((0, len(atm.altitude))),
    )
    return path


def build_reference(path: Path, plain, falloff, custom, branches):
    """Dev-side reference rates on the 91-level moses05 (T, density) profile."""
    import torch
    torch.set_default_dtype(torch.float64)
    atm = kt.parse_kinetics_base_atmosphere(str(EX / "atm/atm.titan.moses05.kt.inp"))
    T = torch.tensor(atm.temperature)
    dens = torch.tensor(atm.density)  # cm^-3
    # synthetic uniform test mixing: every species at 1 ppm of density ->
    # nonzero rate for every reaction; the gate tests rate constants, not SS.
    # N2 absorbs the remainder so SUM(conc) == density EXACTLY: the runtime
    # computes [M] as the sum over storage species, and the gate must compare
    # like for like at 1e-6 tolerance.
    test_vmr = 1.0e-6
    conc = {sanitize(s): test_vmr * dens for s in C2_SPECIES}
    conc["N2"] = dens * (1.0 - len(C2_SPECIES) * test_vmr)

    def mass_action(k, reac):
        rate = k.clone()
        for x in reac:
            if x == "M":
                rate = rate * dens
            else:
                rate = rate * conc[sanitize(x)]
        return rate

    recs = []
    for r in plain + falloff:
        params = dict(A=r["A"], b=r["b"], C=r["C"], D=r["D"], E=r["E"],
                      F=r["F"], Tmin=1.0)
        k = _pun_rate_constant(params, T, dens)  # falloff: per-M 3-body k
        rate = mass_action(k, r["reactants"])
        recs.append((r["id"], equation(r["reactants"], r["products"], drop_m=True),
                     "plain" if r in plain else "falloff", rate.numpy()))
    for r in custom:
        # chemb k for M-reactions is per-M 3-body (cm^6/s); the [M] factor is
        # applied by mass_action via the M in the reactant list. 2-body chemb
        # reactions (no M) return cm^3/s directly.
        k = titan_chemb_rate_constant_by_signature(
            r["reactants"], r["products"], T, dens)
        rate = mass_action(k, r["reactants"])
        recs.append((r["id"], equation(r["reactants"], r["products"], drop_m=True),
                     "custom", rate.numpy()))

    # unattenuated photolysis J (per-bin sigma*F sum, KB convention)
    jtop = []
    for t in branches:
        s = np.asarray(t.parameters["cross_section"])
        f = np.asarray(t.parameters["flux"])
        jtop.append(float(np.sum(s * f)))

    np.savez(
        path,
        ids=np.array([x[0] for x in recs]),
        equations=np.array([x[1] for x in recs]),
        kinds=np.array([x[2] for x in recs]),
        rates=np.stack([x[3] for x in recs]),       # molecule cm^-3 s^-1
        temperature=atm.temperature,
        density=atm.density,
        test_vmr=test_vmr,
        photo_ids=np.array([t.reaction_id for t in branches]),
        photo_jtop=np.array(jtop),
    )
    return path


def main() -> int:
    pun = kt.parse_kinetics_base_pun(str(EX / "kindata/kindata.titan.moses05.pun"))
    plain, falloff, custom = load_c2_thermal(pun)
    print(f"C2 thermal routing: {len(plain)} arrhenius, {len(falloff)} falloff, "
          f"{len(custom)} custom (chemb)")
    for r in falloff:
        print(f"  falloff: {r['id']} {equation(r['reactants'], r['products'], drop_m=True)}")
    for r in custom:
        print(f"  custom : {r['id']} {equation(r['reactants'], r['products'], drop_m=True)}")

    branches = build_photo_terms()
    print(f"photolysis branches: {len(branches)} "
          f"(parents: {sorted({t.reactants[0] for t in branches})})")

    ypath = write_chem_yaml(OUT / "titan_c2_chem.yaml", plain, falloff)
    npath = build_npz(OUT / "titan_c2_data.npz", branches, custom, pun)
    rpath = build_reference(OUT / "c2_ref_rates.npz", plain, falloff, custom, branches)
    print(f"wrote {ypath.name}, {npath.name}, {rpath.name}")

    print("\n[gate] validating under", PYENV)
    # VIRTUAL_ENV lets pyenv-kintera's resource finder locate nasa9.dat
    # (normally set by `source ~/pyenv/bin/activate`).
    env = dict(os.environ, VIRTUAL_ENV=os.path.expanduser("~/pyenv"))
    rc = subprocess.run(
        [PYENV, str(OUT / "validate_c2_network.py"),
         "--yaml", str(ypath), "--data", str(npath), "--ref", str(rpath)],
        cwd=str(OUT), env=env).returncode
    print("GATE A:", "PASS" if rc == 0 else f"FAIL (rc={rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
