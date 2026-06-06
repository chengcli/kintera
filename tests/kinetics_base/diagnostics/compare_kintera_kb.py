"""Compare kintera final state against KB oracle and the initial atmosphere.

Inputs:
  - ``KINTERA_DUMP``: ``.npz`` produced by ``no_grain_stability.py`` with
    ``KINTERA_TITAN_DUMP``. Holds ``species``, ``altitude_km``, ``density``,
    ``concentration`` (nlyr, nspecies), and metadata.
  - ``KB_REF``: a KINETICS-base ``.pun`` output file. Format: most species
    are number densities (cm^-3) but a few special "tracer" species (U,
    JDUST, RAYEAR) are written as mixing-ratio-like values. The conversion
    heuristic here mirrors kintera's own ``_is_kinetics_base_concentration_profile``.
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

import kintera as kt

DEFAULT_ROOT = pathlib.Path(__file__).resolve().parent / "KINETICS-base-compare"
ROOT = pathlib.Path(os.environ.get("KINTERA_KINETICS_BASE_ROOT", DEFAULT_ROOT))
TITAN_DIR = ROOT / "examples" / "titan"

INITIAL_PATH = str(TITAN_DIR / "kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz")
PUN_PATH = str(TITAN_DIR / "kindata_yy_clean" / "Cheng_ions_c6h7+_v3_H2CN.pun")
KB_REF = os.environ.get(
    "KB_REF",
    str(
        ROOT.parent
        / "kinetics_base_oracle_runs"
        / "after_ch4_special_fix_50"
        / "titan-50-steps"
        / "kintitan.out.pun"
    ),
)
KINTERA_DUMP = os.environ.get("KINTERA_DUMP", "/tmp/kintera_200.npz")

# Levels to show in altitude tables.
LEVELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]


def _load_pun(path: str, density: np.ndarray | None = None, *, format: str = "initial") -> dict:
    """Load a .pun file with explicit unit convention.

    Parameters
    ----------
    format:
        - ``"initial"``: most species are mixing ratios; charged species,
          U/JDUST/RAYEAR/PROD/SGA, star-suffix species are written as
          per-molecule concentrations. Mirrors kintera's heuristic in
          ``_is_kinetics_base_concentration_profile``.
        - ``"kb_output"``: KB's ``kintitan.out.pun`` writes ALL species as
          number density (cm^-3) regardless of type — no conversion.
    """
    parsed = kt.parse_kinetics_base_atmosphere(path)
    species = list(parsed.species_profiles.keys())
    nlyr = len(parsed.altitude)
    nsp = len(species)
    conc = np.zeros((nlyr, nsp), dtype=np.float64)
    rho = np.asarray(parsed.density, dtype=np.float64) if density is None else density

    if format == "kb_output":
        for j, name in enumerate(species):
            conc[:, j] = np.asarray(parsed.species_profiles[name], dtype=np.float64)
    else:
        pun_meta = kt.kinetics_base_species_metadata_from_pun(PUN_PATH)
        fixed = {"JDUST", "N2", "PROD", "U", "RAYEAR", "SGA", "M"}
        for j, name in enumerate(species):
            col = np.asarray(parsed.species_profiles[name], dtype=np.float64)
            meta = pun_meta.get(name)
            is_density = False
            if name.endswith("*"):
                is_density = True
            elif name == "E" or name.endswith("+") or name.endswith("-"):
                is_density = True
            elif meta is not None and not any(v != 0 for v in meta.composition):
                is_density = True
            elif (
                meta is not None
                and name in fixed
                and meta.molecular_weight <= 0.0
            ):
                is_density = True
            if is_density:
                conc[:, j] = col
            else:
                conc[:, j] = col * rho
    return {
        "species": species,
        "altitude_km": np.asarray(parsed.altitude, dtype=np.float64),
        "density": rho,
        "concentration": conc,
    }


def _profile_line(L, alt, init_v, kt_v, kb_v):
    def _fmt(x):
        if x == 0:
            return "       0"
        return f"{x:9.2e}"
    ratio_ik = kt_v / init_v if init_v > 0 else float("nan")
    ratio_kk = kt_v / kb_v if kb_v > 0 else float("nan")
    return (
        f"  {L:>3d} {alt:7.1f}  {_fmt(init_v):>10s}  {_fmt(kt_v):>10s}  "
        f"{_fmt(kb_v):>10s}  {ratio_ik:7.3f}  {ratio_kk:7.3f}"
    )


def _print_profile(species_name, idx, alt, init, kt, kb):
    print(f"\n  {species_name} (idx={idx})")
    print(f"  {'lev':>3s} {'alt(km)':>7s}  {'init':>10s}  {'kintera':>10s}  "
          f"{'KB(50)':>10s}  {'kt/init':>7s}  {'kt/KB':>7s}")
    print("  " + "-" * 70)
    for L in LEVELS:
        if L >= len(alt):
            continue
        print(_profile_line(L, alt[L], init[L, idx], kt[L, idx], kb[L, idx]))


def main():
    if not os.path.exists(KINTERA_DUMP):
        print(f"ERROR: kintera dump not found at {KINTERA_DUMP}")
        return 1
    dump = np.load(KINTERA_DUMP, allow_pickle=True)
    kt_species = [str(x) for x in dump["species"]]
    kt_conc = np.asarray(dump["concentration"], dtype=np.float64)
    kt_alt = np.asarray(dump["altitude_km"], dtype=np.float64)
    kt_density = np.asarray(dump["density"], dtype=np.float64)
    kt_ntime = int(dump["ntime"])
    kt_total_t = float(dump["total_simulated_time"])

    init = _load_pun(INITIAL_PATH, density=kt_density, format="initial")
    kb = _load_pun(KB_REF, density=kt_density, format="kb_output")

    print(f"[kintera] NTIME={kt_ntime} total_simulated={kt_total_t:.3e} s, no_grain mode")
    print(f"[KB]      50-step oracle, FULL network (FREEZE=1 SUBLIM=-1)")
    print(f"          ⚠ kintera is no_grain but KB oracle is full — grain/ion comparisons are not apples-to-apples")
    print(f"          ✓ neutral non-grain species are directly comparable")

    assert kt_species == init["species"] == kb["species"]
    assert np.allclose(kt_alt, init["altitude_km"])

    species_idx = {s: i for i, s in enumerate(kt_species)}

    # 1) Major neutral species — should match KB
    print()
    print("=" * 80)
    print("MAJOR NEUTRAL SPECIES (no_grain-relevant — should match KB)")
    print("=" * 80)
    for name in ["N2", "M", "CH4", "JDUST"]:
        if name in species_idx:
            _print_profile(name, species_idx[name], kt_alt,
                          init["concentration"], kt_conc, kb["concentration"])

    # 2) Important neutral byproducts (H, H2, etc.) — kintera should produce some
    print()
    print("=" * 80)
    print("NEUTRAL CHEMISTRY PRODUCTS (kintera should match general magnitude)")
    print("=" * 80)
    for name in ["H", "H2", "C2H2", "C2H6", "HCN", "CH3", "N(2D)"]:
        if name in species_idx:
            _print_profile(name, species_idx[name], kt_alt,
                          init["concentration"], kt_conc, kb["concentration"])

    # 3) Ion species — KB has them populated (quasi-steady-state?), kintera explicitly time-steps
    print()
    print("=" * 80)
    print("ION SPECIES (KB likely uses quasi-steady-state; kintera time-steps)")
    print("=" * 80)
    for name in ["E", "N2+", "CH4+", "CH5+", "C2H5+", "H+", "H2+"]:
        if name in species_idx:
            _print_profile(name, species_idx[name], kt_alt,
                          init["concentration"], kt_conc, kb["concentration"])

    # 4) Grain species — kintera explicitly excludes via no_grain filter
    print()
    print("=" * 80)
    print("GRAIN/MANTLE SPECIES (kintera no_grain excludes — KB has them)")
    print("=" * 80)
    for name in ["SGA", "U", "GCH4", "GC2H6", "GH"]:
        if name in species_idx:
            _print_profile(name, species_idx[name], kt_alt,
                          init["concentration"], kt_conc, kb["concentration"])

    # 5) Global summary: how many species kintera matches KB at each level
    print()
    print("=" * 80)
    print("PER-LEVEL AGREEMENT SUMMARY")
    print("=" * 80)
    print(f"  {'lev':>3s} {'alt(km)':>7s}  {'#matched':>10s}  {'#dev(>10x)':>12s}  {'#dev(>1000x)':>14s}  {'top_offender'}")
    print("  " + "-" * 80)
    for L in LEVELS:
        if L >= len(kt_alt):
            continue
        kt_v = kt_conc[L, :]
        kb_v = kb["concentration"][L, :]
        valid = (np.maximum(kt_v, kb_v) > 1.0)
        if not valid.any():
            continue
        ratios = np.where(valid, np.maximum(kt_v, 1e-30) / np.maximum(kb_v, 1e-30), 1.0)
        log_dev = np.abs(np.log10(np.maximum(ratios, 1e-30)))
        matched = int(((log_dev < 0.1) & valid).sum())
        dev10 = int(((log_dev > 1.0) & valid).sum())
        dev1000 = int(((log_dev > 3.0) & valid).sum())
        if valid.any() and dev10 > 0:
            log_dev_masked = np.where(valid, log_dev, 0.0)
            worst = int(np.argmax(log_dev_masked))
            offender = f"{kt_species[worst]} (kt={kt_v[worst]:.2e}, KB={kb_v[worst]:.2e})"
        else:
            offender = "-"
        print(f"  {L:>3d} {kt_alt[L]:7.1f}  {matched:>10d}  {dev10:>12d}  {dev1000:>14d}  {offender}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
