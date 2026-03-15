#!/usr/bin/env python3
"""Extract comparison data from VULCAN .vul pickle files.

Provides functions to load atmosphere, initial conditions, final state,
and photolysis data from VULCAN output files, with unit conversions
for kintera compatibility.

Usage as a script:
    python extract_vulcan_data.py VULCAN/output/CHO-kintera.vul --summary

Usage as a module:
    from tools.extract_vulcan_data import load_vulcan_output
    vul = load_vulcan_output("VULCAN/output/CHO-kintera.vul")
"""

import argparse
import os
import pickle
import sys

import numpy as np

AVO = 6.02214076e23
RGAS = 8.314462


def load_vulcan_output(filepath):
    """Load a VULCAN .vul file and return a structured dict.

    Returns dict with keys:
        species: list of species names (VULCAN ordering)
        nz: number of vertical layers
        nspecies: number of species

        # Atmosphere (CGS)
        P_cgs: pressure at layer centers (dyne/cm²), shape (nz,)
        T_K: temperature (K), shape (nz,)
        Kzz_cgs: eddy diffusion (cm²/s), shape (nz-1,)
        dzi_cgs: interface layer spacing (cm), shape (nz-1,)
        dz_cgs: layer thickness (cm), shape (nz,)
        M_cgs: total number density (molecule/cm³), shape (nz,)

        # Atmosphere (SI for kintera)
        P_Pa: pressure (Pa), shape (nz,)
        n_molm3: total molar density (mol/m³), shape (nz,)
        Kzz_si: eddy diffusion (m²/s), shape (nz-1,)
        dzi_si: interface spacing (m), shape (nz-1,)

        # Concentrations
        y_ini_cgs: initial number densities (molecule/cm³), shape (nz, nspecies)
        y_cgs: final number densities (molecule/cm³), shape (nz, nspecies)
        ymix: final mixing ratios, shape (nz, nspecies)

        # Photolysis (if present)
        bins: wavelength bins (nm)
        sflux: stellar flux at interfaces, shape (nz+1, nwav)
        aflux: actinic flux at centers, shape (nz, nwav)
        cross: dict of {species: absorption cross-section array}
        cross_J: dict of {(species, branch): branching cross-section}
        n_branch: dict of {species: num_branches}
    """
    with open(filepath, "rb") as f:
        raw = pickle.load(f)

    v = raw["variable"]
    a = raw["atm"]

    species = list(v["species"])
    nz = len(a["pco"])
    ns = len(species)

    P_cgs = np.array(a["pco"], dtype=np.float64)
    T_K = np.array(a["Tco"], dtype=np.float64)
    Kzz_cgs = np.array(a["Kzz"], dtype=np.float64)
    dzi_cgs = np.array(a["dzi"], dtype=np.float64)
    dz_cgs = np.array(a["dz"], dtype=np.float64)
    M_cgs = np.array(a["M"], dtype=np.float64)

    P_Pa = P_cgs * 0.1
    n_molm3 = P_Pa / (RGAS * T_K)
    Kzz_si = Kzz_cgs * 1e-4
    dzi_si = dzi_cgs * 0.01

    y_ini_cgs = np.array(v["y_ini"], dtype=np.float64)
    y_cgs = np.array(v["y"], dtype=np.float64)
    ymix = np.array(v["ymix"], dtype=np.float64)

    result = {
        "species": species, "nz": nz, "nspecies": ns,
        "P_cgs": P_cgs, "T_K": T_K, "Kzz_cgs": Kzz_cgs,
        "dzi_cgs": dzi_cgs, "dz_cgs": dz_cgs, "M_cgs": M_cgs,
        "P_Pa": P_Pa, "n_molm3": n_molm3,
        "Kzz_si": Kzz_si, "dzi_si": dzi_si,
        "y_ini_cgs": y_ini_cgs, "y_cgs": y_cgs, "ymix": ymix,
    }

    if "bins" in v:
        result["bins"] = np.array(v["bins"])
    if "sflux" in v:
        result["sflux"] = np.array(v["sflux"])
    if "aflux" in v:
        result["aflux"] = np.array(v["aflux"])
    if "cross" in v:
        result["cross"] = v["cross"]
    if "cross_J" in v:
        result["cross_J"] = v["cross_J"]
    if "n_branch" in v:
        result["n_branch"] = v["n_branch"]

    return result


def vulcan_to_kintera_ic(vul, kintera_species):
    """Convert VULCAN initial conditions to kintera concentration array.

    Args:
        vul: dict from load_vulcan_output()
        kintera_species: list of kintera species names

    Returns:
        C: concentration array (nz, len(kintera_species)) in mol/m³
    """
    nz = vul["nz"]
    ns = len(kintera_species)
    vul_idx = {sp: i for i, sp in enumerate(vul["species"])}

    C = np.zeros((nz, ns))
    for i, sp in enumerate(kintera_species):
        if sp in vul_idx:
            C[:, i] = np.maximum(
                vul["y_ini_cgs"][:, vul_idx[sp]] * 1e6 / AVO, 0.0)
    return C


def main():
    parser = argparse.ArgumentParser(
        description="Extract data from VULCAN .vul files")
    parser.add_argument("vul_file", help="VULCAN .vul output file")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of contents")
    args = parser.parse_args()

    vul = load_vulcan_output(args.vul_file)

    if args.summary:
        print(f"File: {args.vul_file}")
        print(f"  Layers: {vul['nz']}")
        print(f"  Species: {vul['nspecies']}")
        print(f"  P range: {vul['P_cgs'].min():.2e} - "
              f"{vul['P_cgs'].max():.2e} dyne/cm²")
        print(f"  T range: {vul['T_K'].min():.1f} - "
              f"{vul['T_K'].max():.1f} K")
        print(f"  Species list: {vul['species']}")
        print(f"\n  Major species (bottom layer, by mixing ratio):")
        bot_mix = vul["ymix"][0]
        order = np.argsort(bot_mix)[::-1]
        for i in order[:10]:
            print(f"    {vul['species'][i]:10s}: {bot_mix[i]:.4e}")
        if "bins" in vul:
            print(f"\n  Photolysis: {len(vul['bins'])} wavelength bins")
        if "n_branch" in vul:
            print(f"  Photo species: {list(vul['n_branch'].keys())}")


if __name__ == "__main__":
    main()
