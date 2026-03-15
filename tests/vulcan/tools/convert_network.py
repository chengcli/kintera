#!/usr/bin/env python3
"""Convert any VULCAN chemical network file to kintera YAML format.

Handles:
  - Two-body (Arrhenius) reactions
  - Three-body reactions (with and without high-pressure limit)
  - Lindemann falloff (three-body with k_inf)
  - Photolysis reactions with CSV cross-section embedding
  - Tref^b correction (VULCAN uses A*T^b, kintera uses A*(T/Tref)^b)
  - Branching ratios for multi-channel photolysis

Usage:
    python convert_network.py VULCAN/thermo/CHO_photo_network.txt \\
        --photo-cross-dir VULCAN/thermo/photo_cross \\
        --output cho_photo.yaml
"""

import argparse
import os
import re
import sys

import numpy as np
import yaml

TREF = 300.0
NUM_RE = re.compile(r"[-+]?\d+\.?\d*[eE][-+]?\d+|[-+]?\d+\.?\d*")


# ── YAML formatting helpers ──────────────────────────────────────────

class FlowDict(dict):
    pass

class FlowList(list):
    pass

class CustomDumper(yaml.SafeDumper):
    pass

def _flow_dict_repr(dumper, data):
    return dumper.represent_mapping(
        "tag:yaml.org,2002:map", data.items(), flow_style=True)

def _flow_list_repr(dumper, data):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True)

def _float_repr(dumper, value):
    if value != 0 and (abs(value) >= 1e5 or abs(value) < 0.01):
        s = f"{value:.6e}"
        if "e" in s:
            mantissa, exp = s.split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            if "." not in mantissa:
                mantissa += ".0"
            exp_sign = exp[0]
            exp_val = int(exp[1:])
            s = f"{mantissa}e{exp_sign}{exp_val:02d}"
    else:
        s = repr(value)
    return dumper.represent_scalar("tag:yaml.org,2002:float", s)

CustomDumper.add_representer(FlowDict, _flow_dict_repr)
CustomDumper.add_representer(FlowList, _flow_list_repr)
CustomDumper.add_representer(float, _float_repr)


# ── Parsing ──────────────────────────────────────────────────────────

def parse_formula(name):
    """Parse chemical formula into {element: count}, stripping excited-state
    suffixes like _1 (O_1), _2D (N_2D)."""
    formula = re.sub(r"_\w+$", "", name)
    comp = {}
    for m in re.finditer(r"([A-Z][a-z]?)(\d*)", formula):
        elem = m.group(1)
        count = int(m.group(2)) if m.group(2) else 1
        if elem:
            comp[elem] = comp.get(elem, 0) + count
    return comp


def parse_equation(eq_str):
    parts = eq_str.split("->")
    if len(parts) == 2:
        reactants = [s.strip() for s in parts[0].split("+") if s.strip()]
        products = [s.strip() for s in parts[1].split("+") if s.strip()]
    else:
        reactants = []
        products = [s.strip() for s in eq_str.split("+") if s.strip()]
    return reactants, products


def format_equation(reactants, products, reversible=False):
    arrow = " <=> " if reversible else " => "
    return " + ".join(reactants) + arrow + " + ".join(products)


def tref_correct(A, b):
    """Apply Tref^b correction: VULCAN A*T^b -> kintera A'*(T/Tref)^b
    where A' = A * Tref^b."""
    if b == 0.0:
        return A
    return A * (TREF ** b)


def parse_network(filepath):
    """Parse VULCAN network file into reaction lists."""
    two_body, three_body, photolysis = [], [], []
    section = "two_body"

    with open(filepath) as f:
        lines = f.readlines()

    for raw in lines:
        line = raw.strip()

        if "3-body and Disscoiation" in line or "3-body and Dissociation" in line:
            section = "three_body_kinf"
            continue
        if "3-body reactions without high-pressure" in line:
            section = "three_body_simple"
            continue
        if "# special cases" in line:
            section = "special"
            continue
        if "# reverse stops" in line or "# photo disscoiation" in line:
            section = "photolysis"
            continue

        if line.startswith("#") or not line or "[" not in line:
            continue

        id_m = re.match(r"(\d+)\s+\[", line)
        if not id_m:
            continue
        rxn_id = int(id_m.group(1))

        if rxn_id % 2 == 0:
            continue

        eq_m = re.search(r"\[\s*(.*?)\s*\]", line)
        if not eq_m:
            continue
        equation = eq_m.group(1).strip()
        after = line[eq_m.end():].strip()

        reactants, products = parse_equation(equation)
        if reactants == products:
            continue

        if section == "special":
            continue

        if section == "two_body":
            nums = NUM_RE.findall(after)
            if len(nums) < 3:
                continue
            two_body.append({
                "id": rxn_id, "reactants": reactants, "products": products,
                "A": float(nums[0]), "b": float(nums[1]),
                "Ea_R": float(nums[2]),
            })

        elif section in ("three_body_kinf", "three_body_simple"):
            if "M" in reactants and "M" not in products:
                products.append("M")

            nums = NUM_RE.findall(after)
            if section == "three_body_kinf" and len(nums) >= 6:
                entry = {
                    "id": rxn_id, "reactants": reactants,
                    "products": products,
                    "A0": float(nums[0]), "b0": float(nums[1]),
                    "Ea_R0": float(nums[2]),
                    "A_inf": float(nums[3]), "b_inf": float(nums[4]),
                    "Ea_R_inf": float(nums[5]),
                    "has_kinf": True,
                }
            elif len(nums) >= 3:
                entry = {
                    "id": rxn_id, "reactants": reactants,
                    "products": products,
                    "A0": float(nums[0]), "b0": float(nums[1]),
                    "Ea_R0": float(nums[2]),
                    "has_kinf": False,
                }
            else:
                continue
            three_body.append(entry)

        elif section == "photolysis":
            parts = after.split()
            if len(parts) < 2:
                continue
            parent = parts[0]
            try:
                br_idx = int(parts[1])
            except ValueError:
                continue
            if not reactants:
                reactants = [parent]
            photolysis.append({
                "id": rxn_id, "parent": parent, "br_index": br_idx,
                "reactants": reactants, "products": products,
            })

    return two_body, three_body, photolysis


def collect_species(two_body, three_body, photolysis, extra=None):
    species = set()
    for rxn in two_body + three_body:
        for sp in rxn["reactants"] + rxn["products"]:
            if sp != "M":
                species.add(sp)
    for rxn in photolysis:
        species.add(rxn["parent"])
        for sp in rxn["products"]:
            if sp != "M":
                species.add(sp)
    if extra:
        species.update(extra)
    return sorted(species)


# ── Cross-section handling ───────────────────────────────────────────

def load_cross_section_csv(filepath):
    """Load VULCAN CSV cross-section file.
    Returns (wavelength_nm, absorption_cm2, dissociation_cm2)."""
    data = np.loadtxt(filepath, delimiter=",", comments="#")
    wl = data[:, 0]
    absorption = data[:, 1]
    dissociation = data[:, 2] if data.shape[1] > 2 else absorption
    return wl, absorption, dissociation


def load_branch_ratios(filepath):
    """Load VULCAN branch ratio CSV.
    Returns (wavelength_nm, ratios_array) where ratios is (nwl, nbranch)."""
    data = np.loadtxt(filepath, delimiter=",", comments="#")
    wl = data[:, 0]
    ratios = data[:, 1:]
    return wl, ratios


def build_photo_cross_section(parent, br_index, photo_cross_dir):
    """Build YAML-embeddable cross-section data for a single photolysis branch.

    Returns list of [wavelength_nm, sigma_cm2] rows, or None if files missing.
    """
    cross_file = os.path.join(photo_cross_dir, parent, f"{parent}_cross.csv")
    branch_file = os.path.join(photo_cross_dir, parent, f"{parent}_branch.csv")

    if not os.path.isfile(cross_file):
        return None

    wl, _absorption, dissociation = load_cross_section_csv(cross_file)

    if os.path.isfile(branch_file):
        br_wl, br_ratios = load_branch_ratios(branch_file)
        n_branches = br_ratios.shape[1]
        if br_index <= n_branches:
            br_col = br_ratios[:, br_index - 1]
            br_interp = np.interp(wl, br_wl, br_col, left=br_col[0],
                                  right=br_col[-1])
        else:
            br_interp = np.ones_like(wl)
    else:
        br_interp = np.ones_like(wl)

    sigma = dissociation * br_interp
    sigma[sigma < 1e-200] = 0.0

    mask = sigma > 0
    if not mask.any():
        return [[float(wl[0]), 0.0]]

    rows = []
    for i in range(len(wl)):
        if sigma[i] > 0 or (i > 0 and sigma[i - 1] > 0) or \
           (i < len(wl) - 1 and sigma[i + 1] > 0):
            rows.append(FlowList([float(wl[i]), float(f"{sigma[i]:.6e}")]))

    return rows if rows else [[float(wl[0]), 0.0]]


# ── YAML construction ────────────────────────────────────────────────

def build_yaml(two_body, three_body, photolysis, species_list,
               photo_cross_dir):
    ref_state = {"Tref": TREF, "Pref": 1.0e5}

    species_entries = []
    for sp in species_list:
        comp = parse_formula(sp)
        species_entries.append({
            "name": sp,
            "composition": FlowDict(sorted(comp.items())),
            "cv_R": 2.5,
        })

    reactions = []

    for rxn in two_body:
        r = [s for s in rxn["reactants"] if s != "M"]
        p = [s for s in rxn["products"] if s != "M"]
        eq = format_equation(r, p, reversible=True)
        A_corr = tref_correct(rxn["A"], rxn["b"])
        reactions.append({
            "equation": eq,
            "type": "arrhenius",
            "reversible": True,
            "rate-constant": FlowDict([
                ("A", A_corr), ("b", rxn["b"]), ("Ea_R", rxn["Ea_R"])
            ]),
        })

    for rxn in three_body:
        r = rxn["reactants"]
        p = rxn["products"]
        eq = format_equation(r, p, reversible=True)
        A0_corr = tref_correct(rxn["A0"], rxn["b0"])

        if rxn.get("has_kinf"):
            A_inf_corr = tref_correct(rxn["A_inf"], rxn["b_inf"])
            entry = {
                "equation": eq,
                "type": "falloff",
                "reversible": True,
                "low-P-rate-constant": FlowDict([
                    ("A", A0_corr), ("b", rxn["b0"]),
                    ("Ea_R", rxn["Ea_R0"])
                ]),
                "high-P-rate-constant": FlowDict([
                    ("A", A_inf_corr), ("b", rxn["b_inf"]),
                    ("Ea_R", rxn["Ea_R_inf"])
                ]),
                "efficiencies": FlowDict(),
            }
        else:
            entry = {
                "equation": eq,
                "type": "three-body",
                "reversible": True,
                "rate-constant": FlowDict([
                    ("A", A0_corr), ("b", rxn["b0"]),
                    ("Ea_R", rxn["Ea_R0"])
                ]),
                "efficiencies": FlowDict(),
            }
        reactions.append(entry)

    for rxn in photolysis:
        r = rxn["reactants"]
        p = [s for s in rxn["products"] if s != "M"]
        eq = format_equation(r, p, reversible=False)
        entry = {"equation": eq, "type": "photolysis"}

        if photo_cross_dir:
            rows = build_photo_cross_section(
                rxn["parent"], rxn["br_index"], photo_cross_dir)
            if rows:
                entry["cross-section"] = [{"format": "YAML", "data": rows}]

        reactions.append(entry)

    return {
        "reference-state": ref_state,
        "species": species_entries,
        "reactions": reactions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert VULCAN network to kintera YAML")
    parser.add_argument("network_file", help="VULCAN network .txt file")
    parser.add_argument("--photo-cross-dir",
                        help="Directory containing photo_cross/ subdirs")
    parser.add_argument("-o", "--output", default="network.yaml",
                        help="Output YAML file")
    parser.add_argument("--extra-species", nargs="*", default=[],
                        help="Extra inert species to include (e.g. He)")
    args = parser.parse_args()

    print(f"Reading: {args.network_file}")
    two_body, three_body, photolysis = parse_network(args.network_file)
    species_list = collect_species(two_body, three_body, photolysis,
                                   extra=args.extra_species)

    print(f"  Two-body:   {len(two_body)}")
    print(f"  Three-body: {len(three_body)}")
    falloff = sum(1 for r in three_body if r.get("has_kinf"))
    simple = len(three_body) - falloff
    print(f"    Lindemann falloff: {falloff}")
    print(f"    Simple three-body: {simple}")
    print(f"  Photolysis: {len(photolysis)}")
    print(f"  Species:    {len(species_list)}")
    print(f"  Tref correction: {TREF} K")

    if photolysis and args.photo_cross_dir:
        parents = sorted(set(r["parent"] for r in photolysis))
        found = sum(1 for sp in parents if os.path.isfile(
            os.path.join(args.photo_cross_dir, sp, f"{sp}_cross.csv")))
        print(f"  Cross-sections: {found}/{len(parents)} species found")

    data = build_yaml(two_body, three_body, photolysis, species_list,
                      args.photo_cross_dir)

    with open(args.output, "w") as f:
        yaml.dump(data, f, Dumper=CustomDumper, default_flow_style=False,
                  sort_keys=False, width=120)

    n_rxn = len(data["reactions"])
    print(f"\nWritten: {args.output}")
    print(f"  {len(data['species'])} species, {n_rxn} reactions")


if __name__ == "__main__":
    main()
