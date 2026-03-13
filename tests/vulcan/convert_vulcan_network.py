#!/usr/bin/env python3
"""Convert VULCAN NCHO network file to kintera YAML format."""

import re
import os
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VULCAN_FILE = os.path.join(SCRIPT_DIR, "VULCAN/thermo/NCHO_photo_network.txt")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "ncho_network.yaml")
PHOTO_CROSS_DIR = os.path.join(SCRIPT_DIR, "VULCAN/thermo/photo_cross")

NUM_RE = re.compile(r"[-+]?\d+\.?\d*[eE][-+]?\d+|[-+]?\d+\.?\d*")


class FlowDict(dict):
    pass


class FlowList(list):
    pass


class CustomDumper(yaml.SafeDumper):
    pass


def _flow_dict_repr(dumper, data):
    return dumper.represent_mapping(
        "tag:yaml.org,2002:map", data.items(), flow_style=True
    )


def _flow_list_repr(dumper, data):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    )


def _float_repr(dumper, value):
    if value != 0 and (abs(value) >= 1e5 or abs(value) < 0.01):
        s = f"{value:.4e}"
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


def format_equation(reactants, products):
    return " + ".join(reactants) + " => " + " + ".join(products)


def encode_branch(products):
    """Encode products as branch string, e.g. 'H:1 OH:1'."""
    counts = {}
    for p in products:
        if p != "M":
            counts[p] = counts.get(p, 0) + 1
    return " ".join(f"{sp}:{c}" for sp, c in counts.items())


def find_cross_section(species, base_dir):
    cross = os.path.join(base_dir, species, f"{species}_cross.csv")
    branch = os.path.join(base_dir, species, f"{species}_branch.csv")
    if os.path.isfile(cross):
        rc = f"photo_cross/{species}/{species}_cross.csv"
        rb = (
            f"photo_cross/{species}/{species}_branch.csv"
            if os.path.isfile(branch)
            else None
        )
        return rc, rb
    return None, None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_network(filepath):
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

        if rxn_id % 2 == 0 or rxn_id == 781:
            continue

        eq_m = re.search(r"\[\s*(.*?)\s*\]", line)
        if not eq_m:
            continue
        equation = eq_m.group(1).strip()
        after = line[eq_m.end() :].strip()

        reactants, products = parse_equation(equation)

        if reactants == products:
            continue

        if section == "two_body":
            nums = NUM_RE.findall(after)
            if len(nums) < 3:
                continue
            two_body.append(
                {
                    "id": rxn_id,
                    "reactants": reactants,
                    "products": products,
                    "A": float(nums[0]),
                    "b": float(nums[1]),
                    "Ea_R": float(nums[2]),
                }
            )

        elif section in ("three_body_kinf", "three_body_simple"):
            if "M" in reactants and "M" not in products:
                products.append("M")

            nums = NUM_RE.findall(after)
            if section == "three_body_kinf" and len(nums) >= 6:
                entry = {
                    "id": rxn_id,
                    "reactants": reactants,
                    "products": products,
                    "A0": float(nums[0]),
                    "b0": float(nums[1]),
                    "Ea_R0": float(nums[2]),
                    "A_inf": float(nums[3]),
                    "b_inf": float(nums[4]),
                    "Ea_R_inf": float(nums[5]),
                    "has_kinf": True,
                }
            elif len(nums) >= 3:
                entry = {
                    "id": rxn_id,
                    "reactants": reactants,
                    "products": products,
                    "A0": float(nums[0]),
                    "b0": float(nums[1]),
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
            photolysis.append(
                {
                    "id": rxn_id,
                    "parent": parent,
                    "br_index": br_idx,
                    "reactants": reactants,
                    "products": products,
                }
            )

    return two_body, three_body, photolysis


def collect_species(two_body, three_body, photolysis):
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
    return sorted(species)


# ---------------------------------------------------------------------------
# YAML construction
# ---------------------------------------------------------------------------

def build_yaml(two_body, three_body, photolysis, species_list, photo_cross_dir):
    ref_state = {"Tref": 300.0, "Pref": 1.0e5}

    species_entries = []
    for sp in species_list:
        comp = parse_formula(sp)
        species_entries.append(
            {
                "name": sp,
                "composition": FlowDict(sorted(comp.items())),
                "cv_R": 2.5,
            }
        )

    reactions = []

    for rxn in two_body:
        eq = format_equation(rxn["reactants"], rxn["products"])
        reactions.append(
            {
                "equation": eq,
                "type": "arrhenius",
                "rate-constant": FlowDict(
                    [("A", rxn["A"]), ("b", rxn["b"]), ("Ea_R", rxn["Ea_R"])]
                ),
            }
        )

    for rxn in three_body:
        eq = format_equation(rxn["reactants"], rxn["products"])
        entry = {"equation": eq, "type": "three-body"}
        entry["rate-constant"] = FlowDict(
            [("A", rxn["A0"]), ("b", rxn["b0"]), ("Ea_R", rxn["Ea_R0"])]
        )
        if rxn.get("has_kinf"):
            entry["high-pressure-rate-constant"] = FlowDict(
                [
                    ("A", rxn["A_inf"]),
                    ("b", rxn["b_inf"]),
                    ("Ea_R", rxn["Ea_R_inf"]),
                ]
            )
        entry["efficiencies"] = FlowDict()
        reactions.append(entry)

    for rxn in photolysis:
        eq = format_equation(rxn["reactants"], rxn["products"])
        branch = encode_branch(rxn["products"])
        entry = {"equation": eq, "type": "photolysis"}

        cross_file, branch_file = find_cross_section(rxn["parent"], photo_cross_dir)
        if cross_file:
            cs = {
                "format": "KINETICS7",
                "filename": cross_file,
                "branches": FlowList([branch]),
            }
            if branch_file:
                cs["branch-ratios"] = branch_file
            entry["cross-section"] = [cs]

        reactions.append(entry)

    return {"reference-state": ref_state, "species": species_entries, "reactions": reactions}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Reading VULCAN network: {VULCAN_FILE}")

    two_body, three_body, photolysis = parse_network(VULCAN_FILE)
    species_list = collect_species(two_body, three_body, photolysis)

    print("\n--- Parsing Statistics ---")
    print(f"  Two-body reactions:   {len(two_body)}")
    print(f"  Three-body reactions: {len(three_body)}")
    print(f"  Photolysis reactions: {len(photolysis)}")
    total = len(two_body) + len(three_body) + len(photolysis)
    print(f"  Total reactions:      {total}")
    print(f"  Unique species:       {len(species_list)}")

    print("\n--- Species ---")
    for sp in species_list:
        comp = parse_formula(sp)
        comp_str = ", ".join(f"{k}: {v}" for k, v in sorted(comp.items()))
        print(f"  {sp:12s} -> {{{comp_str}}}")

    photo_parents = sorted(set(r["parent"] for r in photolysis))
    found = missing = 0
    missing_list = []
    for sp in photo_parents:
        cf, _ = find_cross_section(sp, PHOTO_CROSS_DIR)
        if cf:
            found += 1
        else:
            missing += 1
            missing_list.append(sp)

    print("\n--- Cross-section files ---")
    print(f"  Found:   {found}")
    print(f"  Missing: {missing}")
    for sp in missing_list:
        print(f"    WARNING: no cross-section file for {sp}")

    data = build_yaml(two_body, three_body, photolysis, species_list, PHOTO_CROSS_DIR)

    with open(OUTPUT_FILE, "w") as f:
        yaml.dump(
            data,
            f,
            Dumper=CustomDumper,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )

    print(f"\nOutput written to: {OUTPUT_FILE}")
    print(
        f"YAML contains: {len(data['species'])} species, "
        f"{len(data['reactions'])} reactions"
    )


if __name__ == "__main__":
    main()
