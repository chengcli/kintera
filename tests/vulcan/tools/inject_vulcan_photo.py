#!/usr/bin/env python3
"""Inject VULCAN's runtime photolysis cross-sections into a kintera YAML.

Reads the .vul output file (which contains cross_J per species/branch on
VULCAN's wavelength grid) and replaces every photolysis reaction's
cross-section block in the kintera YAML so both codes integrate the
same data.

IMPORTANT: All reactions are written on the SAME wavelength grid (VULCAN's
bins) because kintera's photolysis module uses a single global grid set
by the first reaction.

Usage:
    python inject_vulcan_photo.py cho_photo.yaml \
        VULCAN/output/CHO-kintera.vul \
        -o cho_photo_matched.yaml
"""

import argparse
import os
import pickle
import re
import sys

import numpy as np
import yaml


# ── YAML formatting (same helpers as convert_network.py) ─────────────

class FlowList(list):
    pass

class CustomDumper(yaml.SafeDumper):
    pass

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

CustomDumper.add_representer(FlowList, _flow_list_repr)
CustomDumper.add_representer(float, _float_repr)


# ── Helpers ──────────────────────────────────────────────────────────

COEFF_RE = re.compile(r"^(\d+)\s+")


def parse_reactant(eq_str):
    """Return the parent species from 'H2O => H + OH' etc."""
    lhs = eq_str.split("=>")[0].strip()
    parts = lhs.split("+")
    for p in parts:
        p = p.strip()
        m = COEFF_RE.match(p)
        if m:
            p = p[m.end():].strip()
        if p:
            return p
    return lhs.strip()


def load_vul_photo(vul_path):
    """Load photolysis data from a .vul file."""
    with open(vul_path, "rb") as f:
        raw = pickle.load(f)
    v = raw["variable"]
    bins = np.array(v["bins"], dtype=np.float64)
    cross_J = {}
    for key, val in v["cross_J"].items():
        cross_J[key] = np.array(val, dtype=np.float64)
    n_branch = dict(v["n_branch"])
    return bins, cross_J, n_branch


def sigma_to_yaml_rows(wl, sigma):
    """Convert wavelength + cross-section arrays to YAML rows.
    All points are kept (same grid for every reaction).
    Values below 1e-200 are clamped to zero to avoid denormalized
    doubles that yaml-cpp cannot parse."""
    rows = []
    for i in range(len(wl)):
        val = float(sigma[i]) if sigma[i] > 1e-200 else 0.0
        rows.append(FlowList([float(wl[i]), float(f"{val:.6e}")]))
    return rows


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inject VULCAN cross-sections into kintera YAML")
    parser.add_argument("yaml_file", help="Input kintera YAML")
    parser.add_argument("vul_file", help="VULCAN .vul output")
    parser.add_argument("-o", "--output", default=None,
                        help="Output YAML (default: overwrite input)")
    args = parser.parse_args()

    out_path = args.output or args.yaml_file

    bins, cross_J, n_branch = load_vul_photo(args.vul_file)
    print(f"VULCAN: {len(bins)} wavelength bins, "
          f"{bins[0]:.1f}–{bins[-1]:.1f} nm")
    print(f"  Photo species: {sorted(n_branch.keys())}")
    print(f"  Total branches: {sum(n_branch.values())}")

    with open(args.yaml_file) as f:
        doc = yaml.safe_load(f)

    reactions = doc["reactions"]

    # Track branch counter per parent species
    branch_counter = {}
    n_updated = 0
    n_missing = 0

    for rxn in reactions:
        if rxn.get("type") != "photolysis":
            continue

        eq = rxn["equation"]
        parent = parse_reactant(eq)

        count = branch_counter.get(parent, 0) + 1
        branch_counter[parent] = count
        key = (parent, count)

        if key in cross_J:
            sigma = cross_J[key]
            rows = sigma_to_yaml_rows(bins, sigma)
            rxn["cross-section"] = [{"format": "YAML", "data": rows}]
            n_updated += 1
            max_s = sigma.max()
            if max_s > 0:
                print(f"  {eq:40s} branch {count}: "
                      f"max_sigma={max_s:.3e} cm²")
            else:
                print(f"  {eq:40s} branch {count}: all zero")
        else:
            # No VULCAN data — write zeros on the common grid
            rows = sigma_to_yaml_rows(bins, np.zeros_like(bins))
            rxn["cross-section"] = [{"format": "YAML", "data": rows}]
            n_missing += 1
            print(f"  WARNING: no VULCAN data for: {eq}")

    with open(out_path, "w") as f:
        yaml.dump(doc, f, Dumper=CustomDumper, default_flow_style=False,
                  sort_keys=False, width=120)

    print(f"\nUpdated {n_updated} photolysis cross-sections, "
          f"{n_missing} missing")
    print(f"All reactions use common grid: {len(bins)} pts, "
          f"{bins[0]:.1f}–{bins[-1]:.1f} nm")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
