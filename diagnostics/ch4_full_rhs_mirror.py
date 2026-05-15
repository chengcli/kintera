#!/usr/bin/env python3
"""Dump kintera's full per-reaction CH4 chemistry RHS on a KB state.

This intentionally mirrors the KINETICS-base arrays we need to instrument:
``SRATE``, ``ICOFF/ICOFD`` contribution, and per-reaction production/loss.

The script does not run KINETICS-base. Point ``--state`` at a KB
``kintitan.out.pun`` from a completed run and keep ``--initial`` at the fresh
start atmosphere profile.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import torch

import kintera as kt


def _default_titan_root() -> Path:
    default_root = Path(__file__).resolve().parent / "KINETICS-base-compare"
    return Path(os.environ.get("KINTERA_KINETICS_BASE_ROOT", default_root))


def _default_paths(root: Path) -> dict[str, Path]:
    titan_dir = root / "examples" / "titan"
    return {
        "pun": titan_dir / "kindata_yy_clean" / "Cheng_ions_c6h7+_v3_H2CN.pun",
        "special": titan_dir / "kindata_yy_clean" / "Cheng_ions_c6h7+_v3_H2CN.special",
        "run_input": titan_dir / "ions_c6h7+_H2CN.inp-1",
        "initial": titan_dir / "kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz",
        "boundary": titan_dir / "titan_Cheng_N_ions_H2CN.bc_save",
        "catalog": titan_dir / "Cheng_catalog_v4.dat",
        "cross_dir": titan_dir / "Cheng_cross",
        "flux": titan_dir / "flare_kin_oct2003.inp",
        "truncate": titan_dir / "kintitan.truncate",
    }


def _write_effective_ch4_boundary(src: Path, dst: Path, atmosphere: Any) -> None:
    del atmosphere
    dst.write_text(src.read_text())


def _source_terms(paths: dict[str, Path]) -> tuple[list[kt.KBTitanSourceTerm], dict[str, Any]]:
    terms = kt.build_kinetics_base_titan_source_terms(
        paths["pun"],
        special_path=paths["special"],
        boundary_path=paths["boundary"],
        run_input_path=paths["run_input"],
        photo_catalog_path=paths["catalog"],
        cross_dir=paths["cross_dir"],
        flux_path=paths["flux"],
        truncate_path=paths["truncate"],
    )
    metadata = kt.kinetics_base_species_metadata_from_pun(paths["pun"])
    return terms, metadata


def _species(initial: Any, current: Any) -> list[str]:
    return [
        name
        for name in initial.species_profiles
        if name in current.species_profiles
    ]


def _term_tendency(
    titan_state: kt.KBTitanState,
    term: kt.KBTitanSourceTerm,
    metadata: dict[str, Any],
) -> torch.Tensor | None:
    sources = kt.build_kinetics_base_titan_atm2d_source_terms(
        titan_state,
        [term],
        pun_metadata=metadata,
    )
    if not sources:
        return None
    linearization = kt.build_source_linearization(titan_state.state, sources)
    return linearization.tendency


def _reaction_label(term: kt.KBTitanSourceTerm) -> str:
    left = " + ".join(term.reactants) if term.reactants else "<none>"
    right = " + ".join(term.products) if term.products else "<none>"
    return f"{left} -> {right}"


def dump_ch4_rhs(args: argparse.Namespace) -> None:
    root = Path(args.root) if args.root else _default_titan_root()
    paths = _default_paths(root)
    for name in [
        "pun",
        "special",
        "run_input",
        "boundary",
        "catalog",
        "cross_dir",
        "flux",
        "truncate",
    ]:
        override = getattr(args, name)
        if override is not None:
            paths[name] = Path(override)
    if args.initial is not None:
        paths["initial"] = Path(args.initial)

    state_path = Path(args.state)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    initial = kt.parse_kinetics_base_atmosphere(str(paths["initial"]))
    current = kt.parse_kinetics_base_atmosphere(str(state_path))
    species = _species(initial, current)
    species_index = {name: i for i, name in enumerate(species)}
    if "CH4" not in species_index:
        raise RuntimeError("CH4 is not present in the selected species list")

    boundary_path = out_path.with_suffix(".effective.bc")
    _write_effective_ch4_boundary(paths["boundary"], boundary_path, initial)

    metadata = kt.kinetics_base_species_metadata_from_pun(paths["pun"])
    titan_state = kt.build_kinetics_base_titan_state(
        initial,
        species=species,
        boundary_path=boundary_path,
        pun_path=paths["pun"],
        pun_metadata=metadata,
    )
    current_concentration = kt.kinetics_base_profile_tensor(current, species).view(
        1, len(current.altitude), len(species)
    )
    titan_state.state.concentration = current_concentration

    terms, metadata = _source_terms(paths)
    network = kt.parse_kinetics_base_truncate(paths["truncate"])
    ch4_idx = species_index["CH4"]

    rows: list[dict[str, Any]] = []
    levels = range(current_concentration.shape[1])
    if args.levels:
        levels = [int(value) for value in args.levels.split(",")]

    for term in terms:
        tendency = _term_tendency(titan_state, term, metadata)
        if tendency is None:
            continue
        ch4_tendency = tendency[0, :, ch4_idx]
        if not torch.any(ch4_tendency != 0.0):
            continue
        op_id = (
            network.reaction_mapping.get(term.reaction_id)
            if term.reaction_id is not None
            else None
        )
        for level in levels:
            value = float(ch4_tendency[level].item())
            if value == 0.0 and not args.include_zero:
                continue
            rows.append(
                {
                    "level0": level,
                    "level1": level + 1,
                    "alt_km": float(current.altitude[level]),
                    "op_id": op_id,
                    "orig_id": term.reaction_id,
                    "kind": term.kind,
                    "source": term.parameters.get("source"),
                    "reaction": _reaction_label(term),
                    "ch4_tendency": value,
                    "ch4_prod": max(value, 0.0),
                    "ch4_loss": max(-value, 0.0),
                }
            )

    rows.sort(key=lambda row: (row["level0"], -row["ch4_loss"], -row["ch4_prod"]))
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "level0",
                "level1",
                "alt_km",
                "op_id",
                "orig_id",
                "kind",
                "source",
                "reaction",
                "ch4_tendency",
                "ch4_prod",
                "ch4_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} CH4 RHS rows to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="KINETICS-base checkout root")
    parser.add_argument("--state", required=True, help="KB kintitan.out.pun to evaluate")
    parser.add_argument("--output", required=True, help="CSV output path")
    parser.add_argument("--levels", help="comma-separated 0-indexed levels to dump")
    parser.add_argument("--include-zero", action="store_true")
    parser.add_argument("--pun")
    parser.add_argument("--special")
    parser.add_argument("--run-input")
    parser.add_argument("--initial")
    parser.add_argument("--boundary")
    parser.add_argument("--catalog")
    parser.add_argument("--cross-dir")
    parser.add_argument("--flux")
    parser.add_argument("--truncate")
    dump_ch4_rhs(parser.parse_args())


if __name__ == "__main__":
    main()
