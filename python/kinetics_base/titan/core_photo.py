"""Build core `kintera.PhotolysisOptions` from the Titan Cheng catalog.

Part of `unify-titan-chem-onto-core` Stage 3: route Titan photolysis through
the compiled core `kintera.Photolysis` engine instead of the hand-rolled
`Σ σ_i F_i` Python sum, while preserving the validated rates exactly.

The validated Titan path integrates the photolysis rate as a per-wavelength-bin
sum `J = Σ_i σ_i(λ) F_i` where `F_i` is the actinic flux already integrated
over bin `i` (the Cheng flux grid is coarse and non-uniform, ~46 nm mean, so a
trapezoidal integral over λ would differ from this sum by ~46×). Core
`Photolysis.forward` defaults to a trapezoidal rule; we therefore set
`PhotolysisOptions.quadrature_weights` to unit weights so the core engine
computes the same per-bin sum exactly.

Policy (Titan-special, applied here, not in core): the Cheng `_XSCN_`
(X-ray/EUV radical) cross-section files are excluded by default to match KB
(`KINTERA_TITAN_PHOTO_INCLUDE_XSCN` re-enables); species names are normalized
with the c-/l- isomer strip; absorption (type 0) and branching/quantum-yield
(type 2) datasets are combined as `σ_branch = σ_parent × b_i`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import kintera as kt

from .parsing import (
    _kinetics_base_side_key,
    _parse_kinetics_base_catalog,
    _parse_kinetics_base_cross_section_on_flux,
    _parse_kinetics_base_equation,
    _parse_kinetics_base_flux,
)


def _equation_string(reactants: list[str], products: list[str]) -> str:
    """Build an irreversible core reaction equation from species lists."""

    def side(species: list[str]) -> str:
        counts: dict[str, int] = {}
        order: list[str] = []
        for sp in species:
            if sp not in counts:
                order.append(sp)
            counts[sp] = counts.get(sp, 0) + 1
        return " + ".join(
            (f"{counts[sp]} {sp}" if counts[sp] > 1 else sp) for sp in order
        )

    return f"{side(reactants)} => {side(products)}"


def build_titan_photolysis_options(
    catalog_path: str | Path,
    cross_dir: str | Path,
    flux_path: str | Path,
    *,
    solar_distance_au: float = 1.0,
    nwave1: int | None = None,
    nwave2: int | None = None,
    include_xscn: bool | None = None,
) -> tuple[Any, dict]:
    """Build `kintera.PhotolysisOptions` for the Titan Cheng catalog.

    Returns `(options, info)` where `info` records the wavelength grid, the
    per-reaction reference rate `Σ σ_i F_i` (validated sum), and provenance.
    The reference flux `F` used here is the unattenuated top-of-atmosphere flux;
    at runtime the attenuated actinic-flux field is passed to
    `Photolysis.forward`.
    """
    flux = _parse_kinetics_base_flux(str(flux_path))
    if not flux:
        raise ValueError(f"empty/unreadable Titan flux file: {flux_path}")

    flux_scale = 1.0 / max(float(solar_distance_au), 1.0e-300) ** 2
    flux = [(w, width, value * flux_scale) for (w, width, value) in flux]
    # KB only loads fort.20 rows NWAVE1..NWAVE2; zero the flux outside.
    if nwave1 is not None or nwave2 is not None:
        lo = (nwave1 - 1) if nwave1 is not None else 0
        hi = nwave2 if nwave2 is not None else len(flux)
        flux = [
            (w, width, value if (lo <= i < hi) else 0.0)
            for i, (w, width, value) in enumerate(flux)
        ]

    wavelengths = [row[0] for row in flux]
    flux_values = [row[2] for row in flux]
    nwave = len(flux)

    catalog = _parse_kinetics_base_catalog(str(catalog_path))
    if include_xscn is None:
        from .config import get_titan_config
        include_xscn = get_titan_config().photo_include_xscn
    if not include_xscn:
        catalog = [(eq, fn) for eq, fn in catalog if "_XSCN_" not in fn]

    cross_root = Path(cross_dir)
    cross_cache = {
        fn: _parse_kinetics_base_cross_section_on_flux(cross_root / fn, flux)
        for _, fn in catalog
    }

    # Pre-pass: parent absorption (type 0) cross-sections, for type-2 branching.
    absorption_by_parent: dict[str, list[float]] = {}
    for equation, fn in catalog:
        datasets = cross_cache.get(fn, {})
        absorption = datasets.get(0)
        if not absorption:
            continue
        reactants, _ = _parse_kinetics_base_equation(equation)
        if reactants:
            absorption_by_parent.setdefault(
                _kinetics_base_side_key(reactants), absorption
            )

    reactions: list[Any] = []
    cross_flat: list[float] = []
    branches: list[list[Any]] = []
    branch_names: list[list[str]] = []
    nslabs: list[int] = []
    provenance: list[dict] = []
    ref_rates: list[float] = []

    for equation, fn in catalog:
        datasets = cross_cache.get(fn, {})
        if not datasets:
            continue
        reactants, products = _parse_kinetics_base_equation(equation)
        if not reactants or not products or products == reactants:
            continue  # pure absorption rows define opacity, not a reaction

        cross = datasets.get(0)
        absorption = cross
        if cross is None and 2 in datasets:
            absorption = absorption_by_parent.get(_kinetics_base_side_key(reactants))
            if absorption is not None:
                cross = [a * b for a, b in zip(absorption, datasets[2])]
        if not cross or not any(cross):
            continue

        eq = _equation_string(reactants, products)
        try:
            rxn = kt.Reaction(eq)
        except Exception:
            continue  # species not in the active table; skip

        # branch 0 = photoabsorption (reactants); branch 1 = dissociation (products)
        absorb_comp: dict[str, float] = {}
        for sp in reactants:
            absorb_comp[sp] = absorb_comp.get(sp, 0.0) + 1.0
        diss_comp: dict[str, float] = {}
        for sp in products:
            diss_comp[sp] = diss_comp.get(sp, 0.0) + 1.0

        absorb_vals = absorption if absorption is not None else cross
        # cross-section layout per reaction: (nwave, nbranch) row-major
        for w in range(nwave):
            cross_flat.append(absorb_vals[w])  # branch 0 (unused in J)
            cross_flat.append(cross[w])         # branch 1 (dissociation -> J)

        reactions.append(rxn)
        # Composition is bound as a plain dict (std::map<str,double>).
        branches.append([absorb_comp, diss_comp])
        branch_names.append(
            [" ".join(f"{s}:1" for s in reactants),
             " ".join(f"{s}:1" for s in products)]
        )
        nslabs.append(1)
        ref_rates.append(sum(c * f for c, f in zip(cross, flux_values)))
        provenance.append(
            {"equation": eq, "file": fn, "reactants": reactants, "products": products}
        )

    options = kt.PhotolysisOptions()
    options.reactions(reactions)
    options.wavelength(wavelengths)
    options.cross_section(cross_flat)
    options.cross_section_nslabs(nslabs)
    options.branches(branches)
    options.branch_names(branch_names)
    # unit weights => core forward computes the validated per-bin sum Σ σ_i F_i
    options.quadrature_weights([1.0] * nwave)

    info = {
        "wavelengths": wavelengths,
        "flux": flux_values,
        "ref_rates": ref_rates,
        "provenance": provenance,
        "nwave": nwave,
        "nreaction": len(reactions),
    }
    return options, info
