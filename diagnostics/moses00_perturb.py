"""Perturbation experiment: take a single tiny-dt step from KB fort.7
state, with components selectively disabled, and report Δc per species.

A: baseline (Kzz + photo + escape all on)
B: Kzz = 0
C: Kzz = 0, photo sources removed
D: Kzz = 0, photo removed, upper-boundary escape removed (chemistry only)

Subtraction logic (since KB fort.7 should be at SS, Δc_KB ≈ 0):
  A − B: transport contribution
  B − C: RT/photolysis contribution
  C − D: escape contribution
  D    : pure chemistry residual at KB SS

Output: HTML report at /tmp/moses00_perturb.html with per-experiment
Δc/dt tables and a subtraction summary.
"""
from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch

import kintera as kt

torch.set_default_dtype(torch.float64)

EX_DIR = Path(
    "/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare/examples/titan_moses00"
)
PUN_PATH = EX_DIR / "kindata" / "kindata.titan.moses00.pun"
ATM_PATH = EX_DIR / "atm" / "atm.titan.moses00.kt.inp"
BC_PATH = EX_DIR / "boundary" / "boundary.moses00.inp"
RUN_INPUT = EX_DIR / "case" / "kinetics.inp"
REF_PATH = EX_DIR / "case" / "fort.7.kt"
CHENG_DIR = Path(
    "/home/sam2/dev/kintera/diagnostics/KINETICS-base-compare/examples/titan"
)

FLOATING_IDS = [1] + list(range(3, 46)) + list(range(68, 84))
FIXED_IDS = [67, 2] + list(range(46, 67)) + list(range(84, 87)) + [87]
CONCENTRATION_SPECIES: set = set()
SKIP_INJECT = {"M", "JDUST"}  # JDUST: KB fort.7 has it as aerosol MR up to 4.5e-6 at top,
# and kintera's per-particle σ=1e-8 cm² turns that into τ≈2800. moses00 has no aerosol
# chemistry, so force JDUST=0 (build-time value from zero_profile in atm shim).

DT = float(os.environ.get("KINTERA_PERTURB_DT", "1.0e-2"))
LAYERS_TO_REPORT = [20, 30, 40, 50, 60, 70]
REPORT_SPECIES = [
    "H", "H2", "CH4", "CH3",
    "C2H2", "C2H4", "C2H6", "C2H5",
    "C2H", "C2", "C", "CH", "CN",
    "C3H6", "C3H8", "C4H10", "1-C4H6",
    "HCN", "HC3N", "C2N2", "C6H2", "C6H3",
]
OUT_HTML = Path("/tmp/moses00_perturb.html")


def species_names(pun, ids):
    by_id = {s.id: s.name for s in pun.species}
    return [by_id[i] for i in ids if i in by_id]


def inject_state(ts, ref):
    c = ts.concentration.clone()
    density = ts.density[0]
    for name in ts.species:
        i = ts.species.index(name)
        if name in SKIP_INJECT:
            continue
        if name not in ref.species_profiles:
            continue
        prof = torch.as_tensor(np.array(ref.species_profiles[name]), dtype=c.dtype)
        if prof.numel() != c.shape[1]:
            continue
        if name in CONCENTRATION_SPECIES:
            c[0, :, i] = prof
        else:
            c[0, :, i] = prof * density
    return c


def build_state_and_sources():
    pun = kt.parse_kinetics_base_pun(str(PUN_PATH))
    floating = species_names(pun, FLOATING_IDS)
    fixed = species_names(pun, FIXED_IDS)
    species = floating + fixed

    raw_atm = kt.parse_kinetics_base_atmosphere(str(ATM_PATH))
    missing = [s for s in species if s not in raw_atm.species_profiles]
    # N2, M are bath/3rd-body. JDUST is aerosol loading; moses00's atm file
    # contains a real Cheng-era aerosol profile but kintera's cross-section
    # model (σ=1e-8 cm² per JDUST unit) gives τ≈2800 with that loading,
    # killing photolysis. moses00 has no aerosol chemistry, so force JDUST=0.
    density_like = ("N2", "M")
    force_zero = ("JDUST",)
    zero_profile = [0.0] * len(raw_atm.altitude)
    class _AtmShim:
        altitude = list(raw_atm.altitude)
        density = list(raw_atm.density)
        temperature = list(raw_atm.temperature)
        pressure = list(raw_atm.pressure)
        eddy_diffusion = list(raw_atm.eddy_diffusion)
        species_profiles = dict(raw_atm.species_profiles)
        for s in missing:
            species_profiles[s] = list(raw_atm.density) if s in density_like else list(zero_profile)
        for s in force_zero:
            if s in species_profiles:
                species_profiles[s] = list(zero_profile)
    atm = _AtmShim()
    ts = kt.build_kinetics_base_titan_state(
        atm, species=species, fixed_species=fixed,
        boundary_path=str(BC_PATH), pun_path=str(PUN_PATH),
    )
    sterms = kt.build_kinetics_base_titan_source_terms(
        str(PUN_PATH),
        boundary_path=str(BC_PATH),
        run_input_path=str(RUN_INPUT),
        photo_catalog_path=str(CHENG_DIR / "Cheng_catalog_v4.dat"),
        cross_dir=str(CHENG_DIR / "Cheng_cross"),
        flux_path=str(EX_DIR / "flux2atmos.inp"),
    )
    species_set = set(species)
    filtered = [
        t for t in sterms
        if all(r in species_set for r in (t.reactants or []) + (t.products or []))
    ]
    for t in filtered:
        t.parameters["freeze_actinic_flux"] = False
    return ts, filtered, species


def split_filtered(filtered, include_photo, include_boundary):
    """Split filtered source terms into kinds. include_X flags drop those kinds."""
    out = []
    for t in filtered:
        if not include_photo and t.kind == "pun_photo_rate_reaction":
            continue
        if not include_boundary and t.kind in {
            "upper_boundary_velocity",
            "upper_boundary_flux",
            "lower_boundary_velocity",
            "lower_boundary_flux",
        }:
            continue
        out.append(t)
    return out


def take_step(ts, atm_sources, dt, use_kzz):
    """Return c_after − c_before (Δc per layer per species), tensor shape (nlyr, nspecies)."""
    c0 = ts.state.concentration.clone()
    kzz = ts.kzz if use_kzz else torch.zeros_like(ts.kzz)
    sys_mat, rhs = kt.build_implicit_step_system(
        ts.state, kzz, float(dt),
        density=ts.density,
        transport_form="mr_diffusion",
        source_terms=atm_sources,
    )
    sys_mat, rhs = kt.apply_kinetics_base_titan_dirichlet_rows(sys_mat, rhs, ts)
    c_after = kt.solve_sparse_system(sys_mat, rhs)
    c_after = torch.clamp(c_after, min=0.0)
    c_after = kt.apply_kinetics_base_titan_boundary_pins(c_after, ts)
    delta = c_after - c0
    return delta[0]  # (nlyr, nspecies)


def run_experiment(label, ts, filtered, dt, *, use_kzz, include_photo, include_boundary):
    print(f"\n[exp {label}] use_kzz={use_kzz}, include_photo={include_photo}, "
          f"include_boundary={include_boundary}")
    terms = split_filtered(filtered, include_photo=include_photo, include_boundary=include_boundary)
    n_photo = sum(1 for t in terms if t.kind == "pun_photo_rate_reaction")
    print(f"  source terms: {len(terms)} (photo: {n_photo})")
    atm_sources = kt.build_kinetics_base_titan_atm2d_source_terms(ts, terms)
    delta = take_step(ts, atm_sources, dt, use_kzz)
    return delta.numpy()


def main():
    print(f"[setup] dt = {DT:.1e} s")
    ts, filtered, species = build_state_and_sources()
    print(f"  state: {ts.state.nspecies} species × {ts.state.nlyr} altitudes")

    ref = kt.parse_kinetics_base_atmosphere(str(REF_PATH))
    ts.state.concentration = inject_state(ts, ref)
    ts.concentration = ts.state.concentration
    c0 = ts.state.concentration[0].clone().numpy()
    density = ts.density[0].numpy()
    altitude = ts.state.x1v.numpy() / 1.0e5
    temperature = ts.state.temperature[0].numpy()

    # Run 4 experiments. Each takes one dt step from the same injected state.
    results = {}
    for label, kwargs in [
        ("A", dict(use_kzz=True, include_photo=True, include_boundary=True)),
        ("B", dict(use_kzz=False, include_photo=True, include_boundary=True)),
        ("C", dict(use_kzz=False, include_photo=False, include_boundary=True)),
        ("D", dict(use_kzz=False, include_photo=False, include_boundary=False)),
    ]:
        # Re-inject state before each experiment (Dirichlet pins should keep it close,
        # but be safe).
        ts.state.concentration = inject_state(ts, ref)
        ts.concentration = ts.state.concentration
        results[label] = run_experiment(label, ts, filtered, DT, **kwargs)

    # Tendency dc/dt = Δc / dt (column-wise)
    tend = {k: results[k] / DT for k in results}

    # Compose summary
    species_idx = {s: i for i, s in enumerate(species)}

    # Console: per-species per-layer table
    print()
    print(f"=== dc/dt at KB fort.7 SS (dt={DT:.1e} s), per-species, per-layer ===")
    print(f"{'Species':<10} {'Layer':<6} {'c (cm-3)':<10} "
          f"{'A: full':<12} {'B: Kzz=0':<12} {'C: noRT':<12} {'D: chem':<12}"
          f" {'transp (A-B)':<14} {'RT (B-C)':<12} {'escape (C-D)':<14}")
    rows_html = []
    for sp in REPORT_SPECIES:
        if sp not in species_idx:
            continue
        i = species_idx[sp]
        for L in LAYERS_TO_REPORT:
            cval = c0[L, i]
            a = tend["A"][L, i]
            b = tend["B"][L, i]
            c = tend["C"][L, i]
            d = tend["D"][L, i]
            transp = a - b
            rt = b - c
            esc = c - d
            print(f"{sp:<10} L{L:<5d} {cval:<10.2e} "
                  f"{a:<+12.2e} {b:<+12.2e} {c:<+12.2e} {d:<+12.2e} "
                  f"{transp:<+14.2e} {rt:<+12.2e} {esc:<+14.2e}")
            rows_html.append((sp, L, altitude[L], cval, a, b, c, d, transp, rt, esc))

    # ---- HTML output ----
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>moses00 perturbation experiment</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         margin: 24px; max-width: 1500px; color: #222; font-size: 0.9em; }
  h1 { border-bottom: 2px solid #444; padding-bottom: 8px; }
  h2 { margin-top: 32px; border-bottom: 1px solid #aaa; padding-bottom: 4px; }
  table { border-collapse: collapse; font-family: 'SF Mono', Menlo, monospace;
          font-size: 0.85em; margin-top: 10px; }
  th, td { border: 1px solid #ccc; padding: 3px 7px; text-align: right; }
  th { background: #f0f0f0; }
  td.sp { font-weight: 600; background: #fafafa; text-align: left; }
  td.layer { font-weight: 600; background: #fafafa; }
  .pos { color: #1c7f1c; }
  .neg { color: #a01010; }
  .meta { color: #666; font-size: 0.9em; }
  .formula { background: #f8f8f8; padding: 8px 14px; border-left: 3px solid #888;
             font-family: 'SF Mono', monospace; }
</style></head>
<body>
""")
    html_parts.append("<h1>moses00 perturbation experiment from KB fort.7 SS</h1>")
    html_parts.append(
        f"<p class='meta'>dt = {DT:.1e} s. "
        f"State: KB fort.7 (converged SS). "
        f"Reference: KB at this state has dc/dt ≈ 0 by definition of SS, so "
        f"kintera's dc/dt is the disagreement.</p>"
    )
    html_parts.append("<div class='formula'>")
    html_parts.append("Experiments (kintera one-step Δc/dt from same injected state):<br>")
    html_parts.append("  A — full pipeline (Kzz + photo + escape)<br>")
    html_parts.append("  B — Kzz = 0 (transport off)<br>")
    html_parts.append("  C — Kzz = 0, photolysis sources removed<br>")
    html_parts.append("  D — Kzz = 0, photolysis off, escape off (chemistry only)<br>")
    html_parts.append("Subtractions:<br>")
    html_parts.append("  A − B = transport contribution<br>")
    html_parts.append("  B − C = radiative transfer / photolysis contribution<br>")
    html_parts.append("  C − D = upper-boundary escape contribution<br>")
    html_parts.append("  D = pure chemistry residual at KB SS")
    html_parts.append("</div>")

    html_parts.append("<h2>dc/dt per species per layer (cm⁻³/s)</h2>")
    html_parts.append("<table><tr>")
    for h in ["Species", "Layer", "z (km)", "c (cm⁻³)",
              "A: full", "B: Kzz=0", "C: noRT", "D: chem",
              "transport (A−B)", "RT (B−C)", "escape (C−D)"]:
        html_parts.append(f"<th>{h}</th>")
    html_parts.append("</tr>")
    prev_sp = None
    for sp, L, z, cval, a, b, c, d, tr, rt, esc in rows_html:
        sp_cell = f"<td class='sp'>{sp}</td>" if sp != prev_sp else "<td></td>"
        prev_sp = sp
        def fmt(v):
            cls = "pos" if v > 0 else ("neg" if v < 0 else "")
            return f"<td class='{cls}'>{v:+.2e}</td>"
        html_parts.append(
            f"<tr>{sp_cell}<td class='layer'>L{L}</td>"
            f"<td>{z:.0f}</td><td>{cval:.2e}</td>"
            f"{fmt(a)}{fmt(b)}{fmt(c)}{fmt(d)}{fmt(tr)}{fmt(rt)}{fmt(esc)}"
            "</tr>"
        )
    html_parts.append("</table>")

    # Save raw arrays for follow-up
    np.savez(
        "/tmp/moses00_perturb.npz",
        species=np.array(species, dtype=object),
        altitude=altitude,
        density=density,
        temperature=temperature,
        c_inject=c0,
        dt=DT,
        delta_A=results["A"],
        delta_B=results["B"],
        delta_C=results["C"],
        delta_D=results["D"],
    )
    print(f"\nSaved raw Δc arrays to /tmp/moses00_perturb.npz")

    html_parts.append("</body></html>")
    OUT_HTML.write_text("".join(html_parts))
    print(f"Wrote {OUT_HTML} ({OUT_HTML.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
