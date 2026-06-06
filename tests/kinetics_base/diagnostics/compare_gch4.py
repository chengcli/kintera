"""
Focused comparison: GCH4 and CH4 between KINETICS-base and kintera
after N timesteps.  Run with:
    KINTERA_KINETICS_BASE_ROOT=... KINTERA_KINETICS_BASE_EXECUTABLE=... \
    python diagnostics/compare_gch4.py
"""

import os
import re
import pathlib
import subprocess
import tempfile

import torch
import kintera as kt

DEFAULT_ROOT = pathlib.Path(__file__).resolve().parent / "KINETICS-base-compare"
ROOT = pathlib.Path(os.environ.get("KINTERA_KINETICS_BASE_ROOT", DEFAULT_ROOT))
TITAN_DIR = ROOT / "examples" / "titan"
PUN_PATH = os.path.join(TITAN_DIR, "kindata_yy_clean", "Cheng_ions_c6h7+_v3_H2CN.pun")
RUN_INPUT_PATH = os.path.join(TITAN_DIR, "ions_c6h7+_H2CN.inp-1")
ATMOSPHERE_PATH = os.path.join(TITAN_DIR, "kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz")
EXECUTABLE = os.environ.get(
    "KINTERA_KINETICS_BASE_EXECUTABLE",
    str(ROOT / "build" / "bin" / "titan.release"),
)
NTIME = int(os.environ.get("KINTERA_TITAN_NTIME", "10"))
NETWORK_MODE = os.environ.get("KINTERA_TITAN_NETWORK_MODE", "full")
MAX_SUBDIVISIONS = int(os.environ.get("KINTERA_TITAN_MAX_SUBDIV", "20"))


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def _write_fresh_start_inp(src, dst, ntime, network_mode="full"):
    with open(src) as f:
        text = f.read()
    lines = text.splitlines()
    if network_mode == "no_grain":
        _disable_grain_flags(lines)
    for i, line in enumerate(lines):
        if line.strip().startswith("NTIME "):
            for j in range(i + 1, len(lines)):
                if re.match(r"^\s*\d+", lines[j]):
                    parts = lines[j].split()
                    if len(parts) >= 11:
                        parts[0] = str(ntime)
                        parts[10] = "0"
                        lines[j] = " ".join(parts)
                        with open(dst, "w") as f:
                            f.write("\n".join(lines) + "\n")
                        return
    raise AssertionError("ISTART field not found")


def _disable_grain_flags(lines):
    for i, line in enumerate(lines):
        if line.strip().startswith("FREEZE"):
            if i + 1 >= len(lines):
                raise AssertionError("missing gas-grain flag values")
            parts = lines[i + 1].split()
            if len(parts) < 2:
                raise AssertionError("invalid gas-grain flag values")
            parts[0] = "0"
            parts[1] = "0"
            lines[i + 1] = " ".join(parts)
            return
    raise AssertionError("gas-grain flag header not found")


def _run_kinetics_base(ntime, work, network_mode="full"):
    (work / "prod+loss").mkdir(exist_ok=True)
    inp_dst = work / "fort.81.fresh-start"
    _write_fresh_start_inp(RUN_INPUT_PATH, inp_dst, ntime, network_mode)
    links = {
        "fort.1": PUN_PATH,
        "fort.3": os.path.join(TITAN_DIR, "kintitan.truncate"),
        "fort.4": os.path.join(TITAN_DIR, "kindata_yy_clean", "Cheng_ions_c6h7+_v3_H2CN.special"),
        "fort.15": os.path.join(TITAN_DIR, "titan_Cheng_N_ions_H2CN.bc_save"),
        "fort.20": os.path.join(TITAN_DIR, "Cheng_wavel.dat"),
        "fort.21": os.path.join(TITAN_DIR, "flare_kin_oct2003.inp"),
        "fort.27": os.path.join(TITAN_DIR, "kintitan-difrad-2.inp"),
        "fort.30": os.path.join(TITAN_DIR, "Cheng_catalog_v4.dat"),
        "fort.45": os.path.join(TITAN_DIR, "kintitan_aerosol_interp_albedo.inp"),
        "fort.46": os.path.join(TITAN_DIR, "kintitan_aerosol_interp_gr.inp"),
        "fort.47": os.path.join(TITAN_DIR, "kintitan_aerosol_interp_asymm.inp"),
        "fort.50": ATMOSPHERE_PATH,
        "fort.81": inp_dst,
        "crossfilepath": os.path.join(TITAN_DIR, "Cheng_cross"),
    }
    for name, target in links.items():
        os.symlink(target, work / name)
    for name in ["kintitan.out.pun", "kintitan.res", "titandebug.dat"]:
        (work / name).touch()
    os.symlink(work / "kintitan.out.pun", work / "fort.7")
    os.symlink(work / "kintitan.res", work / "fort.10")
    os.symlink(work / "titandebug.dat", work / "fort.11")
    proc = subprocess.run(
        [EXECUTABLE], cwd=work,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=120, check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stdout[-2000:])
    return work / "kintitan.out.pun"


def _write_effective_bc(src, dst, atmosphere):
    del atmosphere
    with open(src) as f:
        lines = f.readlines()
    with open(dst, "w") as f:
        f.writelines(lines)


def _run_kintera(titan_state, source_terms, pun_metadata, ntime):
    source_terms = _filter_source_terms_for_network(source_terms, NETWORK_MODE)
    concentration = titan_state.concentration
    atm_sources = kt.build_kinetics_base_titan_atm2d_source_terms(
        titan_state, source_terms, pun_metadata=pun_metadata
    )
    species_diffusion_scale = kt.kinetics_base_titan_species_diffusion_scale(
        titan_state.species,
        dtype=titan_state.state.dtype,
        device=titan_state.state.device,
    )

    def step_fn(state, dt):
        system, rhs = kt.build_implicit_step_system(
            state,
            titan_state.kzz,
            dt,
            species_diffusion_scale=species_diffusion_scale,
            source_terms=atm_sources,
        )
        system, rhs = kt.apply_kinetics_base_titan_dirichlet_rows(system, rhs, titan_state)
        new_conc = kt.solve_sparse_system(system, rhs)
        kt.apply_kinetics_base_titan_boundary_pins(new_conc, titan_state)
        return new_conc

    total_accept = 0
    total_reject = 0
    for k, dt_target in enumerate(_fixed_timestep_sequence(ntime)):
        titan_state.state.concentration = concentration
        result = kt.adaptive_advance(
            titan_state.state,
            dt_target,
            step_fn,
            max_subdivisions=MAX_SUBDIVISIONS,
            record_trace=True,
        )
        concentration = result.concentration
        total_accept += result.accepted_steps
        total_reject += result.rejected_steps
        if result.rejected_steps > 0 or result.accepted_steps > 1:
            reasons = {}
            for rec in result.records:
                if rec.action.startswith("reject_"):
                    reasons[rec.action] = reasons.get(rec.action, 0) + 1
            reasons_str = ", ".join(f"{k}={v}" for k, v in reasons.items()) or "-"
            print(
                f"  step {k+1:>3d}/{ntime}: dt_target={dt_target:.3e} "
                f"accepted={result.accepted_steps} rejected={result.rejected_steps} "
                f"last_dt={result.last_accepted_dt:.3e} reasons=[{reasons_str}]"
            )

    print(f"  totals: accepted={total_accept} rejected={total_reject}")
    return concentration[0]  # (nlyr, nspecies)


def _fixed_timestep_sequence(ntime):
    dt = 1.0e-15
    growth = 10 ** 0.5
    sequence = []
    for _ in range(ntime):
        sequence.append(dt)
        dt *= growth
    return sequence


def _filter_source_terms_for_network(source_terms, network_mode):
    if network_mode == "full":
        return source_terms
    if network_mode != "no_grain":
        raise ValueError(f"unknown network mode: {network_mode}")
    return [
        term
        for term in source_terms
        if not any(_is_grain_related_species(name) for name in term.reactants + term.products)
    ]


def _is_grain_related_species(name):
    return name in {"SGA", "U"} or name.startswith("G")


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        work = pathlib.Path(tmpdir)

        # --- run KINETICS-base ---
        print(f"[1/4] Running KINETICS-base for {NTIME} steps ({NETWORK_MODE}) ...")
        kb_out = _run_kinetics_base(NTIME, work, NETWORK_MODE)

        initial = kt.parse_kinetics_base_atmosphere(ATMOSPHERE_PATH)
        reference = kt.parse_kinetics_base_atmosphere(str(kb_out))
        species = [n for n in initial.species_profiles if n in reference.species_profiles]
        print(f"      {len(species)} species in both initial and reference")

        ref_conc = kt.kinetics_base_profile_tensor(reference, species)  # (nlyr, nsp)

        # --- build kintera state ---
        print("[2/4] Building kintera state ...")
        bc_path = work / "effective.bc"
        _write_effective_bc(
            os.path.join(TITAN_DIR, "titan_Cheng_N_ions_H2CN.bc_save"),
            bc_path, initial,
        )
        titan_state = kt.build_kinetics_base_titan_state(
            initial, species=species,
            boundary_path=str(bc_path), pun_path=PUN_PATH,
        )

        # --- build source terms ---
        print("[3/4] Building kintera source terms ...")
        source_terms, pun_metadata = (
            kt.build_kinetics_base_titan_source_terms(
                PUN_PATH,
                special_path=os.path.join(
                    TITAN_DIR, "kindata_yy_clean", "Cheng_ions_c6h7+_v3_H2CN.special"
                ),
                boundary_path=os.path.join(TITAN_DIR, "titan_Cheng_N_ions_H2CN.bc_save"),
                run_input_path=RUN_INPUT_PATH,
                photo_catalog_path=os.path.join(TITAN_DIR, "Cheng_catalog_v4.dat"),
                cross_dir=os.path.join(TITAN_DIR, "Cheng_cross"),
                flux_path=os.path.join(TITAN_DIR, "flare_kin_oct2003.inp"),
            ),
            kt.kinetics_base_species_metadata_from_pun(PUN_PATH),
        )

        # --- run kintera ---
        print(f"[4/4] Running kintera for {NTIME} steps ({NETWORK_MODE}) ...")
        kt_conc = _run_kintera(titan_state, source_terms, pun_metadata, NTIME)

    # --- compare ---
    alt = initial.altitude
    sp_idx = {s: i for i, s in enumerate(species)}

    print()
    print("=" * 90)
    print(f"  {'Lev':>3}  {'Alt(km)':>8}  {'KB_GCH4':>12}  {'KT_GCH4':>12}  {'ratio':>8}  "
          f"{'KB_CH4':>12}  {'KT_CH4':>12}  {'CH4_ratio':>10}")
    print("=" * 90)

    for ilev in range(min(25, len(alt))):
        kb_gch4 = ref_conc[ilev, sp_idx["GCH4"]].item() if "GCH4" in sp_idx else 0.0
        kt_gch4 = kt_conc[ilev, sp_idx["GCH4"]].item() if "GCH4" in sp_idx else 0.0
        kb_ch4  = ref_conc[ilev, sp_idx["CH4"]].item()
        kt_ch4  = kt_conc[ilev, sp_idx["CH4"]].item()
        g_ratio = kt_gch4 / kb_gch4 if abs(kb_gch4) > 1e-30 else float("nan")
        c_ratio = kt_ch4 / kb_ch4   if abs(kb_ch4)  > 1e-30 else float("nan")
        print(
            f"  {ilev+1:3d}  {alt[ilev]:8.1f}  {kb_gch4:12.3e}  {kt_gch4:12.3e}  "
            f"{g_ratio:8.3f}  {kb_ch4:12.3e}  {kt_ch4:12.3e}  {c_ratio:10.6f}"
        )

    # --- overall mismatch summary ---
    print()
    print("Top-10 species by absolute difference (kintera vs KB after 10 steps):")
    diff = (kt_conc - ref_conc).abs()
    species_max = diff.max(dim=0).values
    top_vals, top_idx = torch.topk(species_max, min(10, len(species)))
    for v, i in zip(top_vals, top_idx):
        print(f"  {species[int(i)]:<12s}  {v.item():.4e}")


if __name__ == "__main__":
    main()
