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

TITAN_DIR = "/tmp/KINETICS-base-compare/examples/titan"
PUN_PATH = os.path.join(TITAN_DIR, "kindata_yy_clean", "Cheng_ions_c6h7+_v3_H2CN.pun")
RUN_INPUT_PATH = os.path.join(TITAN_DIR, "ions_c6h7+_H2CN.inp-1")
ATMOSPHERE_PATH = os.path.join(TITAN_DIR, "kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz")
EXECUTABLE = os.environ.get(
    "KINTERA_KINETICS_BASE_EXECUTABLE",
    "/tmp/KINETICS-base-compare/build/bin/titan.release",
)
NTIME = 10


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def _write_fresh_start_inp(src, dst, ntime):
    with open(src) as f:
        text = f.read()
    lines = text.splitlines()
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


def _run_kinetics_base(ntime, work):
    (work / "prod+loss").mkdir(exist_ok=True)
    inp_dst = work / "fort.81.fresh-start"
    _write_fresh_start_inp(RUN_INPUT_PATH, inp_dst, ntime)
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
    ch4_mr = atmosphere.species_profiles["CH4"][0]
    lines = []
    with open(src) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and parts[4] == "CH4" and parts[0] == "5":
                parts[1] = f"{ch4_mr:.8e}"
                line = f"{parts[0]} {parts[1]:<18} {parts[2]} {parts[3]:<18} {parts[4]}\n"
            lines.append(line)
    with open(dst, "w") as f:
        f.writelines(lines)


def _run_kintera(titan_state, source_terms, pun_metadata, ntime):
    concentration = titan_state.concentration
    _lower_boundary_conversions = {
        "lower_boundary_mixing_ratio_times_density",
        "lower_boundary_deposition_velocity_zero",
    }
    _upper_boundary_conversions = {
        "upper_boundary_escape_velocity_zero",
    }
    _cold_trap_conversions = {
        "kinetics_base_cheng_cold_trap_mixing_ratio",
    }
    boundary_idx = [
        titan_state.species.index(name)
        for name, conv in titan_state.conversion.items()
        if conv in _lower_boundary_conversions
    ]
    top_boundary_idx = [
        titan_state.species.index(name)
        for name, conv in titan_state.conversion.items()
        if conv in _upper_boundary_conversions
    ]
    cold_trap_idx = [
        titan_state.species.index(name)
        for name, conv in titan_state.conversion.items()
        if conv in _cold_trap_conversions
    ]
    nonzero_levels = (titan_state.density[0] > 0).nonzero(as_tuple=True)[0]
    last_real_lyr = int(nonzero_levels[-1]) if nonzero_levels.numel() > 0 else titan_state.state.nlyr - 1
    _COLD_TRAP_LEVEL = 23
    fixed_idx = [
        titan_state.species.index(name)
        for name in titan_state.fixed_species
        if name in titan_state.species
    ]
    atm_sources = kt.build_kinetics_base_titan_atm2d_source_terms(
        titan_state, source_terms, pun_metadata=pun_metadata
    )
    dt = 1e-15
    growth = 10 ** 0.5
    for step in range(ntime):
        titan_state.state.concentration = concentration
        system, rhs = kt.build_implicit_step_system(
            titan_state.state, titan_state.kzz, dt, source_terms=atm_sources
        )
        concentration = kt.solve_sparse_system(system, rhs)
        concentration = torch.clamp(concentration, min=0.0)
        if fixed_idx:
            concentration[:, :, fixed_idx] = titan_state.concentration[:, :, fixed_idx]
        if boundary_idx:
            concentration[:, 0, boundary_idx] = titan_state.concentration[:, 0, boundary_idx]
        if top_boundary_idx:
            concentration[:, last_real_lyr, top_boundary_idx] = 0.0
        if cold_trap_idx:
            concentration[:, _COLD_TRAP_LEVEL, cold_trap_idx] = (
                titan_state.concentration[:, _COLD_TRAP_LEVEL, cold_trap_idx]
            )
        dt *= growth
    return concentration[0]  # (nlyr, nspecies)


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        work = pathlib.Path(tmpdir)

        # --- run KINETICS-base ---
        print(f"[1/4] Running KINETICS-base for {NTIME} steps ...")
        kb_out = _run_kinetics_base(NTIME, work)

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
        print(f"[4/4] Running kintera for {NTIME} steps ...")
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
