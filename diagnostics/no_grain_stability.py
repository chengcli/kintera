"""kintera-only no_grain stability check.

Runs the kintera Titan no-grain integration through the adaptive timestep
controller. Verifies the final concentration is finite and not blown up.
Does not depend on the KINETICS-base Fortran binary, so it can run on Linux
without a KB checkout.

Usage:
    KINTERA_TITAN_NTIME=50 python diagnostics/no_grain_stability.py
"""

from __future__ import annotations

import math
import os
import pathlib

import torch

import kintera as kt

DEFAULT_ROOT = pathlib.Path(__file__).resolve().parent / "KINETICS-base-compare"
ROOT = pathlib.Path(os.environ.get("KINTERA_KINETICS_BASE_ROOT", DEFAULT_ROOT))
TITAN_DIR = ROOT / "examples" / "titan"
PUN_PATH = TITAN_DIR / "kindata_yy_clean" / "Cheng_ions_c6h7+_v3_H2CN.pun"
RUN_INPUT_PATH = TITAN_DIR / "ions_c6h7+_H2CN.inp-1"
ATMOSPHERE_PATH = TITAN_DIR / "kintitan.pun_zero_conc_2_mod_atm_orig_3xkzz"

NTIME = int(os.environ.get("KINTERA_TITAN_NTIME", "50"))
MAX_SUBDIVISIONS = int(os.environ.get("KINTERA_TITAN_MAX_SUBDIV", "20"))
# Newton parameters mirror KB's MARCH: ITER=8 max inner iterations, partial
# step (damp=0.3) so we don't fully converge to the non-physical fixed-point
# branch at large dt. With these together kintera reproduces KB's mid-altitude
# chemistry (HCN, C2H6, CH3 lev 15-35) within 1-2x; without them (max_iter=30,
# damp=1.0) Newton finds a runaway ion solution at lev 0-15 around dt~1e+5 s.
NEWTON_MAX_ITER = int(os.environ.get("KINTERA_NEWTON_MAX_ITER", "8"))
# Behavior on Newton iterates that go negative. KB's CONVRG with ICNV=2 takes
# ABS() (reflects back positive). Set to "abs" to mirror KB; default is
# clip-to-zero (less aggressive).
NEWTON_CLIP_NEG = os.environ.get("KINTERA_NEWTON_CLIP_NEGATIVE", "clip")
_clip_arg = "abs" if NEWTON_CLIP_NEG == "abs" else (NEWTON_CLIP_NEG != "false")
NEWTON_TOL = float(os.environ.get("KINTERA_NEWTON_TOL", "1e-4"))
NEWTON_DAMP_FACTOR = float(os.environ.get("KINTERA_NEWTON_DAMP_FACTOR", "0.3"))
NEWTON_DAMP_TRIGGER = float(os.environ.get("KINTERA_NEWTON_DAMP_TRIGGER", "0.5"))
SCHEDULE = os.environ.get("KINTERA_TITAN_SCHEDULE", "kb")
# Photochemical QSS initialization (multi-stage chemistry-only Newton at growing
# dt) was useful when the legacy schedule capped at dt=10800s. With the fixed
# KB schedule (NCYCLE=2, no stage cap) the natural ramp from dt=1e-15 already
# covers chemistry warmup; QSS init is redundant and tends to over-shoot fast
# species at lev 0-15. Default off; set KINTERA_TITAN_QSS_INIT_DT > 0 to enable.
QSS_INIT_DT = float(os.environ.get("KINTERA_TITAN_QSS_INIT_DT", "0"))
QSS_INIT_MAX_ITER = int(os.environ.get("KINTERA_TITAN_QSS_INIT_MAX_ITER", "100"))


def _fixed_timestep_sequence(ntime):
    """Legacy diagnostic schedule (1e-15 × √10/step). Not KB-equivalent.

    Kept as ``KINTERA_TITAN_SCHEDULE=legacy_sqrt10`` for backward comparison;
    the default ``kb`` schedule mirrors the KINETICS-base ``DELTIM=-1e-15``
    ramp with ``NCYCLE=10`` (factor 10^0.1 per step).
    """
    dt = 1.0e-15
    growth = 10 ** 0.5
    seq = []
    for _ in range(ntime):
        seq.append(dt)
        dt *= growth
    return seq


def _kb_timestep_sequence(ntime):
    """KB Titan stage-based schedule (mirrors DELTIM=-1e-15, NCYCLE=10)."""
    return kt.kinetics_base_titan_dt_schedule(ntime)


def _build_sequence(ntime):
    if SCHEDULE == "kb":
        return _kb_timestep_sequence(ntime)
    if SCHEDULE == "legacy_sqrt10":
        return _fixed_timestep_sequence(ntime)
    raise ValueError(f"unknown schedule {SCHEDULE}; use kb or legacy_sqrt10")


def _is_grain_related(name):
    return name in {"SGA", "U"} or name.startswith("G")


def main():
    print(f"[setup] loading Titan atmosphere from {ATMOSPHERE_PATH}")
    initial = kt.parse_kinetics_base_atmosphere(str(ATMOSPHERE_PATH))
    species = list(initial.species_profiles.keys())
    print(f"[setup] {len(species)} species")

    titan_state = kt.build_kinetics_base_titan_state(
        initial,
        species=species,
        boundary_path=str(TITAN_DIR / "titan_Cheng_N_ions_H2CN.bc_save"),
        pun_path=str(PUN_PATH),
    )

    print("[setup] building source terms ...")
    source_terms, pun_metadata = (
        kt.build_kinetics_base_titan_source_terms(
            str(PUN_PATH),
            special_path=str(TITAN_DIR / "kindata_yy_clean" / "Cheng_ions_c6h7+_v3_H2CN.special"),
            boundary_path=str(TITAN_DIR / "titan_Cheng_N_ions_H2CN.bc_save"),
            run_input_path=str(RUN_INPUT_PATH),
            photo_catalog_path=str(TITAN_DIR / "Cheng_catalog_v4.dat"),
            cross_dir=str(TITAN_DIR / "Cheng_cross"),
            flux_path=str(TITAN_DIR / "flare_kin_oct2003.inp"),
        ),
        kt.kinetics_base_species_metadata_from_pun(str(PUN_PATH)),
    )

    source_terms = [
        term for term in source_terms
        if not any(_is_grain_related(name) for name in term.reactants + term.products)
    ]

    atm_sources = kt.build_kinetics_base_titan_atm2d_source_terms(
        titan_state, source_terms, pun_metadata=pun_metadata
    )
    species_diffusion_scale = kt.kinetics_base_titan_species_diffusion_scale(
        titan_state.species,
        dtype=titan_state.state.dtype,
        device=titan_state.state.device,
    )

    concentration = titan_state.concentration.clone()
    print(f"[setup] initial concentration shape: {tuple(concentration.shape)}")
    print(f"[setup] initial max value: {concentration.max().item():.3e}")

    def _apply_dirichlet(system, rhs):
        return kt.apply_kinetics_base_titan_dirichlet_rows(system, rhs, titan_state)

    def _apply_pins(new_conc):
        kt.apply_kinetics_base_titan_boundary_pins(new_conc, titan_state)
        return new_conc

    newton_stats = {"runs": 0, "iters": 0, "non_converged": 0, "max_iters_in_step": 0}
    NEWTON_DEBUG = os.environ.get("KINTERA_NEWTON_DEBUG", "0") == "1"
    SOLVER_MODE = os.environ.get("KINTERA_SOLVER_MODE", "split")

    def step_fn_coupled(state, dt):
        """Original coupled-Newton step (transport + chemistry in one inner iter).

        Set ``KINTERA_SOLVER_MODE=coupled`` to use this path; default ``split``.
        """
        return kt.newton_implicit_step(
            state,
            dt,
            kzz=titan_state.kzz,
            source_terms=atm_sources,
            species_diffusion_scale=species_diffusion_scale,
            system_postprocess=_apply_dirichlet,
            concentration_postprocess=_apply_pins,
            max_iterations=NEWTON_MAX_ITER,
            convergence_tol=NEWTON_TOL,
            damping_factor=NEWTON_DAMP_FACTOR,
            damping_trigger=NEWTON_DAMP_TRIGGER,
            record_residuals=NEWTON_DEBUG,
        )

    def step_fn_split(state, dt):
        """Operator-split: transport-only BE step, then chemistry-only Newton.

        Mirrors KINETICS-base which does ``FLOW2D``/diffusion separately from
        the ``MARCH`` chemistry Newton. Each cell's chemistry residual is
        independent so Newton converges at much larger ``dt`` than the
        fully-coupled variant.
        """
        pristine_c0 = state.concentration.clone()
        # Step 1: transport-only backward Euler.
        transport_system, transport_rhs = kt.build_implicit_step_system(
            state,
            titan_state.kzz,
            dt,
            species_diffusion_scale=species_diffusion_scale,
            source_terms=None,
        )
        transport_system, transport_rhs = _apply_dirichlet(transport_system, transport_rhs)
        c_after_transport = kt.solve_sparse_system(transport_system, transport_rhs)
        _apply_pins(c_after_transport)
        # Step 2: per-cell chemistry Newton with c_after_transport as BE start.
        state.concentration = c_after_transport
        chem_result = kt.chemistry_only_newton_step(
            state,
            dt,
            source_terms=atm_sources,
            concentration_postprocess=_apply_pins,
            max_iterations=NEWTON_MAX_ITER,
            convergence_tol=NEWTON_TOL,
            damping_factor=NEWTON_DAMP_FACTOR,
            damping_trigger=NEWTON_DAMP_TRIGGER,
            clip_negative=_clip_arg,
            record_residuals=NEWTON_DEBUG,
        )
        state.concentration = pristine_c0
        return chem_result

    def step_fn(state, dt):
        if SOLVER_MODE == "coupled":
            result = step_fn_coupled(state, dt)
        elif SOLVER_MODE == "split":
            result = step_fn_split(state, dt)
        else:
            raise ValueError(f"unknown KINTERA_SOLVER_MODE={SOLVER_MODE}; use split or coupled")
        newton_stats["runs"] += 1
        newton_stats["iters"] += result.iterations
        if result.iterations > newton_stats["max_iters_in_step"]:
            newton_stats["max_iters_in_step"] = result.iterations
        if not result.converged:
            newton_stats["non_converged"] += 1
        if NEWTON_DEBUG:
            c = result.concentration
            old = state.concentration
            old_max = old.abs().amax(dim=tuple(range(old.dim() - 1))).clamp(min=1.0)
            min_val = c.min().item()
            # find species with worst negative
            neg_ratios = c.amin(dim=tuple(range(c.dim() - 1))) / old_max
            worst_neg_species = int(neg_ratios.argmin().item())
            worst_neg_ratio = float(neg_ratios.min().item())
            print(
                f"    [newton] dt={dt:.3e} iters={result.iterations} "
                f"converged={result.converged} "
                f"max_rel={result.max_relative_change:.3e} "
                f"min={min_val:.2e} worst_neg_ratio={worst_neg_ratio:.2e} "
                f"sp={worst_neg_species}",
                flush=True,
            )
        return result.concentration

    if QSS_INIT_DT > 0:
        # Multi-stage QSS: geometrically grow dt so each stage starts from the
        # previous stage's near-equilibrium state. Fast species settle at small
        # dt; slower chemistry chains propagate at larger dt. This mirrors
        # KB's apparent initialization to photochemical equilibrium.
        # Final stages with very large dt push the chemistry Newton to solve
        # the steady-state condition S(c)=0 (the dt*S term dominates), matching
        # KB's converged photochemical equilibrium.
        stages = [1.0, 60.0, 3600.0, 86400.0, QSS_INIT_DT]
        stages = sorted({s for s in stages if 0 < s <= QSS_INIT_DT})
        before_all = concentration.clone()
        print(f"[qss-init] multi-stage QSS: dt sequence = {[f'{s:.1e}' for s in stages]}, "
              f"max_iter per stage = {QSS_INIT_MAX_ITER}")
        for stage_dt in stages:
            titan_state.state.concentration = concentration
            qss_result = kt.chemistry_only_newton_step(
                titan_state.state,
                stage_dt,
                source_terms=atm_sources,
                concentration_postprocess=_apply_pins,
                max_iterations=QSS_INIT_MAX_ITER,
                convergence_tol=NEWTON_TOL,
                damping_factor=NEWTON_DAMP_FACTOR,
                damping_trigger=NEWTON_DAMP_TRIGGER,
            )
            concentration = qss_result.concentration
            print(f"[qss-init]   dt={stage_dt:.2e}s converged={qss_result.converged} "
                  f"iters={qss_result.iterations} max_rel={qss_result.max_relative_change:.3e}")
        # Diagnostic: which species moved the most across all QSS stages
        delta = (concentration - before_all).abs()
        species_max = before_all.abs().amax(dim=tuple(range(before_all.dim() - 1))).clamp(min=1.0)
        delta_max = delta.amax(dim=tuple(range(delta.dim() - 1)))
        rel_change = delta_max / species_max
        top_changed = rel_change.topk(min(10, len(rel_change))).indices.tolist()
        print(f"[qss-init] top-10 species that moved most (cumulative across stages):")
        for sp_idx in top_changed:
            sp = species[int(sp_idx)]
            old_max = float(before_all[..., int(sp_idx)].max().item())
            new_max = float(concentration[..., int(sp_idx)].max().item())
            print(f"           {sp:<14s} max: {old_max:.3e} -> {new_max:.3e}")

        # Dump the post-QSS state if requested. KB's NTIME=50 oracle is at
        # ~1 ns of physical time, essentially their QSS state — so post-QSS
        # kintera ≈ KB oracle in the matched-time sense.
        qss_dump_path = os.environ.get("KINTERA_TITAN_QSS_DUMP")
        if qss_dump_path:
            import numpy as np
            alt = np.asarray(initial.altitude, dtype=np.float64)
            density = np.asarray(initial.density, dtype=np.float64)
            conc_np = concentration.detach().cpu().numpy().squeeze(0)
            np.savez(
                qss_dump_path,
                species=np.array(species),
                altitude_km=alt,
                density=density,
                concentration=conc_np,
                ntime=0,
                schedule="post-qss-only",
                solver_mode="qss",
                dt_sequence=np.asarray(stages, dtype=np.float64),
                total_simulated_time=0.0,
            )
            print(f"[qss-init] dumped post-QSS state to {qss_dump_path}")

    seq = _build_sequence(NTIME)
    total_accept = 0
    total_reject = 0
    print(f"[run] NTIME={NTIME}, schedule={SCHEDULE}, max_subdivisions={MAX_SUBDIVISIONS}")
    print(f"[run] dt sequence: {seq[0]:.2e} .. {seq[-1]:.2e}, total={sum(seq):.2e} s")
    last_accepted_dt = None
    for k, dt_target in enumerate(seq):
        titan_state.state.concentration = concentration
        try:
            result = kt.adaptive_advance(
                titan_state.state,
                dt_target,
                step_fn,
                max_subdivisions=MAX_SUBDIVISIONS,
                record_trace=True,
                initial_attempt=last_accepted_dt,
            )
        except RuntimeError as e:
            print(f"  step {k+1:>3d}/{NTIME}: dt_target={dt_target:.3e} -> FLOOR HIT: {e}")
            return 1
        concentration = result.concentration
        total_accept += result.accepted_steps
        total_reject += result.rejected_steps
        if result.rejected_steps == 0:
            last_accepted_dt = None
        elif result.max_accepted_dt > 0:
            last_accepted_dt = result.max_accepted_dt
        finite = bool(torch.isfinite(concentration).all().item())
        max_abs = concentration.abs().max().item()
        min_val = concentration.min().item()
        marker = "" if finite else " NON_FINITE!"
        reasons = {}
        for rec in result.records:
            if rec.action.startswith("reject_"):
                reasons[rec.action] = reasons.get(rec.action, 0) + 1
        reasons_str = ", ".join(f"{r}={n}" for r, n in reasons.items()) or "-"
        print(
            f"  step {k+1:>3d}/{NTIME}: dt_target={dt_target:.3e} "
            f"acc={result.accepted_steps} rej={result.rejected_steps} "
            f"last_dt={result.last_accepted_dt:.3e} "
            f"max|c|={max_abs:.3e} min(c)={min_val:.3e} reasons=[{reasons_str}]{marker}"
        )
        if not finite:
            print(f"  -> integration produced non-finite values at step {k+1}; abort.")
            return 1

    print()
    print(f"[done] totals: accepted={total_accept} rejected={total_reject}")
    print(
        f"[done] newton: runs={newton_stats['runs']} "
        f"total_iters={newton_stats['iters']} "
        f"non_converged={newton_stats['non_converged']} "
        f"avg_iter_per_run={newton_stats['iters']/max(newton_stats['runs'],1):.2f} "
        f"max_iters_in_step={newton_stats['max_iters_in_step']}"
    )
    print(f"[done] final max abs concentration: {concentration.abs().max().item():.3e}")
    print(f"[done] final min concentration:    {concentration.min().item():.3e}")

    if "C6N2" in species:
        i = species.index("C6N2")
        print(f"[done] C6N2 max: {concentration[..., i].max().item():.3e}")
    if "H2" in species:
        i = species.index("H2")
        print(f"[done] H2   max: {concentration[..., i].max().item():.3e}")
    if "C2H3CN" in species:
        i = species.index("C2H3CN")
        print(f"[done] C2H3CN max: {concentration[..., i].max().item():.3e}")

    dump_path = os.environ.get("KINTERA_TITAN_DUMP")
    if dump_path:
        import numpy as np
        alt = np.asarray(initial.altitude, dtype=np.float64)
        density = np.asarray(initial.density, dtype=np.float64)
        conc_np = concentration.detach().cpu().numpy().squeeze(0)  # (nlyr, nspecies)
        np.savez(
            dump_path,
            species=np.array(species),
            altitude_km=alt,
            density=density,
            concentration=conc_np,
            ntime=NTIME,
            schedule=SCHEDULE,
            solver_mode=SOLVER_MODE,
            dt_sequence=np.asarray(seq, dtype=np.float64),
            total_simulated_time=float(sum(seq)),
        )
        print(f"[done] dumped final state to {dump_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
