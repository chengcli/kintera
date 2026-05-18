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

    network_mode = os.environ.get("KINTERA_TITAN_NETWORK_MODE", "no_grain")
    if network_mode == "no_grain":
        source_terms = [
            term for term in source_terms
            if not any(_is_grain_related(name) for name in term.reactants + term.products)
        ]
        print(f"[setup] no_grain mode: filtered to {len(source_terms)} non-grain terms")
    elif network_mode == "full":
        print(f"[setup] full mode: all {len(source_terms)} terms (ion + grain)")
    else:
        raise ValueError(f"unknown KINTERA_TITAN_NETWORK_MODE: {network_mode}")


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

    # Atomic-budget projection: per-column atom counts for N, C, H, O at t=0.
    # After each chemistry step we rescale non-fixed atom-bearing species so
    # totals stay near initial — this enforces hard mass conservation even
    # when the Newton at very-large dt finds a non-physical fixed-point that
    # creates atoms. Fixed species (N2, CH4 reservoir at low alt via cold trap,
    # M, JDUST, etc.) are excluded from rescaling — they're our atom source.
    import re as _re

    def _count_atom(name, element):
        # Drop charge marker.
        n = name.rstrip("+-*")
        cnt = 0
        i = 0
        while i < len(n):
            ch = n[i]
            if ch == "(":  # skip parenthesized state, e.g., (1)CH2, (2D).
                while i < len(n) and n[i] != ")":
                    i += 1
                i += 1
                continue
            # Match element starting here (single-letter or 2-letter capitalized)
            if ch == element and (i + 1 >= len(n) or not n[i + 1].islower() or
                                  element == "C" and n[i:i+2] == "Cl"):
                # Handle 2-letter elements like 'Cl' explicitly if needed
                pass
            if n[i:i+len(element)] == element:
                # Confirm this is element not part of a multichar element name
                end = i + len(element)
                if end < len(n) and n[end].islower():
                    # part of a multichar name like "Cl", "Br" — skip
                    i += 1
                    continue
                # Look for trailing digit
                if end < len(n) and n[end].isdigit():
                    cnt += int(n[end])
                    i = end + 1
                else:
                    cnt += 1
                    i = end
            else:
                i += 1
        return cnt

    fixed_indices = [titan_state.species.index(n) for n in titan_state.fixed_species
                     if n in titan_state.species]
    fixed_mask = torch.zeros(len(titan_state.species), dtype=torch.bool, device=concentration.device)
    fixed_mask[fixed_indices] = True

    # (nspecies,) integer atom-count tensors per element.
    atom_counts = {}
    for elem in ["N", "C", "H", "O"]:
        counts = torch.tensor(
            [_count_atom(name, elem) for name in titan_state.species],
            dtype=concentration.dtype, device=concentration.device,
        )
        atom_counts[elem] = counts

    # Per-cell initial atom counts (i.e., not column-integrated; rescaling
    # is per-cell to preserve spatial profile).
    initial_atoms_per_cell = {
        elem: (concentration * counts).sum(dim=-1, keepdim=True)  # (1, nlyr, 1)
        for elem, counts in atom_counts.items()
    }

    # Per-element budget = initial variable atoms (≈0 for most elements here)
    # PLUS a chemistry headroom = ATOM_HEADROOM_FRACTION × fixed-species atoms.
    # This lets photo-chemistry build up trace species up to ~headroom% of the
    # reservoir over the run; KB's NT=50 trace N column is ~5e+18 vs N2's
    # 8.45e+21, so a headroom of 1e-3 captures KB's typical maximum. Tighter
    # = stricter conservation but kills slow chemistry chains; looser = closer
    # to current 1.28× violation.
    ATOM_HEADROOM = float(os.environ.get("KINTERA_ATOM_HEADROOM", "1e-3"))
    fixed_atoms_per_cell = {
        elem: (concentration * counts * fixed_mask).sum(dim=-1, keepdim=True)
        for elem, counts in atom_counts.items()
    }
    initial_variable_atoms = {
        elem: (initial_atoms_per_cell[elem] - fixed_atoms_per_cell[elem]).clamp(min=0.0)
        for elem in atom_counts
    }
    variable_atom_budget = {
        elem: initial_variable_atoms[elem] + ATOM_HEADROOM * fixed_atoms_per_cell[elem]
        for elem in atom_counts
    }

    nonfixed_mask = (~fixed_mask).view(1, 1, -1)

    # Cation indices and E index for implicit charge-balance Jacobian
    # coupling. When passed to chemistry_only_newton_step, each species
    # row's Jacobian gets dS/dc_E added to the cation columns — propagating
    # the implicit constraint E = Σ(cations) directly into the Newton
    # solve instead of via lagged Picard iteration through apply_pins.
    species_list = titan_state.species
    cation_indices = [j for j, n in enumerate(species_list) if n.endswith("+")]
    e_index = species_list.index("E") if "E" in species_list else None
    if e_index is not None and cation_indices:
        charge_balance_indices = (cation_indices, e_index)
    else:
        charge_balance_indices = None
    CHARGE_BALANCE_JACOBIAN = (
        os.environ.get("KINTERA_CHARGE_BALANCE_JACOBIAN", "1") == "1"
    )
    _charge_arg = charge_balance_indices if CHARGE_BALANCE_JACOBIAN else None

    def _project_atomic_budget(c):
        """Per-cell, per-element atomic conservation: rescale ONLY the species
        that are above their fair share of the budget, leaving smaller species
        alone.

        For each element with current variable atoms > budget at a cell:
        1. Find species containing that element that are > median (the hogs).
        2. Compute the excess (current - budget).
        3. Subtract excess proportionally from hog species' atom contribution.

        This preserves slow-chemistry species (which stay small) while clipping
        the runaway species that are eating the budget.
        """
        for elem, counts in atom_counts.items():
            mask = (counts > 0) & (~fixed_mask)
            if not mask.any():
                continue
            counts_view = counts.view(1, 1, -1)
            mask_view = mask.view(1, 1, -1)
            # Per-cell current atoms vs budget.
            cur = (c * counts_view * mask).sum(dim=-1, keepdim=True)
            budget = variable_atom_budget[elem]
            excess = (cur - budget).clamp(min=0.0)
            if not (excess > 0).any():
                continue
            # Identify hog species: those whose atom-contribution exceeds the
            # mean per-species contribution. Subtract excess proportionally
            # from hogs only.
            per_species_atoms = c * counts_view * mask  # (1, nlyr, nspecies)
            # Per-cell mean across non-fixed atom species
            n_species_with_elem = mask.sum().clamp(min=1.0).item()
            mean_atoms = per_species_atoms.sum(dim=-1, keepdim=True) / n_species_with_elem
            hog_mask = per_species_atoms > mean_atoms  # (1, nlyr, nspecies)
            hog_atoms = torch.where(hog_mask, per_species_atoms, torch.zeros_like(per_species_atoms))
            hog_total = hog_atoms.sum(dim=-1, keepdim=True).clamp(min=1.0)
            # Fraction of excess each hog needs to give up.
            hog_fraction = hog_atoms / hog_total  # how much of the excess this hog absorbs
            # Atoms to subtract from each hog species at each cell.
            atoms_to_subtract = hog_fraction * excess
            # Translate atoms-to-subtract back into concentration delta:
            # delta_c[j] = atoms_subtract[j] / counts[j]
            # But avoid divide-by-zero where counts=0 (mask handles it).
            counts_safe = counts_view.clamp(min=1.0)
            delta_c = atoms_to_subtract / counts_safe
            # Apply only where hog_mask and excess>0.
            apply_mask = hog_mask & (excess > 0).expand_as(hog_mask)
            c = torch.where(apply_mask, (c - delta_c).clamp(min=0.0), c)
        return c

    def _apply_dirichlet(system, rhs):
        return kt.apply_kinetics_base_titan_dirichlet_rows(system, rhs, titan_state)

    def _apply_pins(new_conc):
        kt.apply_kinetics_base_titan_boundary_pins(new_conc, titan_state)
        return new_conc

    ATOMIC_PROJECTION = os.environ.get("KINTERA_ATOMIC_PROJECTION", "1") == "1"

    newton_stats = {"runs": 0, "iters": 0, "non_converged": 0, "max_iters_in_step": 0}
    NEWTON_DEBUG = os.environ.get("KINTERA_NEWTON_DEBUG", "0") == "1"
    SOLVER_MODE = os.environ.get("KINTERA_SOLVER_MODE", "split")

    def step_fn_coupled(state, dt):
        """Coupled-Newton step (transport + chemistry in one inner iter).

        Set ``KINTERA_SOLVER_MODE=coupled`` to use this path; default ``split``.

        Atomic projection is NOT applied in coupled mode: experimentation
        showed coupled Newton re-equilibrates around the projected state
        and finds worse fixed-points than without (e.g. C+ at lev 30 went
        from 3.8e+3 to 4.3e+6 when projection was added). The coupled
        solver's own mass cap is the only post-Newton trust region used.
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

    # Sub-cycling: at outer dt > SUBCYCLE_THRESHOLD, internally subdivide
    # the operator-split step into multiple smaller sub-steps. This stops
    # transport from over-mixing across many scale heights in one step
    # (which causes the cation cascade between NT=40 and NT=50 — see
    # trajectory analysis).
    #
    # OFF BY DEFAULT: sub-cycling is mathematically equivalent to lowering
    # dt_max in the schedule. Compute cost scales linearly with outer
    # dt / sub_dt, so at large dt the run becomes impractically slow
    # (NTIME=100 schedule reaches dt=3e+8 by step ~50 — with sub_threshold
    # 1e+6 that's 300 sub-cycles per outer step). To get the same physics
    # at lower cost, just lower max_dt in the schedule. Sub-cycling here
    # is preserved as opt-in infrastructure.
    SUBCYCLE_THRESHOLD = float(os.environ.get("KINTERA_SUBCYCLE_THRESHOLD", "1e+6"))
    SUBCYCLE_ENABLED = os.environ.get("KINTERA_SUBCYCLE", "0") == "1"

    def _one_split_substep(state, dt_sub):
        """One transport-then-chemistry step at dt_sub."""
        transport_system, transport_rhs = kt.build_implicit_step_system(
            state,
            titan_state.kzz,
            dt_sub,
            species_diffusion_scale=species_diffusion_scale,
            source_terms=None,
        )
        transport_system, transport_rhs = _apply_dirichlet(transport_system, transport_rhs)
        c_after_transport = kt.solve_sparse_system(transport_system, transport_rhs)
        _apply_pins(c_after_transport)
        state.concentration = c_after_transport
        return kt.chemistry_only_newton_step(
            state,
            dt_sub,
            source_terms=atm_sources,
            concentration_postprocess=_apply_pins,
            max_iterations=NEWTON_MAX_ITER,
            convergence_tol=NEWTON_TOL,
            damping_factor=NEWTON_DAMP_FACTOR,
            damping_trigger=NEWTON_DAMP_TRIGGER,
            clip_negative=_clip_arg,
            charge_balance_indices=_charge_arg,
            record_residuals=NEWTON_DEBUG,
        )

    def step_fn_split(state, dt):
        """Operator-split: transport-only BE step, then chemistry-only Newton.

        Mirrors KINETICS-base which does ``FLOW2D``/diffusion separately from
        the ``MARCH`` chemistry Newton. Each cell's chemistry residual is
        independent so Newton converges at much larger ``dt`` than the
        fully-coupled variant.

        When ``KINTERA_SUBCYCLE=1`` (default) and ``dt > SUBCYCLE_THRESHOLD``,
        the outer step is internally divided into N sub-cycles of dt/N each
        so vertical transport doesn't mix species across many scale heights
        per step. The chemistry Newton sees each sub-step independently.
        """
        pristine_c0 = state.concentration.clone()

        if SUBCYCLE_ENABLED and dt > SUBCYCLE_THRESHOLD:
            n_subcycles = max(2, int(dt / SUBCYCLE_THRESHOLD))
            dt_sub = dt / n_subcycles
            chem_result = None
            agg_iters = 0
            agg_conv = True
            for _ in range(n_subcycles):
                chem_result = _one_split_substep(state, dt_sub)
                if ATOMIC_PROJECTION:
                    projected = _project_atomic_budget(chem_result.concentration)
                    projected = _apply_pins(projected)
                    chem_result = type(chem_result)(
                        concentration=projected,
                        converged=chem_result.converged,
                        iterations=chem_result.iterations,
                        max_relative_change=chem_result.max_relative_change,
                        residual_history=chem_result.residual_history,
                    )
                agg_iters += chem_result.iterations
                agg_conv = agg_conv and chem_result.converged
                state.concentration = chem_result.concentration
            assert chem_result is not None
            # Return aggregate result.
            final = type(chem_result)(
                concentration=chem_result.concentration,
                converged=agg_conv,
                iterations=agg_iters,
                max_relative_change=chem_result.max_relative_change,
                residual_history=chem_result.residual_history,
            )
            state.concentration = pristine_c0
            return final

        chem_result = _one_split_substep(state, dt)
        if ATOMIC_PROJECTION:
            projected = _project_atomic_budget(chem_result.concentration)
            projected = _apply_pins(projected)
            chem_result = type(chem_result)(
                concentration=projected,
                converged=chem_result.converged,
                iterations=chem_result.iterations,
                max_relative_change=chem_result.max_relative_change,
                residual_history=chem_result.residual_history,
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
