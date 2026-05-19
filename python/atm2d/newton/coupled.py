"""Coupled transport + chemistry Newton (the original
``newton_implicit_step``). Solves one BE step where transport and
chemistry are both inside the same global sparse system.
"""

from __future__ import annotations

import torch

from ..assembly import build_implicit_step_system
from ..atm_state2d import AtmState2D
from ..solver import solve_sparse_system
from .result import (
    ConcentrationPostprocess,
    NewtonResult,
    SystemPostprocess,
    per_species_relative_change,
)


def newton_implicit_step(
    state: AtmState2D,
    dt: float,
    *,
    kzz: torch.Tensor,
    source_terms,
    species_diffusion_scale: torch.Tensor | None = None,
    system_postprocess: SystemPostprocess | None = None,
    concentration_postprocess: ConcentrationPostprocess | None = None,
    max_iterations: int = 100,
    convergence_tol: float = 1e-4,
    species_scale_floor: float = 1.0,
    damping_trigger: float = 1.0,
    damping_factor: float = 0.5,
    out_of_basin_threshold: float = float("inf"),
    divergence_growth_factor: float = float("inf"),
    divergence_threshold: float = float("inf"),
    clip_negative: bool | str = True,
    mass_conservation_cap: bool = True,
    max_concentration_cap: "torch.Tensor | None" = None,
    charge_balance_indices: "tuple[list[int], int] | None" = None,
    record_residuals: bool = False,
) -> NewtonResult:
    """Run Newton iteration on one backward-Euler step of size ``dt``.

    See :func:`atm2d.newton.coupled.newton_implicit_step` for the full
    parameter docstring (preserved from the pre-refactor ``newton.py``).
    """
    c0 = state.concentration.clone()
    c_k = c0
    best_iterate = c0
    best_max_rel = float("inf")
    residual_history: list[float] = []
    max_rel = float("inf")
    converged = False
    iters = 0

    if max_concentration_cap is not None:
        cap = max_concentration_cap
        if cap.dim() == 2:
            cap = cap.unsqueeze(-1)
        cell_total_density = cap.to(dtype=c0.dtype, device=c0.device)
    elif mass_conservation_cap:
        cell_total_density = c0.sum(dim=-1, keepdim=True)
    else:
        cell_total_density = None

    for k in range(max_iterations):
        state.concentration = c_k
        system, rhs = build_implicit_step_system(
            state,
            kzz,
            dt,
            species_diffusion_scale=species_diffusion_scale,
            source_terms=source_terms,
            c0=c0,
            charge_balance_indices=charge_balance_indices,
        )
        if system_postprocess is not None:
            system, rhs = system_postprocess(system, rhs)
        c_proposed = solve_sparse_system(system, rhs)
        if damping_factor < 1.0:
            raw_change = per_species_relative_change(
                c_proposed, c_k, species_scale_floor=species_scale_floor
            )
            if raw_change > damping_trigger:
                c_new = c_k + damping_factor * (c_proposed - c_k)
            else:
                c_new = c_proposed
        else:
            c_new = c_proposed
        if clip_negative == "abs":
            c_new = torch.abs(c_new)
        elif clip_negative:
            c_new = torch.clamp(c_new, min=0.0)
        if cell_total_density is not None:
            c_new = torch.minimum(c_new, cell_total_density)
        if concentration_postprocess is not None:
            c_new = concentration_postprocess(c_new)
        iters = k + 1
        if not torch.isfinite(c_new).all():
            state.concentration = c_new
            max_rel = float("inf")
            if record_residuals:
                residual_history.append(max_rel)
            return NewtonResult(
                concentration=c_new,
                converged=False,
                iterations=iters,
                max_relative_change=max_rel,
                residual_history=residual_history,
            )
        max_rel = per_species_relative_change(
            c_new, c_k, species_scale_floor=species_scale_floor
        )
        prev_max_rel = residual_history[-1] if residual_history else None
        if record_residuals:
            residual_history.append(max_rel)
        if torch.isfinite(c_new).all() and max_rel < best_max_rel:
            best_iterate = c_new
            best_max_rel = max_rel
        c_k = c_new
        if max_rel < convergence_tol:
            converged = True
            break
        if max_rel > divergence_threshold:
            break
        if (
            prev_max_rel is not None
            and max_rel > convergence_tol
            and max_rel > divergence_growth_factor * prev_max_rel
        ):
            break
        if k == 0 and max_rel > out_of_basin_threshold:
            break

    if not converged and best_max_rel < max_rel:
        c_k = best_iterate
        max_rel = best_max_rel
    state.concentration = c0
    return NewtonResult(
        concentration=c_k,
        converged=converged,
        iterations=iters,
        max_relative_change=max_rel,
        residual_history=residual_history,
    )


__all__ = ["newton_implicit_step"]
