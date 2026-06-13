"""Per-cell chemistry-only Newton (``chemistry_only_newton_step``).

Unlike the coupled solver, this variant ignores transport and solves
only the cell-local chemistry residual::

    F(c_new) = c_new − c0 − dt · S(c_new)   per cell

The system is dense ``(B, N, N)`` with ``B = ncol·nlyr`` and
``N = nspecies``, solved with ``torch.linalg.solve``. KB's ``MARCH``
runs in this regime (chemistry rates only; ``FLOW2D`` diffusion is
applied via a separate operator-split step).
"""

from __future__ import annotations

import torch

from ..atm_state2d import AtmState2D
from ..sources import build_source_linearization
from .result import (
    ConcentrationPostprocess,
    NewtonResult,
    per_species_relative_change,
)


def chemistry_only_newton_step(
    state: AtmState2D,
    dt: float,
    *,
    source_terms,
    concentration_postprocess: ConcentrationPostprocess | None = None,
    max_iterations: int = 100,
    convergence_tol: float = 1e-4,
    species_scale_floor: float = 1.0,
    damping_trigger: float = 1.0,
    damping_factor: float = 0.5,
    out_of_basin_threshold: float = float("inf"),
    divergence_growth_factor: float = 10.0,
    divergence_threshold: float = 1e6,
    clip_negative: bool | str = True,
    mass_conservation_cap: bool = True,
    max_concentration_cap: "torch.Tensor | None" = None,
    charge_balance_indices: "tuple[list[int], int] | None" = None,
    record_residuals: bool = False,
) -> NewtonResult:
    """Per-cell chemistry-only Newton iteration of one BE step.

    See the pre-refactor ``atm2d.newton.chemistry_only_newton_step``
    docstring for the full parameter description.
    """
    c0 = state.concentration.clone()
    c_k = c0.clone()
    best_iterate = c0
    best_max_rel = float("inf")
    residual_history: list[float] = []
    max_rel = float("inf")
    converged = False
    iters = 0

    nspecies = state.nspecies
    eye = torch.eye(nspecies, dtype=state.dtype, device=state.device)

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
        linearization = build_source_linearization(
            state, source_terms,
            charge_balance_indices=charge_balance_indices,
        )
        S_at_ck = linearization.tendency
        J_at_ck = linearization.jacobian

        # KB BMATRM/RESIDM 1/dt-scaled formulation:
        #   QV = S(c) − (c − c0)/dt
        #   B  = I/dt − J
        #   B · dc = QV
        dt_inv = 1.0 / float(dt)
        QV = S_at_ck - dt_inv * (c_k - c0)
        B = dt_inv * eye.view(1, 1, nspecies, nspecies) - J_at_ck

        A_flat = B.reshape(-1, nspecies, nspecies)
        rhs_flat = QV.reshape(-1, nspecies, 1)
        try:
            delta_c_flat = torch.linalg.solve(A_flat, rhs_flat)
        except torch._C._LinAlgError:
            iters = k + 1
            state.concentration = c0
            return NewtonResult(
                concentration=c_k,
                converged=False,
                iterations=iters,
                max_relative_change=float("inf"),
                residual_history=residual_history,
            )
        delta_c = delta_c_flat.reshape(c_k.shape)
        c_proposed = c_k + delta_c

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
            state.concentration = c0
            return NewtonResult(
                concentration=c_new,
                converged=False,
                iterations=iters,
                max_relative_change=float("inf"),
                residual_history=residual_history,
            )

        max_rel = per_species_relative_change(
            c_new, c_k, species_scale_floor=species_scale_floor
        )
        prev_max_rel = residual_history[-1] if residual_history else None
        if record_residuals:
            residual_history.append(max_rel)
        if max_rel < best_max_rel:
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


__all__ = ["chemistry_only_newton_step"]
