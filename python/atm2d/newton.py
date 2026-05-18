"""Newton inner iteration for one backward-Euler implicit step.

This is the Step 1.5 layer on top of the adaptive controller. Inside a single
``adaptive_advance`` substep, the network can be too stiff for a single frozen
linearization to give an accurate answer at the proposed dt — even when the
linear solve "succeeds" numerically the linearization error makes the candidate
state physically wrong (e.g., big negative concentrations the controller has
to reject). KINETICS-base avoids this by Newton-iterating each step:
re-evaluate the rate coefficients and Jacobian at the current iterate, solve
for a correction, repeat until per-species fractional change is below a
tolerance.

``newton_implicit_step`` implements that loop on top of
``build_implicit_step_system`` (which now accepts a fixed ``c0`` for the
backward-Euler residual while ``state.concentration`` carries the current
Newton iterate ``c_k``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from .assembly import build_implicit_step_system
from .atm_state2d import AtmState2D
from .matrix import SparseSystemMatrix
from .solver import solve_sparse_system
from .source import build_source_linearization


class _SystemPostprocess(Protocol):
    def __call__(
        self,
        system: SparseSystemMatrix,
        rhs: torch.Tensor,
    ) -> tuple[SparseSystemMatrix, torch.Tensor]: ...


class _ConcentrationPostprocess(Protocol):
    def __call__(self, concentration: torch.Tensor) -> torch.Tensor: ...


@dataclass
class NewtonResult:
    concentration: torch.Tensor
    converged: bool
    iterations: int
    max_relative_change: float
    residual_history: list[float] = field(default_factory=list)


def per_species_relative_change(
    new_conc: torch.Tensor,
    old_conc: torch.Tensor,
    *,
    species_scale_floor: float = 1.0,
) -> float:
    """Return ``max | new - old | / max(|old|, floor)`` over all species.

    This is the same family of "fractional change" check KINETICS-base uses in
    ``CONVRG``: a Newton iterate is considered converged when the per-species
    relative change drops below the tolerance. The floor prevents trace species
    near zero from dominating the test.
    """
    diff = (new_conc - old_conc).abs()
    scale = old_conc.abs().clamp(min=species_scale_floor)
    return diff.div(scale).amax().item()


def newton_implicit_step(
    state: AtmState2D,
    dt: float,
    *,
    kzz: torch.Tensor,
    source_terms,
    species_diffusion_scale: torch.Tensor | None = None,
    system_postprocess: _SystemPostprocess | None = None,
    concentration_postprocess: _ConcentrationPostprocess | None = None,
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

    Parameters
    ----------
    state:
        ``AtmState2D``. Its ``concentration`` field is the starting point
        ``c0`` of the BE step on entry. The function temporarily mutates
        ``state.concentration`` to hold the current Newton iterate ``c_k``;
        on return ``state.concentration`` is set to the final iterate.
    dt:
        Timestep size.
    kzz, source_terms, species_diffusion_scale:
        Forwarded to ``build_implicit_step_system``.
    system_postprocess:
        Optional callable ``(system, rhs) -> (system, rhs)`` to apply Dirichlet
        rows or other system-level transformations before solving. Called on
        every iteration.
    concentration_postprocess:
        Optional callable ``(c_new) -> c_new`` applied after each solve, e.g.
        to enforce boundary pins. The post-processed value is what the
        convergence check sees and what is stored as the next iterate.
    max_iterations:
        Maximum Newton iterations. If exceeded, the function returns the last
        iterate with ``converged=False``.
    convergence_tol:
        Per-species fractional-change tolerance. Convergence is declared when
        ``per_species_relative_change(c_{k+1}, c_k) < convergence_tol``.
    species_scale_floor:
        Floor used in the relative-change denominator to avoid blowing up for
        trace species near zero.
    damping_trigger, damping_factor:
        When the raw Newton step would change a species by more than
        ``damping_trigger`` (relative, with the same scale floor), the update
        is damped via ``c_new = c_k + damping_factor * (c_proposed - c_k)``.
        Default mirrors the KINETICS-base ``IDAMP=1`` behaviour
        (``damping_factor=0.5`` averaging) but only kicks in when the raw step
        is large enough to push the linearization out of its basin. Set
        ``damping_trigger=inf`` or ``damping_factor=1.0`` to disable.
    out_of_basin_threshold:
        If iteration 1 produces a relative change above this threshold, the
        iterate is considered outside Newton's basin of attraction and further
        iteration would re-linearize at a worse point. In that case the
        function returns the iteration-1 result (equivalent to a single frozen
        linearization step) and lets the outer adaptive controller decide
        whether to accept it or subdivide ``dt``. Defaults to ``1.0`` (a 100%
        per-species change).
    divergence_growth_factor:
        If the per-iteration relative change grows by more than this factor
        between consecutive iterations and is already above
        ``convergence_tol``, declare divergence and exit early with
        ``converged=False``. Default ``10.0``.
    divergence_threshold:
        Absolute threshold on the per-iteration relative change. Once the
        residual exceeds this value Newton has clearly left the basin and we
        exit with ``converged=False`` instead of wasting more iterations.
        Default ``1e3``.
    record_residuals:
        If true, populate ``NewtonResult.residual_history`` with the
        per-iteration ``max_relative_change`` values.
    """
    c0 = state.concentration.clone()
    c_k = c0
    best_iterate = c0
    best_max_rel = float("inf")
    residual_history: list[float] = []
    max_rel = float("inf")
    converged = False
    iters = 0

    # Mass-conservation cap: trust region that prevents Newton from
    # producing per-cell species concentrations larger than the local
    # atmospheric density. Without this, coupled Newton at large dt
    # diverges to NaN because the Jacobian doesn't see the implicit
    # cap on c[j] ≤ total density.
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
        # Clip negatives same as chemistry_only_newton_step: tiny rounding
        # in the global sparse solve can produce sub-floor noise that
        # otherwise trips adaptive_advance's severe_negative reject path.
        if clip_negative == "abs":
            c_new = torch.abs(c_new)
        elif clip_negative:
            c_new = torch.clamp(c_new, min=0.0)
        # Mass-conservation cap: trust region that bounds the proposed
        # iterate. Coupled Newton at large dt without this bound can
        # diverge to non-physical (and eventually NaN) solutions because
        # the Jacobian sees no upper bound on c.
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
            # iter 1 is far from c0 — re-linearizing at c_1 would land outside
            # Newton's basin of attraction and make things worse. Return the
            # frozen-J iter-1 result and let the outer controller decide.
            break

    if not converged and best_max_rel < max_rel:
        c_k = best_iterate
        max_rel = best_max_rel
    # Restore state.concentration to its entry value so the caller can run
    # acceptance checks against the original c0 (not the in-progress iterate).
    # The caller is responsible for committing the returned ``concentration``
    # if the candidate is accepted.
    state.concentration = c0
    return NewtonResult(
        concentration=c_k,
        converged=converged,
        iterations=iters,
        max_relative_change=max_rel,
        residual_history=residual_history,
    )


def chemistry_only_newton_step(
    state: AtmState2D,
    dt: float,
    *,
    source_terms,
    concentration_postprocess: _ConcentrationPostprocess | None = None,
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
    """Per-cell chemistry-only Newton iteration of one backward-Euler step.

    Unlike :func:`newton_implicit_step` which couples transport and chemistry
    into one global sparse system, this variant solves only the **cell-local**
    chemistry residual::

        F(c_new) = c_new - c0 - dt * S(c_new)  per cell

    Cells are independent. With ``ncol*nlyr`` cells and ``nspecies`` species,
    each Newton iteration is a batched ``(B, N, N)`` dense solve via
    ``torch.linalg.solve``, where ``B = ncol*nlyr`` and ``N = nspecies``.
    Transport is **not** included; the caller is expected to operator-split
    transport into a separate step (see e.g. ``build_transport_matrix``).

    The chemistry residual is per-cell and well-conditioned, so this Newton
    converges at much larger ``dt`` than the coupled-transport variant — it
    is the mechanism KINETICS-base uses (``MARCH`` operates on chemistry
    rates only, with ``FLOW2D``/diffusion applied separately).

    Parameters mirror :func:`newton_implicit_step`. ``out_of_basin_threshold``
    defaults to ``inf`` here: for chemistry-only Newton at KB's stage dts
    we expect iter 1 to be far from ``c0`` (this is normal stiff chemistry)
    and want Newton to iterate to convergence rather than bail out.

    ``clip_negative`` (default ``True``) clamps negative concentrations to
    zero after each Newton update, before the convergence check. This is
    equivalent in spirit to KB's ``CONVRG`` with ``ICNV=2``/``IDAMP=1``,
    which folds back small negatives so they don't propagate as physical
    state. Without it, trace ion/radical species accumulate tiny negatives
    (~1e-3) from the BE linearization that fall just outside
    ``default_accept``'s tolerance and force the outer controller to
    subdivide ``dt`` unnecessarily.
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

    # Mass-conservation cap: no individual species can exceed a reference
    # density. Without this, Newton at very large dt (~1e+8 s+) finds
    # non-physical fixed-points where trace species accumulate to 10–1000× the
    # local density, violating atomic conservation. Mirrors a soft version of
    # KB CONVRG's behaviour: KB doesn't let trace species explode past the
    # dominant background density either.
    #
    # When ``max_concentration_cap`` is provided (shape (ncol, nlyr, 1) or
    # (ncol, nlyr) — broadcast to (..., 1)), it's used directly as the per-cell
    # cap. This should be set to the **initial** total density so the cap
    # doesn't grow if trace species themselves grow (otherwise the cap becomes
    # self-fulfilling). When omitted, falls back to ``sum(c0)`` at this BE
    # step, which is OK at small dt but drifts at large dt.
    if max_concentration_cap is not None:
        cap = max_concentration_cap
        if cap.dim() == 2:
            cap = cap.unsqueeze(-1)
        cell_total_density = cap.to(dtype=c0.dtype, device=c0.device)
    elif mass_conservation_cap:
        cell_total_density = c0.sum(dim=-1, keepdim=True)  # (ncol, nlyr, 1)
    else:
        cell_total_density = None

    for k in range(max_iterations):
        state.concentration = c_k
        linearization = build_source_linearization(
            state, source_terms,
            charge_balance_indices=charge_balance_indices,
        )
        S_at_ck = linearization.tendency  # (ncol, nlyr, nspecies)
        J_at_ck = linearization.jacobian  # (ncol, nlyr, nspecies, nspecies)

        # Backward-Euler residual per cell, scaled by 1/dt (KB BMATRM/RESIDM
        # formulation). This keeps matrix and RHS magnitudes O(|J|) and
        # O(|S|+|c0-c|/dt) instead of O(dt*|J|), which matters at large dt:
        # for J eigenvalues ~ -4e+6 and dt=1e+5, the dt-scaled form has
        # condition number ~ 1e+13 and loses 13 digits of precision; the
        # 1/dt-scaled form has cond ~ 1e+9 and is solvable.
        # QV = S(c) - (c - c0)/dt
        # B  = I/dt - J
        # B * dc = QV  (equivalent to (I - dt*J) dc = -F = dt*QV)
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
            # KB CONVRG ICNV=2 behavior: replace c<0 with |c|. This is the
            # equivalent of "mirror" reflection that keeps c positive without
            # erasing magnitude information (unlike clip-to-zero).
            c_new = torch.abs(c_new)
        elif clip_negative:
            c_new = torch.clamp(c_new, min=0.0)

        # Mass-conservation cap: no individual species can exceed the total
        # cell density. Without this, Newton at very large dt finds non-physical
        # fixed-points where trace species accumulate to 10–1000× the local
        # density.
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
