"""Second-order Rosenbrock (Ros2) step for coupled transport + chemistry.

Ros2 is *linearly implicit*: it takes two linear solves against a frozen
Jacobian and has no iteration to converge. That matters for stiff
photochemistry columns, where backward Euler + Newton can stall. On a
100-layer Mars CO2 column the Newton path tops out around
``dt ~ 300 s`` and *falls* from there (rejections cancel the step-size
growth), which puts a steady state months of wall time away; VULCAN
solves the same column with Ros2 and sustains ``dt = 1e10 s``.

The scheme is Verwer et al. (1997), matching VULCAN's ``op.Ros2``:

    gamma = 1 + 1/sqrt(2)
    A     = I/(gamma*dt) - df/dy
    k1    = A^-1 f(y)
    y2    = y + k1/gamma
    k2    = A^-1 [ f(y2) - (2/(gamma*dt)) k1 ]
    y_new = y + (3/(2 gamma)) k1 + (1/(2 gamma)) k2

with the embedded error estimate ``|y_new - y2|``, which is what drives
step-size control.

Both solves use the *same* ``A``, so the factorization is built once and
reused -- ``SparseSystemMatrix`` caches it on the instance. The per-step
cost is therefore one Jacobian assembly plus one factorization plus two
back-substitutions and two source evaluations, roughly the cost of two
Newton iterations, against the many a stiff Newton solve needs.
"""

from __future__ import annotations

import math

import torch

from .assembly import build_implicit_operator
from .atm_state2d import AtmState2D
from .solver import solve_sparse_system
from .source import build_source_global_operator, build_source_linearization
from .transport import build_transport_matrix

GAMMA = 1.0 + 1.0 / math.sqrt(2.0)


class Ros2Result:
    """Outcome of one :func:`rosenbrock2_step`.

    ``error`` is the embedded estimate ``|y_new - y2|`` in concentration
    units; ``relative_error`` normalizes it the way VULCAN's ``delta``
    does, masked to cells above ``error_floor`` so that species at a few
    particles per cm^3 cannot dominate the step-size controller.
    """

    __slots__ = ("concentration", "error", "relative_error", "finite")

    def __init__(self, concentration, error, relative_error, finite):
        self.concentration = concentration
        self.error = error
        self.relative_error = relative_error
        self.finite = finite

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Ros2Result(relative_error={self.relative_error:.3e}, "
            f"finite={self.finite})"
        )


def _tendency(state, transport, source_terms, charge_balance_indices):
    """f(y) = transport(y) + S(y), evaluated at ``state.concentration``."""
    f = transport.matvec(state.concentration)
    lin = None
    if source_terms is not None:
        lin = build_source_linearization(
            state, source_terms, charge_balance_indices=charge_balance_indices
        )
        f = f + lin.tendency
    return f, lin


def rosenbrock2_step(
    state: AtmState2D,
    dt: float,
    *,
    kzz: torch.Tensor,
    source_terms=None,
    species_diffusion_scale: torch.Tensor | None = None,
    binary_diffusion: torch.Tensor | None = None,
    molecular_weights: torch.Tensor | None = None,
    density: torch.Tensor | None = None,
    transport_form: str | None = None,
    charge_balance_indices: "tuple[list[int], int] | None" = None,
    concentration_postprocess=None,
    clip_negative: bool = True,
    error_floor: float = 0.0,
) -> Ros2Result:
    """Advance ``state`` by ``dt`` with one Ros2 step.

    ``state`` is left untouched -- the new concentration is returned in
    the result, so a rejecting step-size controller can retry from the
    unmodified entry state (the contract ``adaptive_advance`` relies on).

    ``error_floor`` masks the relative error estimate to cells whose
    concentration exceeds it. Leaving it at 0 reproduces an unweighted
    maximum, which in practice is pinned by whichever trace species is
    moving between negligible values and never becomes small.
    """
    c0 = state.concentration
    inv_gdt = 1.0 / (GAMMA * float(dt))

    transport = build_transport_matrix(
        state,
        kzz,
        binary_diffusion=binary_diffusion,
        molecular_weights=molecular_weights,
        species_diffusion_scale=species_diffusion_scale,
        density=density,
        form=transport_form,
        boundary_conditions=None,
    )

    try:
        # --- stage 1: f(y) and the frozen Jacobian -------------------
        f1, lin = _tendency(state, transport, source_terms, charge_balance_indices)
        global_op = (
            build_source_global_operator(state, source_terms)
            if source_terms is not None
            else None
        )
        if global_op is not None:
            f1 = f1 + global_op.matvec(c0)

        operator = build_implicit_operator(
            state,
            kzz,
            binary_diffusion=binary_diffusion,
            molecular_weights=molecular_weights,
            species_diffusion_scale=species_diffusion_scale,
            density=density,
            transport_form=transport_form,
            source_terms=source_terms,
            charge_balance_indices=charge_balance_indices,
            source_linearization=lin,
            global_source_operator=global_op,
            _source_operator_computed=True,
        )
        # A = I/(gamma*dt) - df/dy, assembled sparsely.
        system = operator.affine_with_identity(-1.0, inv_gdt)

        k1 = solve_sparse_system(system, f1)
        if not torch.isfinite(k1).all():
            return Ros2Result(c0, None, float("inf"), False)

        # --- stage 2: re-evaluate f at y2, reuse the factorization ---
        y2 = c0 + k1 / GAMMA
        state.concentration = y2
        f2, _ = _tendency(state, transport, source_terms, charge_balance_indices)
        if global_op is not None:
            f2 = f2 + global_op.matvec(y2)

        k2 = solve_sparse_system(system, f2 - 2.0 * inv_gdt * k1)
        if not torch.isfinite(k2).all():
            return Ros2Result(c0, None, float("inf"), False)

        c_new = c0 + (1.5 / GAMMA) * k1 + (0.5 / GAMMA) * k2
    finally:
        # Stage 2 parks y2 in ``state`` so the source terms see it; always
        # hand the caller back the state it gave us.
        state.concentration = c0

    if not torch.isfinite(c_new).all():
        return Ros2Result(c0, None, float("inf"), False)

    if clip_negative:
        c_new = torch.clamp(c_new, min=0.0)

    # Embedded estimate: the difference between the 2nd-order solution and
    # the 1st-order stage value.
    #
    # Computed BEFORE ``concentration_postprocess``. The postprocess hook
    # carries things like a hydrostatic renormalization or a pinned surface
    # composition, which shift the solution by a finite amount that has
    # nothing to do with truncation error -- folding that in inflates the
    # estimate and makes the step controller collapse dt (measured: dt fell
    # 1.0 -> 2.9e-5 over 50 steps with 26 rejections). VULCAN keeps the same
    # separation: its Ros2 computes ``delta`` from the raw stage values and
    # the ``y = n_0 * ymix`` renormalization happens outside the solver.
    err = (c_new - y2).abs()
    denom = c_new.abs().clamp_min(torch.finfo(c_new.dtype).tiny)
    rel = err / denom
    if error_floor > 0.0:
        # Require the ENTRY value to be resolved too, not just the result.
        # A species that is exactly zero in the initial condition (VULCAN's
        # ini_mix populates only CO2/N2/O2/CO/H2O/H2/O3, so every other
        # species starts at 0) appears for the first time during the step,
        # and |c_new - y2| / c_new is then O(1) *at any dt* -- there is no
        # smallness to exploit, because this is the species' first
        # appearance rather than a truncation error. Measured on the Mars
        # cold start, the estimate was pinned by exactly such cells
        # (dt=1e-4: O@L99 with c0=0, rel=10.5; dt=1e-2: CN@L97 with c0=0,
        # y2=1.1e-47 against c_new=1.30, rel=1.0000), which made it *rise*
        # as dt fell and stopped the step controller from ever growing dt.
        resolved = (c_new > error_floor) & (c0 > error_floor)
        rel = torch.where(resolved, rel, torch.zeros_like(rel))
    # Also drop the bottom layer when a pin owns it, as VULCAN does
    # (`if use_botflux or use_fix_sp_bot: delta[0] = 0`).
    if concentration_postprocess is not None:
        rel = rel.clone()
        rel[:, 0, :] = 0.0
    relative_error = float(rel.max())

    if concentration_postprocess is not None:
        c_new = concentration_postprocess(c_new)
    return Ros2Result(c_new, err, relative_error, True)


__all__ = ["rosenbrock2_step", "Ros2Result", "GAMMA"]
