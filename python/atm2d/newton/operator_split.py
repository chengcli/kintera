"""Operator-split transport-then-chemistry advance.

KINETICS-base's MARCH operates on chemistry rates only; FLOW2D handles
diffusion in a separate operator-split step. The chemistry-only Newton
(see :mod:`atm2d.newton.chemistry_only`) converges at much larger ``dt``
than the coupled-transport variant precisely because each cell becomes
independent.

This module provides the one-step and sub-cycled drivers used by KB
Titan diagnostics and by future non-KB ion-network drivers.

The transport half uses :func:`atm2d.assembly.build_implicit_step_system`
with ``source_terms=None`` and then the chemistry half feeds the same
source-term list into :func:`atm2d.newton.chemistry_only.chemistry_only_newton_step`.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch

from ..assembly import build_implicit_step_system
from ..atm_state2d import AtmState2D
from ..matrix import SparseSystemMatrix
from ..solver import solve_sparse_system
from .chemistry_only import chemistry_only_newton_step
from .chemistry_only_bdf import chemistry_only_bdf_step
from .result import NewtonResult


def _select_chem_step():
    """Choose chemistry solver implementation via KINTERA_CHEM_SOLVER env var.

    - ``newton`` (default): the existing per-cell BE Newton iteration.
    - ``bdf``: scipy.integrate.solve_ivp with BDF method (stiff).
    - ``lsoda``: scipy LSODA with auto-switching Adams/BDF.
    - ``radau``: scipy Radau (5th-order implicit Runge-Kutta).

    The BDF variants integrate dc/dt = S(c) from 0 to dt with adaptive
    internal substeps, avoiding the silent-non-convergence failure of
    the single-step Newton at large macro dt.
    """
    mode = os.environ.get("KINTERA_CHEM_SOLVER", "newton").lower()
    if mode == "newton":
        return chemistry_only_newton_step, None
    if mode in ("bdf", "lsoda", "radau"):
        method = {"bdf": "BDF", "lsoda": "LSODA", "radau": "Radau"}[mode]
        return chemistry_only_bdf_step, method
    raise ValueError(
        f"unknown KINTERA_CHEM_SOLVER={mode!r}; use 'newton', 'bdf', 'lsoda', or 'radau'"
    )


SystemPostprocessFn = Callable[
    [SparseSystemMatrix, torch.Tensor],
    tuple[SparseSystemMatrix, torch.Tensor],
]
ConcentrationPostprocessFn = Callable[[torch.Tensor], torch.Tensor]


def operator_split_step(
    state: AtmState2D,
    dt: float,
    *,
    kzz: torch.Tensor,
    source_terms,
    species_diffusion_scale: torch.Tensor | None = None,
    transport_system_postprocess: SystemPostprocessFn | None = None,
    transport_concentration_postprocess: ConcentrationPostprocessFn | None = None,
    chemistry_postprocess: ConcentrationPostprocessFn | None = None,
    newton_kwargs: dict | None = None,
) -> NewtonResult:
    """One operator-split BE step at ``dt``.

    Steps:

    1. Build the **transport-only** implicit system (no source terms),
       apply optional Dirichlet/BC postprocess, solve it. The result
       is the concentration after transport.
    2. Apply optional concentration postprocess (e.g. boundary pins) to
       the transport result.
    3. Hand off to :func:`chemistry_only_newton_step` for the chemistry
       Newton, forwarding ``chemistry_postprocess`` as its iteration-level
       postprocess (so pins re-apply after every Newton iter).

    The function does **not** modify ``state.concentration`` on return;
    the caller is expected to commit the returned :class:`NewtonResult`
    (or reject and retry at smaller dt).
    """
    transport_system, transport_rhs = build_implicit_step_system(
        state,
        kzz,
        dt,
        species_diffusion_scale=species_diffusion_scale,
        source_terms=None,
    )
    if transport_system_postprocess is not None:
        transport_system, transport_rhs = transport_system_postprocess(
            transport_system, transport_rhs
        )
    c_after_transport = solve_sparse_system(transport_system, transport_rhs)
    if transport_concentration_postprocess is not None:
        c_after_transport = transport_concentration_postprocess(c_after_transport)
    state.concentration = c_after_transport

    kwargs = dict(newton_kwargs or {})
    kwargs.setdefault("source_terms", source_terms)
    if chemistry_postprocess is not None:
        kwargs.setdefault("concentration_postprocess", chemistry_postprocess)
    chem_step, method = _select_chem_step()
    if method is not None:
        kwargs.setdefault("method", method)
    return chem_step(state, dt, **kwargs)


def operator_split_advance(
    state: AtmState2D,
    dt: float,
    *,
    n_subcycles: int = 1,
    between_substeps_postprocess: ConcentrationPostprocessFn | None = None,
    **step_kwargs,
) -> NewtonResult:
    """Run ``n_subcycles`` operator-split sub-steps of ``dt / n_subcycles``
    each, threading ``between_substeps_postprocess`` (typically the atomic
    -budget projection + pin reset) between sub-steps.

    Returns an aggregated :class:`NewtonResult` whose ``converged`` flag
    is the conjunction of all sub-step convergences and whose
    ``iterations`` is the sum.

    If ``n_subcycles == 1`` this degenerates to one
    :func:`operator_split_step` call.
    """
    if n_subcycles < 1:
        raise ValueError(f"n_subcycles must be >= 1, got {n_subcycles}")
    pristine_c0 = state.concentration.clone()
    dt_sub = float(dt) / n_subcycles

    chem_result: NewtonResult | None = None
    agg_iters = 0
    agg_conv = True
    for _ in range(n_subcycles):
        chem_result = operator_split_step(state, dt_sub, **step_kwargs)
        if between_substeps_postprocess is not None:
            projected = between_substeps_postprocess(chem_result.concentration)
            chem_result = NewtonResult(
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
    final = NewtonResult(
        concentration=chem_result.concentration,
        converged=agg_conv,
        iterations=agg_iters,
        max_relative_change=chem_result.max_relative_change,
        residual_history=chem_result.residual_history,
    )
    state.concentration = pristine_c0
    return final


__all__ = ["operator_split_step", "operator_split_advance"]
