"""BDF stiff-ODE chemistry-only step.

Drop-in replacement for ``chemistry_only_newton_step`` that uses
``scipy.integrate.solve_ivp`` with the BDF method to integrate

    dc/dt = S(c)

from t=0 to t=dt internally. BDF (backward differentiation formulas) is a
well-understood implicit stiff method that adaptively chooses internal
substep size — exactly what's needed when the network has reaction
timescales spanning many orders of magnitude.

The kintera Newton step in ``chemistry_only.py`` does a single BE step
which fails to converge at large dt for stiff Titan chemistry (the
Newton iteration diverges when the time-step jumps faster than the
local timescale). BDF handles this by stepping internally at whatever
small dt the system actually requires.

Returns the same :class:`NewtonResult` as ``chemistry_only_newton_step``
so it can be swapped in via the operator-split wrapper.

Implementation notes
--------------------
* Jacobian: kintera's ``build_source_linearization`` already returns
  the per-cell Jacobian ``(ncol, nlyr, nspecies, nspecies)`` analytically.
  This is block-diagonal in the global ``(B, B)`` system where
  ``B = ncol·nlyr·nspecies``. We provide it to scipy as a CSR sparse
  matrix; ``jac_sparsity`` is also set so BDF can use sparse LU.
* CPU only: scipy.integrate is numpy-based, so each f/Jac call ferries
  the state through ``tensor.cpu().numpy()`` and back. For Titan
  (ncol=1, nlyr=50, nspecies=128) that's ~6400 floats per call; the
  overhead is acceptable. GPU would require a torch-native stiff
  solver (torchdiffeq's ``odeint`` doesn't currently ship BDF, but
  could be added).
* Per-cell vs whole-column: we treat the whole column as one big ODE
  system because BDF's internal step-size selection is set by the
  fastest cell anyway. The block-diagonal sparsity pattern means
  factorization cost scales linearly with ncol·nlyr, not their cube.
"""

from __future__ import annotations

import numpy as np
import scipy.integrate
import scipy.sparse
import torch

from ..atm_state2d import AtmState2D
from ..sources import build_source_linearization
from .result import ConcentrationPostprocess, NewtonResult


def _build_block_diag_sparsity(ncol: int, nlyr: int, nspecies: int) -> scipy.sparse.csr_matrix:
    """Sparsity pattern for the cell-local Jacobian — block-diagonal of
    ``ncol*nlyr`` dense ``(nspecies, nspecies)`` blocks."""
    nblocks = ncol * nlyr
    block = np.ones((nspecies, nspecies), dtype=np.float64)
    blocks = [block] * nblocks
    return scipy.sparse.block_diag(blocks, format="csr")


def chemistry_only_bdf_step(
    state: AtmState2D,
    dt: float,
    *,
    source_terms,
    concentration_postprocess: ConcentrationPostprocess | None = None,
    rtol: float = 1e-4,
    atol_rel: float = 1e-12,
    method: str = "BDF",
    clip_negative: bool | str = True,
    mass_conservation_cap: bool = True,
    max_concentration_cap: "torch.Tensor | None" = None,
    charge_balance_indices: "tuple[list[int], int] | None" = None,
    max_steps: int = 10000,
    **_unused,
) -> NewtonResult:
    """BDF stiff-ODE chemistry-only step.

    Parameters
    ----------
    state, dt, source_terms : same as :func:`chemistry_only_newton_step`.
    rtol : float
        Relative tolerance for BDF internal step control.
    atol_rel : float
        Absolute tolerance is computed as ``atol_rel * max(c0)`` per
        species. Default 1e-12 means species below 1e-12 × max-concentration
        are treated as numerical noise.
    method : str
        scipy method name. ``BDF`` is the default; ``LSODA`` and
        ``Radau`` also work for stiff problems.
    clip_negative, mass_conservation_cap, max_concentration_cap,
    concentration_postprocess :
        Applied after the integration completes (BDF can produce small
        negative values from rounding).

    ``**_unused`` swallows the Newton-only kwargs
    (``max_iterations``, ``convergence_tol``, ``damping_factor``,
    ``divergence_threshold``, etc.) so this can be a drop-in replacement.
    """
    c0 = state.concentration.detach().clone()
    ncol, nlyr, nspecies = c0.shape
    dtype = state.dtype
    device = state.device

    if mass_conservation_cap and max_concentration_cap is None:
        cell_total_density = c0.sum(dim=-1, keepdim=True)
    elif max_concentration_cap is not None:
        cap = max_concentration_cap
        if cap.dim() == 2:
            cap = cap.unsqueeze(-1)
        cell_total_density = cap.to(dtype=dtype, device=device)
    else:
        cell_total_density = None

    y0 = c0.detach().cpu().numpy().astype(np.float64).ravel()
    # atol per-species: atol_rel × per-species column max (over cells), with floor
    species_max = c0.abs().amax(dim=(0, 1)).cpu().numpy()
    species_max = np.maximum(species_max, 1.0)
    atol_per_species = atol_rel * species_max
    atol = np.tile(atol_per_species, ncol * nlyr)

    # Allocate a working state used inside f/jac callbacks. We do NOT
    # mutate the user's state — we build a shallow clone.
    work_state = state
    saved_conc = state.concentration

    def _y_to_tensor(y_flat):
        t = torch.as_tensor(y_flat, dtype=dtype, device=device).reshape(c0.shape)
        return t

    def f(t, y_flat):
        c = _y_to_tensor(y_flat)
        work_state.concentration = c
        lin = build_source_linearization(
            work_state, source_terms,
            charge_balance_indices=charge_balance_indices,
        )
        return lin.tendency.detach().cpu().numpy().astype(np.float64).ravel()

    def jac(t, y_flat):
        c = _y_to_tensor(y_flat)
        work_state.concentration = c
        lin = build_source_linearization(
            work_state, source_terms,
            charge_balance_indices=charge_balance_indices,
        )
        jac_np = lin.jacobian.detach().cpu().numpy().astype(np.float64)
        # jac_np shape: (ncol, nlyr, nspecies, nspecies). Build CSR block-diag.
        blocks = jac_np.reshape(-1, nspecies, nspecies)
        return scipy.sparse.block_diag(list(blocks), format="csr")

    jac_sparsity = _build_block_diag_sparsity(ncol, nlyr, nspecies)

    try:
        sol = scipy.integrate.solve_ivp(
            f,
            (0.0, float(dt)),
            y0,
            method=method,
            jac=jac,
            jac_sparsity=jac_sparsity,
            rtol=rtol,
            atol=atol,
            dense_output=False,
            max_step=float(dt),
            first_step=min(float(dt) * 1.0e-3, 1.0e-3),
        )
    except Exception as exc:
        # Anything goes wrong (LinAlg error, etc.): restore state and report
        # non-convergence so the adaptive_advance wrapper can reject and retry.
        state.concentration = saved_conc
        return NewtonResult(
            concentration=torch.full_like(c0, float("nan")),
            converged=False,
            iterations=0,
            max_relative_change=float("inf"),
            residual_history=[],
        )

    y_final = sol.y[:, -1]
    c_new = torch.from_numpy(y_final).to(dtype=dtype, device=device).reshape(c0.shape)

    if clip_negative == "abs":
        c_new = torch.abs(c_new)
    elif clip_negative:
        c_new = torch.clamp(c_new, min=0.0)

    if cell_total_density is not None:
        c_new = torch.minimum(c_new, cell_total_density)

    if concentration_postprocess is not None:
        c_new = concentration_postprocess(c_new)

    state.concentration = saved_conc

    converged = bool(sol.success)
    # solve_ivp reports nfev (function evals); use as "iterations" proxy
    iters = int(getattr(sol, "nfev", 0))

    if not converged or not bool(torch.isfinite(c_new).all().item()):
        return NewtonResult(
            concentration=torch.full_like(c0, float("nan")),
            converged=False,
            iterations=iters,
            max_relative_change=float("inf"),
            residual_history=[],
        )

    return NewtonResult(
        concentration=c_new,
        converged=True,
        iterations=iters,
        max_relative_change=0.0,
        residual_history=[],
    )


__all__ = ["chemistry_only_bdf_step"]
