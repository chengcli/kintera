from __future__ import annotations

import torch

from .chemistry import build_chemistry_jacobian, build_photochemistry_jacobian
from .matrix import SparseSystemMatrix, add_sparse_system_matrices
from .source import (
    LocalSourceTerm,
    build_source_global_operator,
    build_source_linearization,
)
from .atm_state2d import AtmState2D, SpeciesBoundaryConditions2D
from .transport import build_transport_matrix


def build_implicit_operator(
    state: AtmState2D,
    kzz: torch.Tensor,
    *,
    kyy: torch.Tensor | None = None,
    kzy: torch.Tensor | None = None,
    kyz: torch.Tensor | None = None,
    binary_diffusion: torch.Tensor | None = None,
    molecular_weights: torch.Tensor | None = None,
    species_diffusion_scale: torch.Tensor | None = None,
    kinetics=None,
    photo_chem=None,
    actinic_flux: torch.Tensor | None = None,
    source_terms: list[LocalSourceTerm] | None = None,
    include_identity: bool = False,
    dt: float | None = None,
    boundary_conditions: SpeciesBoundaryConditions2D | None = None,
) -> SparseSystemMatrix:
    """Assemble the full implicit operator for transport and chemistry.

    The returned matrix acts on the flattened species state. Transport terms
    come from ``build_transport_matrix``. Chemistry and photochemistry
    contribute cell-local Jacobian blocks on the diagonal. When
    ``include_identity`` is enabled, ``I / dt`` is also added to the diagonal
    for backward-Euler style steady-state or implicit time-marching solves.
    """
    operator = build_transport_matrix(
        state,
        kzz,
        kyy=kyy,
        kzy=kzy,
        kyz=kyz,
        binary_diffusion=binary_diffusion,
        molecular_weights=molecular_weights,
        species_diffusion_scale=species_diffusion_scale,
        boundary_conditions=None,
    )

    diag_update = torch.zeros(
        (state.ncol, state.nlyr, state.nspecies, state.nspecies),
        dtype=state.dtype,
        device=state.device,
    )
    if kinetics is not None:
        diag_update = diag_update + build_chemistry_jacobian(
            kinetics, state.temperature, state.pressure, state.concentration
        )
    if photo_chem is not None:
        if actinic_flux is None:
            raise ValueError("actinic_flux is required when photo_chem is provided")
        diag_update = diag_update + build_photochemistry_jacobian(
            photo_chem, state.temperature, state.concentration, actinic_flux
        )
    if source_terms is not None:
        diag_update = diag_update + build_source_linearization(
            state, source_terms
        ).jacobian

    if include_identity:
        if dt is None:
            raise ValueError("dt is required when include_identity is True")
        eye = torch.eye(state.nspecies, dtype=state.dtype, device=state.device)
        diag_update = diag_update + eye.view(1, 1, state.nspecies, state.nspecies) / dt

    matrix = operator.add_diagonal(diag_update)
    if source_terms is not None:
        global_source_operator = build_source_global_operator(state, source_terms)
        if global_source_operator is not None:
            matrix = add_sparse_system_matrices(matrix, global_source_operator)
    if boundary_conditions is None:
        return matrix

    from .transport import _apply_boundary_conditions

    return _apply_boundary_conditions(state, matrix, boundary_conditions)


def build_implicit_step_system(
    state: AtmState2D,
    kzz: torch.Tensor,
    dt: float,
    *,
    kyy: torch.Tensor | None = None,
    kzy: torch.Tensor | None = None,
    kyz: torch.Tensor | None = None,
    binary_diffusion: torch.Tensor | None = None,
    molecular_weights: torch.Tensor | None = None,
    species_diffusion_scale: torch.Tensor | None = None,
    source_terms: list[LocalSourceTerm] | None = None,
    c0: torch.Tensor | None = None,
) -> tuple[SparseSystemMatrix, torch.Tensor]:
    """Build a backward-Euler system with linearized local source terms.

    Source terms are linearized around ``state.concentration`` as
    ``S(c_new) ~= S(c_k) + J(c_k) * (c_new - c_k)`` where ``c_k`` is taken from
    ``state.concentration``. The returned system solves
    ``(I - dt * (T + J(c_k))) c_new = c0 + dt * (S(c_k) - J(c_k) c_k)``.

    When ``c0`` is omitted the RHS uses ``state.concentration`` itself as the
    backward-Euler starting point — this is the original single-shot frozen
    linearization. Pass ``c0`` explicitly to keep the BE starting point fixed
    across Newton iterations that re-linearize at successive ``c_k``.
    """

    source_linearization = None
    global_source_operator = None
    if source_terms is not None:
        source_linearization = build_source_linearization(state, source_terms)
        global_source_operator = build_source_global_operator(state, source_terms)
    operator = build_implicit_operator(
        state,
        kzz,
        kyy=kyy,
        kzy=kzy,
        kyz=kyz,
        binary_diffusion=binary_diffusion,
        molecular_weights=molecular_weights,
        species_diffusion_scale=species_diffusion_scale,
        source_terms=source_terms,
    )
    identity = torch.eye(operator.nstate, dtype=state.dtype, device=state.device)
    system = SparseSystemMatrix.from_dense(
        identity - float(dt) * operator.global_csr.to_dense(),
        ncol=state.ncol,
        nlyr=state.nlyr,
        nspecies=state.nspecies,
    )
    rhs = state.concentration if c0 is None else c0
    if source_linearization is not None:
        jacobian_state = torch.einsum(
            "clij,clj->cli",
            source_linearization.jacobian,
            state.concentration,
        )
        if global_source_operator is not None:
            jacobian_state = jacobian_state + global_source_operator.matvec(
                state.concentration
            )
        rhs = rhs + float(dt) * (source_linearization.tendency - jacobian_state)
    return system, rhs
