from __future__ import annotations

import torch

from .chemistry import build_chemistry_jacobian, build_photochemistry_jacobian
from .matrix import SparseSystemMatrix
from .state import AtmState2D, SpeciesBoundaryConditions2D
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
    kinetics=None,
    photo_chem=None,
    actinic_flux: torch.Tensor | None = None,
    include_identity: bool = False,
    dt: float | None = None,
    boundary_conditions: SpeciesBoundaryConditions2D | None = None,
) -> SparseSystemMatrix:
    operator = build_transport_matrix(
        state,
        kzz,
        kyy=kyy,
        kzy=kzy,
        kyz=kyz,
        binary_diffusion=binary_diffusion,
        molecular_weights=molecular_weights,
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

    if include_identity:
        if dt is None:
            raise ValueError("dt is required when include_identity is True")
        eye = torch.eye(state.nspecies, dtype=state.dtype, device=state.device)
        diag_update = diag_update + eye.view(1, 1, state.nspecies, state.nspecies) / dt

    matrix = operator.add_diagonal(diag_update)
    if boundary_conditions is None:
        return matrix

    from .transport import _apply_boundary_conditions

    return _apply_boundary_conditions(state, matrix, boundary_conditions)
