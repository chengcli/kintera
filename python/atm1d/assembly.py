from __future__ import annotations

import torch

from .chemistry import build_chemistry_jacobian, build_photochemistry_jacobian
from .matrix import BlockTridiagonalMatrix
from .state import ColumnState1D
from .transport import build_transport_matrix


def build_implicit_operator(
    state: ColumnState1D,
    kzz: torch.Tensor,
    *,
    binary_diffusion: torch.Tensor | None = None,
    molecular_weights: torch.Tensor | None = None,
    kinetics=None,
    photo_chem=None,
    actinic_flux: torch.Tensor | None = None,
    include_identity: bool = False,
    dt: float | None = None,
) -> BlockTridiagonalMatrix:
    operator = build_transport_matrix(
        state,
        kzz,
        binary_diffusion=binary_diffusion,
        molecular_weights=molecular_weights,
    )

    diag = operator.diag_packed.clone()
    if kinetics is not None:
        diag = diag + build_chemistry_jacobian(
            kinetics, state.temperature, state.pressure, state.concentration
        )
    if photo_chem is not None:
        if actinic_flux is None:
            raise ValueError("actinic_flux is required when photo_chem is provided")
        diag = diag + build_photochemistry_jacobian(
            photo_chem, state.temperature, state.concentration, actinic_flux
        )

    if include_identity:
        if dt is None:
            raise ValueError("dt is required when include_identity is True")
        eye = torch.eye(state.nspecies, dtype=state.dtype, device=state.device)
        diag = diag + eye.unsqueeze(0) / dt

    return BlockTridiagonalMatrix.from_dense(
        operator.lower_packed, diag, operator.upper_packed
    )
