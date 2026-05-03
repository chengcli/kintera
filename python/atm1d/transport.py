from __future__ import annotations

import torch

from .matrix import BlockTridiagonalMatrix, add_block_tridiagonal
from .state import ColumnState1D

GAS_CONSTANT_CGS = 8.31446261815324e7


def build_eddy_diffusion_blocks(
    state: ColumnState1D,
    kzz: torch.Tensor,
) -> BlockTridiagonalMatrix:
    kzz = _as_interface_vector(kzz, state)
    eye = torch.eye(state.nspecies, dtype=state.dtype, device=state.device)
    matrix = kzz[:, None, None] * eye.unsqueeze(0)
    gravity = torch.zeros((state.nz - 1, state.nspecies), dtype=state.dtype, device=state.device)
    return _assemble_transport_blocks(state, matrix, gravity)


def build_binary_diffusion_blocks(
    state: ColumnState1D,
    binary_diffusion: torch.Tensor,
    molecular_weights: torch.Tensor,
    *,
    include_gravity: bool = True,
    gas_constant: float = GAS_CONSTANT_CGS,
) -> BlockTridiagonalMatrix:
    binary_diffusion = _as_interface_matrix(binary_diffusion, state)
    molecular_weights = torch.as_tensor(
        molecular_weights, dtype=state.dtype, device=state.device
    )
    if molecular_weights.dim() != 1 or molecular_weights.numel() != state.nspecies:
        raise ValueError("molecular_weights must have shape (nspecies,)")

    if include_gravity:
        _xmid, tmid, dwtm = _interface_thermo(state, molecular_weights)
        gravity_term = 0.5 * torch.einsum(
            "bij,bj->bi", binary_diffusion, dwtm
        ) * (state.gravity / (tmid[:, None] * gas_constant))
    else:
        gravity_term = torch.zeros(
            (state.nz - 1, state.nspecies), dtype=state.dtype, device=state.device
        )

    return _assemble_transport_blocks(state, binary_diffusion, gravity_term)


def build_transport_matrix(
    state: ColumnState1D,
    kzz: torch.Tensor,
    *,
    binary_diffusion: torch.Tensor | None = None,
    molecular_weights: torch.Tensor | None = None,
    include_gravity: bool = True,
    gas_constant: float = GAS_CONSTANT_CGS,
) -> BlockTridiagonalMatrix:
    matrices = [build_eddy_diffusion_blocks(state, kzz)]
    if binary_diffusion is not None:
        if molecular_weights is None:
            raise ValueError("molecular_weights are required with binary_diffusion")
        matrices.append(
            build_binary_diffusion_blocks(
                state,
                binary_diffusion,
                molecular_weights,
                include_gravity=include_gravity,
                gas_constant=gas_constant,
            )
        )
    return add_block_tridiagonal(*matrices)


def _assemble_transport_blocks(
    state: ColumnState1D,
    interface_matrix: torch.Tensor,
    gravity_term: torch.Tensor,
) -> BlockTridiagonalMatrix:
    nz, ns = state.nz, state.nspecies
    lower = torch.zeros((nz, ns, ns), dtype=state.dtype, device=state.device)
    diag = torch.zeros_like(lower)
    upper = torch.zeros_like(lower)

    dzi = _interface_spacing(state.z)
    delta_z = _cell_spacing(state.z)

    for interface in range(nz - 1):
        inv_scale_left = 1.0 / (delta_z[interface] * dzi[interface])
        inv_scale_right = 1.0 / (delta_z[interface + 1] * dzi[interface])
        coupling = interface_matrix[interface]
        gravity_diag = torch.diag(gravity_term[interface])

        diag[interface] += -inv_scale_left * coupling
        upper[interface] += inv_scale_left * coupling - gravity_diag / delta_z[interface]

        lower[interface + 1] += inv_scale_right * coupling
        diag[interface + 1] += (
            -inv_scale_right * coupling + gravity_diag / delta_z[interface + 1]
        )

    return BlockTridiagonalMatrix.from_dense(lower, diag, upper)


def _as_interface_vector(value: torch.Tensor, state: ColumnState1D) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=state.dtype, device=state.device)
    if tensor.dim() != 1 or tensor.numel() != state.nz - 1:
        raise ValueError("interface vector must have shape (nz - 1,)")
    return tensor


def _as_interface_matrix(value: torch.Tensor, state: ColumnState1D) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=state.dtype, device=state.device)
    if tensor.shape != (state.nz - 1, state.nspecies, state.nspecies):
        raise ValueError(
            "binary_diffusion must have shape (nz - 1, nspecies, nspecies)"
        )
    return tensor


def _interface_spacing(z: torch.Tensor) -> torch.Tensor:
    return z[1:] - z[:-1]


def _cell_spacing(z: torch.Tensor) -> torch.Tensor:
    nz = z.numel()
    delta_z = torch.empty_like(z)
    delta_z[0] = z[1] - z[0]
    if nz > 2:
        delta_z[1:-1] = 0.5 * (z[2:] - z[:-2])
    delta_z[-1] = z[-1] - z[-2]
    return delta_z


def _interface_thermo(
    state: ColumnState1D, molecular_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _mole_fraction(state.concentration)
    xmid = 0.5 * (x[:-1] + x[1:])
    tmid = 0.5 * (state.temperature[:-1] + state.temperature[1:])
    wtm_mid = torch.einsum("bi,i->b", xmid, molecular_weights)
    dwtm = molecular_weights.unsqueeze(0) - wtm_mid.unsqueeze(-1)
    return xmid, tmid, dwtm


def _mole_fraction(concentration: torch.Tensor) -> torch.Tensor:
    return concentration / concentration.sum(dim=-1, keepdim=True).clamp_min(1.0e-300)
