from __future__ import annotations

import torch

from .matrix import SparseSystemMatrix, add_sparse_system_matrices, flatten_state_index
from .state import AtmState2D, SpeciesBoundaryCondition, SpeciesBoundaryConditions2D

GAS_CONSTANT_CGS = 8.31446261815324e7


def build_eddy_diffusion_matrix(
    state: AtmState2D,
    kzz: torch.Tensor,
    *,
    kyy: torch.Tensor | None = None,
    kzy: torch.Tensor | None = None,
    kyz: torch.Tensor | None = None,
    boundary_conditions: SpeciesBoundaryConditions2D | None = None,
) -> SparseSystemMatrix:
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []
    _assemble_vertical_scalar_diffusion(rows, cols, vals, state, _as_vertical_face_scalar(kzz, state))
    if kyy is not None:
        _assemble_horizontal_scalar_diffusion(rows, cols, vals, state, _as_horizontal_face_scalar(kyy, state))
    cross_center = None
    if kzy is not None:
        cross_center = _as_center_scalar(kzy, state)
    if kyz is not None:
        cross_center = _as_center_scalar(kyz, state) if cross_center is None else 0.5 * (
            cross_center + _as_center_scalar(kyz, state)
        )
    if cross_center is not None and state.ncol > 1 and state.nlyr > 1:
        _assemble_cross_diffusion(rows, cols, vals, state, cross_center)
    matrix = _matrix_from_triplets(rows, cols, vals, state)
    return _apply_boundary_conditions(state, matrix, boundary_conditions)


def build_binary_diffusion_matrix(
    state: AtmState2D,
    binary_diffusion: torch.Tensor,
    molecular_weights: torch.Tensor,
    *,
    include_gravity: bool = True,
    gas_constant: float = GAS_CONSTANT_CGS,
    boundary_conditions: SpeciesBoundaryConditions2D | None = None,
) -> SparseSystemMatrix:
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []
    binary_4d = _as_vertical_face_matrix(binary_diffusion, state)
    molecular_weights = torch.as_tensor(molecular_weights, dtype=state.dtype, device=state.device)
    if molecular_weights.shape != (state.nspecies,):
        raise ValueError("molecular_weights must have shape (nspecies,)")

    gravity_term = None
    if include_gravity:
        xmid, tmid, dwtm = _interface_thermo(state, molecular_weights)
        gravity_term = 0.5 * torch.einsum("czij,czj->czi", binary_4d, dwtm)
        gravity_term = gravity_term * (state.gravity / (tmid * gas_constant))

    _assemble_vertical_matrix_diffusion(rows, cols, vals, state, binary_4d, gravity_term)
    matrix = _matrix_from_triplets(rows, cols, vals, state)
    return _apply_boundary_conditions(state, matrix, boundary_conditions)


def build_transport_matrix(
    state: AtmState2D,
    kzz: torch.Tensor,
    *,
    kyy: torch.Tensor | None = None,
    kzy: torch.Tensor | None = None,
    kyz: torch.Tensor | None = None,
    binary_diffusion: torch.Tensor | None = None,
    molecular_weights: torch.Tensor | None = None,
    include_gravity: bool = True,
    gas_constant: float = GAS_CONSTANT_CGS,
    boundary_conditions: SpeciesBoundaryConditions2D | None = None,
) -> SparseSystemMatrix:
    matrices = [
        build_eddy_diffusion_matrix(
            state,
            kzz,
            kyy=kyy,
            kzy=kzy,
            kyz=kyz,
            boundary_conditions=None,
        )
    ]
    if binary_diffusion is not None:
        if molecular_weights is None:
            raise ValueError("molecular_weights are required with binary_diffusion")
        matrices.append(
            build_binary_diffusion_matrix(
                state,
                binary_diffusion,
                molecular_weights,
                include_gravity=include_gravity,
                gas_constant=gas_constant,
                boundary_conditions=None,
            )
        )
    transport = add_sparse_system_matrices(*matrices)
    return _apply_boundary_conditions(state, transport, boundary_conditions)


def _assemble_vertical_scalar_diffusion(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
    kzz: torch.Tensor,
) -> None:
    eye = torch.eye(state.nspecies, dtype=state.dtype, device=state.device)
    for icol in range(state.ncol):
        for ilev in range(state.nlyr - 1):
            block = kzz[icol, ilev] * eye
            _add_two_cell_block(rows, cols, vals, state, icol, ilev, icol, ilev + 1, block, axis="z")


def _assemble_horizontal_scalar_diffusion(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
    kyy: torch.Tensor,
) -> None:
    eye = torch.eye(state.nspecies, dtype=state.dtype, device=state.device)
    for icol in range(state.ncol - 1):
        for ilev in range(state.nlyr):
            block = kyy[icol, ilev] * eye
            _add_two_cell_block(rows, cols, vals, state, icol, ilev, icol + 1, ilev, block, axis="y")


def _assemble_vertical_matrix_diffusion(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
    binary_diffusion: torch.Tensor,
    gravity_term: torch.Tensor | None,
) -> None:
    for icol in range(state.ncol):
        for ilev in range(state.nlyr - 1):
            block = binary_diffusion[icol, ilev]
            _add_two_cell_block(rows, cols, vals, state, icol, ilev, icol, ilev + 1, block, axis="z")
            if gravity_term is not None:
                gravity_diag = torch.diag(gravity_term[icol, ilev])
                _add_block(
                    rows,
                    cols,
                    vals,
                    state,
                    icol,
                    ilev,
                    icol,
                    ilev + 1,
                    -gravity_diag / state.dx1f[ilev],
                )
                _add_block(
                    rows,
                    cols,
                    vals,
                    state,
                    icol,
                    ilev + 1,
                    icol,
                    ilev + 1,
                    gravity_diag / state.dx1f[ilev + 1],
                )


def _assemble_cross_diffusion(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
    kzy_center: torch.Tensor,
) -> None:
    for icol in range(state.ncol):
        for ilev in range(state.nlyr - 1):
            coeff = 0.5 * (kzy_center[icol, ilev] + kzy_center[icol, ilev + 1])
            grad_weights = _average_gradient_weights_y(state, icol, ilev, ilev + 1)
            for src_col, src_lyr, weight in grad_weights:
                block = coeff * weight * torch.eye(state.nspecies, dtype=state.dtype, device=state.device)
                _add_scaled_flux_pair(rows, cols, vals, state, icol, ilev, icol, ilev + 1, src_col, src_lyr, block, axis="z")

    for icol in range(state.ncol - 1):
        for ilev in range(state.nlyr):
            coeff = 0.5 * (kzy_center[icol, ilev] + kzy_center[icol + 1, ilev])
            grad_weights = _average_gradient_weights_z(state, icol, icol + 1, ilev)
            for src_col, src_lyr, weight in grad_weights:
                block = coeff * weight * torch.eye(state.nspecies, dtype=state.dtype, device=state.device)
                _add_scaled_flux_pair(rows, cols, vals, state, icol, ilev, icol + 1, ilev, src_col, src_lyr, block, axis="y")


def _add_two_cell_block(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
    left_col: int,
    left_lyr: int,
    right_col: int,
    right_lyr: int,
    block: torch.Tensor,
    *,
    axis: str,
) -> None:
    if axis == "z":
        left_scale = 1.0 / (state.dx1f[left_lyr] * state.dx1v[left_lyr])
        right_scale = 1.0 / (state.dx1f[right_lyr] * state.dx1v[left_lyr])
    elif axis == "y":
        left_scale = 1.0 / (state.dx2f[left_col] * state.dx2v[left_col])
        right_scale = 1.0 / (state.dx2f[right_col] * state.dx2v[left_col])
    else:
        raise ValueError("axis must be 'y' or 'z'")

    _add_block(rows, cols, vals, state, left_col, left_lyr, left_col, left_lyr, -left_scale * block)
    _add_block(rows, cols, vals, state, left_col, left_lyr, right_col, right_lyr, left_scale * block)
    _add_block(rows, cols, vals, state, right_col, right_lyr, left_col, left_lyr, right_scale * block)
    _add_block(rows, cols, vals, state, right_col, right_lyr, right_col, right_lyr, -right_scale * block)


def _add_scaled_flux_pair(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
    left_col: int,
    left_lyr: int,
    right_col: int,
    right_lyr: int,
    src_col: int,
    src_lyr: int,
    block: torch.Tensor,
    *,
    axis: str,
) -> None:
    if axis == "z":
        left_scale = 1.0 / state.dx1f[left_lyr]
        right_scale = -1.0 / state.dx1f[right_lyr]
    elif axis == "y":
        left_scale = 1.0 / state.dx2f[left_col]
        right_scale = -1.0 / state.dx2f[right_col]
    else:
        raise ValueError("axis must be 'y' or 'z'")
    _add_block(rows, cols, vals, state, left_col, left_lyr, src_col, src_lyr, left_scale * block)
    _add_block(rows, cols, vals, state, right_col, right_lyr, src_col, src_lyr, right_scale * block)


def _add_block(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
    row_col: int,
    row_lyr: int,
    col_col: int,
    col_lyr: int,
    block: torch.Tensor,
) -> None:
    nz = block.nonzero(as_tuple=False)
    if nz.numel() == 0:
        return
    row_base = flatten_state_index(
        torch.full((nz.size(0),), row_col, dtype=torch.int64, device=state.device),
        torch.full((nz.size(0),), row_lyr, dtype=torch.int64, device=state.device),
        nz[:, 0].to(device=state.device, dtype=torch.int64),
        state.nlyr,
        state.nspecies,
    )
    col_base = flatten_state_index(
        torch.full((nz.size(0),), col_col, dtype=torch.int64, device=state.device),
        torch.full((nz.size(0),), col_lyr, dtype=torch.int64, device=state.device),
        nz[:, 1].to(device=state.device, dtype=torch.int64),
        state.nlyr,
        state.nspecies,
    )
    rows.append(row_base)
    cols.append(col_base)
    vals.append(block[nz[:, 0], nz[:, 1]])


def _matrix_from_triplets(
    rows: list[torch.Tensor],
    cols: list[torch.Tensor],
    vals: list[torch.Tensor],
    state: AtmState2D,
) -> SparseSystemMatrix:
    if rows:
        indices = torch.stack([torch.cat(rows), torch.cat(cols)])
        values = torch.cat(vals)
    else:
        indices = torch.empty((2, 0), dtype=torch.int64, device=state.device)
        values = torch.empty((0,), dtype=state.dtype, device=state.device)
    return SparseSystemMatrix.from_coo(
        indices,
        values,
        ncol=state.ncol,
        nlyr=state.nlyr,
        nspecies=state.nspecies,
        device=state.device,
        dtype=state.dtype,
    )


def _apply_boundary_conditions(
    state: AtmState2D,
    matrix: SparseSystemMatrix,
    boundary_conditions: SpeciesBoundaryConditions2D | None,
) -> SparseSystemMatrix:
    if boundary_conditions is None:
        return matrix

    override_mask = torch.zeros(
        (state.ncol, state.nlyr, state.nspecies), dtype=torch.bool, device=state.device
    )
    override_values = torch.zeros(
        (state.ncol, state.nlyr, state.nspecies), dtype=state.dtype, device=state.device
    )

    row_ids: list[int] = []
    col_ids: list[int] = []
    values: list[float] = []

    if boundary_conditions.left is not None:
        _apply_single_boundary(
            row_ids, col_ids, values, override_mask, override_values, state, boundary_conditions.left, side="left"
        )
    if boundary_conditions.right is not None:
        _apply_single_boundary(
            row_ids, col_ids, values, override_mask, override_values, state, boundary_conditions.right, side="right"
        )
    if boundary_conditions.bottom is not None:
        _apply_single_boundary(
            row_ids, col_ids, values, override_mask, override_values, state, boundary_conditions.bottom, side="bottom"
        )
    if boundary_conditions.top is not None:
        _apply_single_boundary(
            row_ids, col_ids, values, override_mask, override_values, state, boundary_conditions.top, side="top"
        )

    if not row_ids:
        return matrix
    return matrix.replace_rows(
        torch.tensor(row_ids, dtype=torch.int64, device=state.device),
        torch.tensor(col_ids, dtype=torch.int64, device=state.device),
        torch.tensor(values, dtype=state.dtype, device=state.device),
        rhs_override_mask=override_mask,
        rhs_override_values=override_values,
    )


def _apply_single_boundary(
    row_ids: list[int],
    col_ids: list[int],
    values: list[float],
    override_mask: torch.Tensor,
    override_values: torch.Tensor,
    state: AtmState2D,
    condition: SpeciesBoundaryCondition,
    *,
    side: str,
) -> None:
    kinds = condition.kinds(state.nspecies)
    if side in {"left", "right"}:
        nedge = state.nlyr
    else:
        nedge = state.ncol
    bc_values = condition.values(nedge, state.nspecies, dtype=state.dtype, device=state.device)

    for edge_idx in range(nedge):
        if side == "left":
            row_col, row_lyr = 0, edge_idx
            nei_col, nei_lyr = 1, edge_idx
            spacing = state.dx2v[0]
        elif side == "right":
            row_col, row_lyr = state.ncol - 1, edge_idx
            nei_col, nei_lyr = state.ncol - 2, edge_idx
            spacing = state.dx2v[-1]
        elif side == "bottom":
            row_col, row_lyr = edge_idx, 0
            nei_col, nei_lyr = edge_idx, 1
            spacing = state.dx1v[0]
        elif side == "top":
            row_col, row_lyr = edge_idx, state.nlyr - 1
            nei_col, nei_lyr = edge_idx, state.nlyr - 2
            spacing = state.dx1v[-1]
        else:
            raise ValueError("unknown boundary side")

        for ispecies, kind in enumerate(kinds):
            row_index = int(flatten_state_index(row_col, row_lyr, ispecies, state.nlyr, state.nspecies).item())
            kind = kind.lower()
            if kind == "none":
                continue
            override_mask[row_col, row_lyr, ispecies] = True
            override_values[row_col, row_lyr, ispecies] = bc_values[edge_idx, ispecies]
            if kind == "dirichlet":
                row_ids.append(row_index)
                col_ids.append(row_index)
                values.append(1.0)
            elif kind == "neumann":
                nei_index = int(
                    flatten_state_index(nei_col, nei_lyr, ispecies, state.nlyr, state.nspecies).item()
                )
                if side in {"left", "bottom"}:
                    row_ids.extend([row_index, row_index])
                    col_ids.extend([row_index, nei_index])
                    values.extend([-1.0 / float(spacing), 1.0 / float(spacing)])
                else:
                    row_ids.extend([row_index, row_index])
                    col_ids.extend([nei_index, row_index])
                    values.extend([-1.0 / float(spacing), 1.0 / float(spacing)])
            else:
                raise ValueError("boundary condition kind must be 'none', 'dirichlet', or 'neumann'")


def _average_gradient_weights_y(
    state: AtmState2D,
    col_idx: int,
    lower_lyr: int,
    upper_lyr: int,
) -> list[tuple[int, int, torch.Tensor]]:
    weights = _cell_gradient_weights_y(state, col_idx, lower_lyr)
    other = _cell_gradient_weights_y(state, col_idx, upper_lyr)
    merged: dict[tuple[int, int], torch.Tensor] = {}
    for src_col, src_lyr, weight in weights + other:
        key = (src_col, src_lyr)
        merged[key] = merged.get(key, torch.zeros((), dtype=state.dtype, device=state.device)) + 0.5 * weight
    return [(src_col, src_lyr, weight) for (src_col, src_lyr), weight in merged.items()]


def _average_gradient_weights_z(
    state: AtmState2D,
    left_col: int,
    right_col: int,
    lyr_idx: int,
) -> list[tuple[int, int, torch.Tensor]]:
    weights = _cell_gradient_weights_z(state, left_col, lyr_idx)
    other = _cell_gradient_weights_z(state, right_col, lyr_idx)
    merged: dict[tuple[int, int], torch.Tensor] = {}
    for src_col, src_lyr, weight in weights + other:
        key = (src_col, src_lyr)
        merged[key] = merged.get(key, torch.zeros((), dtype=state.dtype, device=state.device)) + 0.5 * weight
    return [(src_col, src_lyr, weight) for (src_col, src_lyr), weight in merged.items()]


def _cell_gradient_weights_y(
    state: AtmState2D,
    col_idx: int,
    lyr_idx: int,
) -> list[tuple[int, int, torch.Tensor]]:
    if state.ncol < 2:
        return []
    if col_idx == 0:
        denom = state.dx2v[0]
        return [(0, lyr_idx, -1.0 / denom), (1, lyr_idx, 1.0 / denom)]
    if col_idx == state.ncol - 1:
        denom = state.dx2v[-1]
        return [(state.ncol - 2, lyr_idx, -1.0 / denom), (state.ncol - 1, lyr_idx, 1.0 / denom)]
    denom = state.x2v[col_idx + 1] - state.x2v[col_idx - 1]
    return [(col_idx - 1, lyr_idx, -1.0 / denom), (col_idx + 1, lyr_idx, 1.0 / denom)]


def _cell_gradient_weights_z(
    state: AtmState2D,
    col_idx: int,
    lyr_idx: int,
) -> list[tuple[int, int, torch.Tensor]]:
    if state.nlyr < 2:
        return []
    if lyr_idx == 0:
        denom = state.dx1v[0]
        return [(col_idx, 0, -1.0 / denom), (col_idx, 1, 1.0 / denom)]
    if lyr_idx == state.nlyr - 1:
        denom = state.dx1v[-1]
        return [(col_idx, state.nlyr - 2, -1.0 / denom), (col_idx, state.nlyr - 1, 1.0 / denom)]
    denom = state.x1v[lyr_idx + 1] - state.x1v[lyr_idx - 1]
    return [(col_idx, lyr_idx - 1, -1.0 / denom), (col_idx, lyr_idx + 1, 1.0 / denom)]


def _as_vertical_face_scalar(value: torch.Tensor, state: AtmState2D) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=state.dtype, device=state.device)
    if tensor.dim() == 0:
        return tensor.expand(state.ncol, state.nlyr - 1)
    if tensor.dim() == 1 and tensor.numel() == state.nlyr - 1:
        return tensor.unsqueeze(0).expand(state.ncol, state.nlyr - 1)
    if tensor.shape != (state.ncol, state.nlyr - 1):
        raise ValueError("vertical-face tensor must have shape (ncol, nlyr - 1)")
    return tensor


def _as_horizontal_face_scalar(value: torch.Tensor, state: AtmState2D) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=state.dtype, device=state.device)
    if tensor.dim() == 0:
        return tensor.expand(state.ncol - 1, state.nlyr)
    if tensor.dim() == 1 and tensor.numel() == state.nlyr:
        return tensor.unsqueeze(0).expand(state.ncol - 1, state.nlyr)
    if tensor.shape != (state.ncol - 1, state.nlyr):
        raise ValueError("horizontal-face tensor must have shape (ncol - 1, nlyr)")
    return tensor


def _as_center_scalar(value: torch.Tensor, state: AtmState2D) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=state.dtype, device=state.device)
    if tensor.dim() == 0:
        return tensor.expand(state.ncol, state.nlyr)
    if tensor.shape != (state.ncol, state.nlyr):
        raise ValueError("centered tensor must have shape (ncol, nlyr)")
    return tensor


def _as_vertical_face_matrix(value: torch.Tensor, state: AtmState2D) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=state.dtype, device=state.device)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0).expand(state.ncol, -1, -1, -1)
    if tensor.shape != (state.ncol, state.nlyr - 1, state.nspecies, state.nspecies):
        raise ValueError(
            "binary_diffusion must have shape (ncol, nlyr - 1, nspecies, nspecies)"
        )
    return tensor


def _interface_thermo(
    state: AtmState2D,
    molecular_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = _mole_fraction(state.concentration)
    xmid = 0.5 * (x[:, :-1] + x[:, 1:])
    tmid = 0.5 * (state.temperature[:, :-1] + state.temperature[:, 1:])
    wtm_mid = torch.einsum("czi,i->cz", xmid, molecular_weights)
    dwtm = molecular_weights.view(1, 1, -1) - wtm_mid.unsqueeze(-1)
    return xmid, tmid, dwtm


def _mole_fraction(concentration: torch.Tensor) -> torch.Tensor:
    return concentration / concentration.sum(dim=-1, keepdim=True).clamp_min(1.0e-300)
