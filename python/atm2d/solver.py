from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch

from .matrix import SparseSystemMatrix


def solve_sparse_system(matrix: SparseSystemMatrix, rhs: torch.Tensor) -> torch.Tensor:
    rhs_3d = _normalize_rhs(rhs, matrix)
    rhs_3d = matrix.apply_rhs_overrides(rhs_3d)
    rhs_vec = rhs_3d.reshape(-1)
    sparse_matrix = matrix.global_csr
    if sparse_matrix.device.type == "cuda":
        sol_vec = torch.sparse.spsolve(sparse_matrix, rhs_vec)
    else:
        sparse_cpu = sparse_matrix.cpu()
        scipy_csr = scipy.sparse.csr_matrix(
            (
                sparse_cpu.values().numpy(),
                sparse_cpu.col_indices().numpy(),
                sparse_cpu.crow_indices().numpy(),
            ),
            shape=sparse_cpu.shape,
        )
        sol_np = scipy.sparse.linalg.spsolve(scipy_csr, rhs_vec.cpu().numpy())
        sol_vec = torch.from_numpy(np.asarray(sol_np)).to(device=matrix.device, dtype=matrix.dtype)
    return sol_vec.reshape(matrix.ncol, matrix.nlyr, matrix.nspecies)


def _normalize_rhs(rhs: torch.Tensor, matrix: SparseSystemMatrix) -> torch.Tensor:
    tensor = torch.as_tensor(rhs, dtype=matrix.dtype, device=matrix.device)
    if tensor.shape != (matrix.ncol, matrix.nlyr, matrix.nspecies):
        raise ValueError("rhs must have shape (ncol, nlyr, nspecies)")
    return tensor
