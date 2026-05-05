from __future__ import annotations

import numpy as np
import torch

from .matrix import SparseSystemMatrix


def solve_sparse_system(matrix: SparseSystemMatrix, rhs: torch.Tensor) -> torch.Tensor:
    """Solve ``A x = rhs`` for a state-shaped right-hand side.

    Parameters
    ----------
    matrix:
        Sparse operator assembled by ``atm2d``.
    rhs:
        Tensor with shape ``(ncol, nlyr, nspecies)``.

    Notes
    -----
    CPU solves reuse a cached SciPy factorization stored on the matrix
    instance. CUDA solves call the native cuSolver CSR binding.
    """
    rhs_3d = _normalize_rhs(rhs, matrix)
    rhs_3d = matrix.apply_rhs_overrides(rhs_3d)
    rhs_vec = rhs_3d.reshape(-1)
    sparse_matrix = matrix.global_csr
    if sparse_matrix.device.type == "cuda":
        from ..kintera import cuda_csr_solve_cusolver

        crow_indices, col_indices = matrix.cuda_csr_indices_int32()
        sol_vec = cuda_csr_solve_cusolver(
            crow_indices,
            col_indices,
            sparse_matrix.values(),
            rhs_vec,
            0.0,
            0,
        )
    else:
        solve_cpu = matrix.factorized_cpu_solver()
        sol_np = solve_cpu(rhs_vec.cpu().numpy())
        sol_vec = torch.from_numpy(np.asarray(sol_np)).to(device=matrix.device, dtype=matrix.dtype)
    return sol_vec.reshape(matrix.ncol, matrix.nlyr, matrix.nspecies)


def _normalize_rhs(rhs: torch.Tensor, matrix: SparseSystemMatrix) -> torch.Tensor:
    tensor = torch.as_tensor(rhs, dtype=matrix.dtype, device=matrix.device)
    if tensor.shape != (matrix.ncol, matrix.nlyr, matrix.nspecies):
        raise ValueError("rhs must have shape (ncol, nlyr, nspecies)")
    return tensor
