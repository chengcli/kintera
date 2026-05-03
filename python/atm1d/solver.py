from __future__ import annotations

import torch

from .matrix import BlockTridiagonalMatrix


def solve_block_tridiagonal(
    matrix: BlockTridiagonalMatrix, rhs: torch.Tensor
) -> torch.Tensor:
    if rhs.dim() == 2:
        rhs_work = rhs.unsqueeze(-1)
        squeeze = True
    elif rhs.dim() == 3:
        rhs_work = rhs
        squeeze = False
    else:
        raise ValueError("rhs must have shape (nz, nspecies) or (nz, nspecies, nrhs)")

    if rhs_work.size(0) != matrix.nz or rhs_work.size(1) != matrix.nspecies:
        raise ValueError("rhs shape must match matrix dimensions")

    diag_eff = []
    rhs_eff = []
    upper_solve = []

    diag_eff.append(matrix.diag_packed[0])
    rhs_eff.append(rhs_work[0])
    upper_solve.append(torch.linalg.solve(diag_eff[0], matrix.upper_packed[0]))

    for idx in range(1, matrix.nz):
        lower = matrix.lower_packed[idx]
        rhs_corr = lower @ torch.linalg.solve(diag_eff[idx - 1], rhs_eff[idx - 1])
        diag_corr = lower @ upper_solve[idx - 1]
        diag_now = matrix.diag_packed[idx] - diag_corr
        diag_eff.append(diag_now)
        rhs_eff.append(rhs_work[idx] - rhs_corr)
        if idx < matrix.nz - 1:
            upper_solve.append(torch.linalg.solve(diag_now, matrix.upper_packed[idx]))

    solution = [None] * matrix.nz
    solution[-1] = torch.linalg.solve(diag_eff[-1], rhs_eff[-1])
    for idx in range(matrix.nz - 2, -1, -1):
        solution[idx] = torch.linalg.solve(
            diag_eff[idx], rhs_eff[idx] - matrix.upper_packed[idx] @ solution[idx + 1]
        )

    stacked = torch.stack(solution, dim=0)
    if squeeze:
        return stacked.squeeze(-1)
    return stacked


def solve_block_tridiagonal_cpu(
    matrix: BlockTridiagonalMatrix, rhs: torch.Tensor
) -> torch.Tensor:
    if matrix.device.type != "cpu":
        raise ValueError("matrix must live on CPU")
    return solve_block_tridiagonal(matrix, rhs)


def solve_block_tridiagonal_cuda(
    matrix: BlockTridiagonalMatrix, rhs: torch.Tensor
) -> torch.Tensor:
    if matrix.device.type != "cuda":
        raise ValueError("matrix must live on CUDA")
    return solve_block_tridiagonal(matrix, rhs)
