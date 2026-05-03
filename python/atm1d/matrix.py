from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BlockTridiagonalMatrix:
    lower_blocks: tuple[torch.Tensor, ...]
    diag_blocks: tuple[torch.Tensor, ...]
    upper_blocks: tuple[torch.Tensor, ...]
    lower_packed: torch.Tensor
    diag_packed: torch.Tensor
    upper_packed: torch.Tensor

    @classmethod
    def from_dense(
        cls, lower: torch.Tensor, diag: torch.Tensor, upper: torch.Tensor
    ) -> "BlockTridiagonalMatrix":
        _validate_packed(lower, "lower")
        _validate_packed(diag, "diag")
        _validate_packed(upper, "upper")
        if lower.shape != diag.shape or upper.shape != diag.shape:
            raise ValueError("lower, diag, and upper must share the same shape")

        return cls(
            lower_blocks=_to_sparse_tuple(lower),
            diag_blocks=_to_sparse_tuple(diag),
            upper_blocks=_to_sparse_tuple(upper),
            lower_packed=lower,
            diag_packed=diag,
            upper_packed=upper,
        )

    @property
    def nz(self) -> int:
        return int(self.diag_packed.size(0))

    @property
    def nspecies(self) -> int:
        return int(self.diag_packed.size(1))

    @property
    def device(self) -> torch.device:
        return self.diag_packed.device

    @property
    def dtype(self) -> torch.dtype:
        return self.diag_packed.dtype

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("x must have shape (nz, nspecies)")
        if x.shape != (self.nz, self.nspecies):
            raise ValueError("x shape must match matrix dimensions")

        out = torch.einsum("bij,bj->bi", self.diag_packed, x)
        if self.nz > 1:
            out[1:] += torch.einsum("bij,bj->bi", self.lower_packed[1:], x[:-1])
            out[:-1] += torch.einsum("bij,bj->bi", self.upper_packed[:-1], x[1:])
        return out

    def add_diagonal(self, blocks: torch.Tensor) -> "BlockTridiagonalMatrix":
        if blocks.shape != self.diag_packed.shape:
            raise ValueError("blocks must match the packed diagonal shape")
        return BlockTridiagonalMatrix.from_dense(
            self.lower_packed,
            self.diag_packed + blocks,
            self.upper_packed,
        )


def add_block_tridiagonal(
    *matrices: BlockTridiagonalMatrix,
) -> BlockTridiagonalMatrix:
    if not matrices:
        raise ValueError("at least one matrix is required")
    lower = sum(matrix.lower_packed for matrix in matrices)
    diag = sum(matrix.diag_packed for matrix in matrices)
    upper = sum(matrix.upper_packed for matrix in matrices)
    return BlockTridiagonalMatrix.from_dense(lower, diag, upper)


def _validate_packed(blocks: torch.Tensor, name: str) -> None:
    if blocks.dim() != 3:
        raise ValueError(f"{name} must have shape (nz, nspecies, nspecies)")
    if blocks.size(1) != blocks.size(2):
        raise ValueError(f"{name} blocks must be square")


def _to_sparse_tuple(blocks: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(block.to_sparse_coo().coalesce() for block in blocks.unbind(0))
