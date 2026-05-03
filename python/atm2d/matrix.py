from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SparseSystemMatrix:
    ncol: int
    nlyr: int
    nspecies: int
    global_csr: torch.Tensor
    rhs_override_mask: torch.Tensor | None = None
    rhs_override_values: torch.Tensor | None = None

    @classmethod
    def from_dense(
        cls,
        matrix: torch.Tensor,
        *,
        ncol: int,
        nlyr: int,
        nspecies: int,
        rhs_override_mask: torch.Tensor | None = None,
        rhs_override_values: torch.Tensor | None = None,
    ) -> "SparseSystemMatrix":
        dense = torch.as_tensor(matrix)
        expected = ncol * nlyr * nspecies
        if dense.shape != (expected, expected):
            raise ValueError("dense matrix has incompatible shape")
        coo = dense.to_sparse_coo().coalesce()
        return cls.from_coo(
            coo.indices(),
            coo.values(),
            ncol=ncol,
            nlyr=nlyr,
            nspecies=nspecies,
            device=dense.device,
            dtype=dense.dtype,
            rhs_override_mask=rhs_override_mask,
            rhs_override_values=rhs_override_values,
        )

    @classmethod
    def from_coo(
        cls,
        indices: torch.Tensor,
        values: torch.Tensor,
        *,
        ncol: int,
        nlyr: int,
        nspecies: int,
        device: torch.device,
        dtype: torch.dtype,
        rhs_override_mask: torch.Tensor | None = None,
        rhs_override_values: torch.Tensor | None = None,
    ) -> "SparseSystemMatrix":
        size = ncol * nlyr * nspecies
        index_tensor = torch.as_tensor(indices, dtype=torch.int64, device=device)
        value_tensor = torch.as_tensor(values, dtype=dtype, device=device)
        if index_tensor.numel() == 0:
            index_tensor = torch.empty((2, 0), dtype=torch.int64, device=device)
            value_tensor = torch.empty((0,), dtype=dtype, device=device)
        if index_tensor.shape[0] != 2:
            raise ValueError("indices must have shape (2, nnz)")

        if rhs_override_mask is not None:
            rhs_override_mask = _normalize_rhs_tensor(
                rhs_override_mask, (ncol, nlyr, nspecies), dtype=torch.bool, device=device
            )
        if rhs_override_values is not None:
            rhs_override_values = _normalize_rhs_tensor(
                rhs_override_values, (ncol, nlyr, nspecies), dtype=dtype, device=device
            )
        if (rhs_override_mask is None) != (rhs_override_values is None):
            raise ValueError("rhs override mask and values must be provided together")

        coo = torch.sparse_coo_tensor(
            index_tensor,
            value_tensor,
            (size, size),
            dtype=dtype,
            device=device,
        ).coalesce()
        return cls(
            ncol=ncol,
            nlyr=nlyr,
            nspecies=nspecies,
            global_csr=coo.to_sparse_csr(),
            rhs_override_mask=rhs_override_mask,
            rhs_override_values=rhs_override_values,
        )

    @property
    def device(self) -> torch.device:
        return self.global_csr.device

    @property
    def dtype(self) -> torch.dtype:
        return self.global_csr.dtype

    @property
    def nstate(self) -> int:
        return self.ncol * self.nlyr * self.nspecies

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        rhs = _normalize_rhs_tensor(
            x, (self.ncol, self.nlyr, self.nspecies), dtype=self.dtype, device=self.device
        )
        flat = rhs.reshape(-1, 1)
        out = torch.sparse.mm(self.global_csr, flat).reshape(self.ncol, self.nlyr, self.nspecies)
        return out

    def add_diagonal(self, blocks: torch.Tensor) -> "SparseSystemMatrix":
        block_tensor = torch.as_tensor(blocks, dtype=self.dtype, device=self.device)
        if block_tensor.shape != (self.ncol, self.nlyr, self.nspecies, self.nspecies):
            raise ValueError("blocks must have shape (ncol, nlyr, nspecies, nspecies)")
        rows: list[torch.Tensor] = []
        cols: list[torch.Tensor] = []
        vals: list[torch.Tensor] = []
        nz = block_tensor.nonzero(as_tuple=False)
        if nz.numel() > 0:
            row_ids = flatten_state_index(nz[:, 0], nz[:, 1], nz[:, 2], self.nlyr, self.nspecies)
            col_ids = flatten_state_index(nz[:, 0], nz[:, 1], nz[:, 3], self.nlyr, self.nspecies)
            rows.append(row_ids)
            cols.append(col_ids)
            vals.append(block_tensor[nz[:, 0], nz[:, 1], nz[:, 2], nz[:, 3]])
        return add_sparse_system_matrices(
            self,
            SparseSystemMatrix.from_coo(
                torch.stack([torch.cat(rows), torch.cat(cols)]) if rows else torch.empty((2, 0), dtype=torch.int64, device=self.device),
                torch.cat(vals) if vals else torch.empty((0,), dtype=self.dtype, device=self.device),
                ncol=self.ncol,
                nlyr=self.nlyr,
                nspecies=self.nspecies,
                device=self.device,
                dtype=self.dtype,
            ),
        ).with_rhs_overrides(self.rhs_override_mask, self.rhs_override_values)

    def with_rhs_overrides(
        self,
        rhs_override_mask: torch.Tensor | None,
        rhs_override_values: torch.Tensor | None,
    ) -> "SparseSystemMatrix":
        return SparseSystemMatrix.from_coo(
            self.global_csr.to_sparse_coo().indices(),
            self.global_csr.to_sparse_coo().values(),
            ncol=self.ncol,
            nlyr=self.nlyr,
            nspecies=self.nspecies,
            device=self.device,
            dtype=self.dtype,
            rhs_override_mask=rhs_override_mask,
            rhs_override_values=rhs_override_values,
        )

    def replace_rows(
        self,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        values: torch.Tensor,
        *,
        rhs_override_mask: torch.Tensor | None = None,
        rhs_override_values: torch.Tensor | None = None,
    ) -> "SparseSystemMatrix":
        coo = self.global_csr.to_sparse_coo().coalesce()
        rows = coo.indices()[0]
        keep = ~torch.isin(rows, row_ids.to(device=self.device, dtype=torch.int64))
        new_indices = torch.cat(
            [
                coo.indices()[:, keep],
                torch.stack(
                    [
                        row_ids.to(device=self.device, dtype=torch.int64),
                        col_ids.to(device=self.device, dtype=torch.int64),
                    ]
                ),
            ],
            dim=1,
        )
        new_values = torch.cat(
            [
                coo.values()[keep],
                torch.as_tensor(values, dtype=self.dtype, device=self.device),
            ]
        )
        return SparseSystemMatrix.from_coo(
            new_indices,
            new_values,
            ncol=self.ncol,
            nlyr=self.nlyr,
            nspecies=self.nspecies,
            device=self.device,
            dtype=self.dtype,
            rhs_override_mask=rhs_override_mask,
            rhs_override_values=rhs_override_values,
        )

    def apply_rhs_overrides(self, rhs: torch.Tensor) -> torch.Tensor:
        rhs_3d = _normalize_rhs_tensor(
            rhs, (self.ncol, self.nlyr, self.nspecies), dtype=self.dtype, device=self.device
        )
        if self.rhs_override_mask is None:
            return rhs_3d
        return torch.where(self.rhs_override_mask, self.rhs_override_values, rhs_3d)


def add_sparse_system_matrices(*matrices: SparseSystemMatrix) -> SparseSystemMatrix:
    if not matrices:
        raise ValueError("at least one matrix is required")
    base = matrices[0]
    for other in matrices[1:]:
        if (other.ncol, other.nlyr, other.nspecies) != (base.ncol, base.nlyr, base.nspecies):
            raise ValueError("all matrices must share dimensions")
    indices = [matrix.global_csr.to_sparse_coo().indices() for matrix in matrices]
    values = [matrix.global_csr.to_sparse_coo().values() for matrix in matrices]
    return SparseSystemMatrix.from_coo(
        torch.cat(indices, dim=1),
        torch.cat(values),
        ncol=base.ncol,
        nlyr=base.nlyr,
        nspecies=base.nspecies,
        device=base.device,
        dtype=base.dtype,
    )


def flatten_state_index(
    col_idx: torch.Tensor | int,
    lyr_idx: torch.Tensor | int,
    species_idx: torch.Tensor | int,
    nlyr: int,
    nspecies: int,
) -> torch.Tensor:
    col_tensor = torch.as_tensor(col_idx, dtype=torch.int64)
    lyr_tensor = torch.as_tensor(lyr_idx, dtype=torch.int64, device=col_tensor.device)
    spc_tensor = torch.as_tensor(species_idx, dtype=torch.int64, device=col_tensor.device)
    return ((col_tensor * nlyr + lyr_tensor) * nspecies) + spc_tensor


def _normalize_rhs_tensor(
    tensor: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    out = torch.as_tensor(tensor, dtype=dtype, device=device)
    if out.shape != shape:
        raise ValueError(f"tensor must have shape {shape}")
    return out
