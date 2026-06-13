"""Combine a list of :class:`LocalSourceTerm` into one linearization.

Two assemblers:

  - :func:`build_source_linearization` — sums per-term tendency and
    Jacobian into a single ``LocalSourceLinearization``. When
    ``charge_balance_indices`` is provided, folds the implicit
    ``E = Σ(cations)`` constraint into the Jacobian via
    :func:`fold_charge_balance_into_jacobian`.
  - :func:`build_source_global_operator` — sums per-term *non-local*
    Jacobians (e.g. the shifted-sublimation source) into one sparse
    ``SparseSystemMatrix``.
"""

from __future__ import annotations

import torch

from ..atm_state2d import AtmState2D
from ..matrix import SparseSystemMatrix, add_sparse_system_matrices
from .charge_balance import fold_charge_balance_into_jacobian
from .protocol import LocalSourceLinearization, LocalSourceTerm


def build_source_linearization(
    state: AtmState2D,
    source_terms: list[LocalSourceTerm],
    charge_balance_indices: "tuple[list[int], int] | None" = None,
) -> LocalSourceLinearization:
    """Combine local source-term tendencies and Jacobian blocks.

    When ``charge_balance_indices=(cation_indices, e_index)`` is provided,
    fold the implicit constraint ``E = Σ(cations)`` into the Jacobian via
    :func:`fold_charge_balance_into_jacobian`.

    Without the fold the BE Newton sees ``c_E`` as an independent variable
    and lags it via post-Newton charge-balance reset (Picard iteration),
    which causes the cation cascade to keep growing until E catches up.
    """

    tendency = torch.zeros_like(state.concentration)
    jacobian = torch.zeros(
        (state.ncol, state.nlyr, state.nspecies, state.nspecies),
        dtype=state.dtype,
        device=state.device,
    )
    for term in source_terms:
        linearization = term.linearize(state)
        term_tendency = torch.as_tensor(
            linearization.tendency, dtype=state.dtype, device=state.device
        )
        term_jacobian = torch.as_tensor(
            linearization.jacobian, dtype=state.dtype, device=state.device
        )
        if term_tendency.shape != state.concentration.shape:
            raise ValueError("source tendency must match state concentration shape")
        if term_jacobian.shape != jacobian.shape:
            raise ValueError(
                "source jacobian must have shape (ncol, nlyr, nspecies, nspecies)"
            )
        tendency = tendency + term_tendency
        jacobian = jacobian + term_jacobian

    if charge_balance_indices is not None:
        cation_idx_list, e_index = charge_balance_indices
        jacobian = fold_charge_balance_into_jacobian(
            jacobian, cation_indices=cation_idx_list, e_index=e_index
        )
    return LocalSourceLinearization(tendency=tendency, jacobian=jacobian)


def build_source_global_operator(
    state: AtmState2D,
    source_terms: list[LocalSourceTerm],
) -> SparseSystemMatrix | None:
    """Combine optional non-local source Jacobians into a global operator."""

    matrices: list[SparseSystemMatrix] = []
    for term in source_terms:
        global_operator = getattr(term, "global_operator", None)
        if global_operator is None:
            continue
        matrix = global_operator(state)
        if matrix is not None:
            matrices.append(matrix)
    if not matrices:
        return None
    return add_sparse_system_matrices(*matrices)


__all__ = ["build_source_linearization", "build_source_global_operator"]
