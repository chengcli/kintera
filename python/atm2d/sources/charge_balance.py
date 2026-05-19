"""Implicit charge-balance fold for the BE source Jacobian.

When a chemistry network includes an electron species ``E`` and a set of
cations, charge neutrality demands ``E = Σ(cations)``. Solvers can either
(a) treat ``E`` as a free Newton variable and reset it after each step
via Picard iteration, or (b) fold the constraint into the Jacobian so
the BE Newton sees it.

(a) lags the cation cascade — cations grow each step until ``E`` catches
up, which at large ``dt`` produces multi-OoM ion overshoots. (b) is
what KB's MARCH effectively does and is what we apply here.

The fold: for every species row ``i``, propagate the ``E`` column's
derivative into each cation column ``j`` via the implicit chain rule
``dc_E/dc_X+ = 1``::

    J[..., i, j] += J[..., i, e_index]   for each j in cation_indices

This is the single-line transform that, combined with apply_pins
recomputing ``E = Σ(cations) − Σ(anions)``, gives a self-consistent
charge balance every Newton iteration.
"""

from __future__ import annotations

import torch


def fold_charge_balance_into_jacobian(
    jacobian: torch.Tensor,
    *,
    cation_indices: list[int],
    e_index: int,
) -> torch.Tensor:
    """Return a new Jacobian with ``J[..., :, j] += J[..., :, e_index]`` for
    every ``j`` in ``cation_indices``.

    Parameters
    ----------
    jacobian:
        Tensor of shape ``(ncol, nlyr, nspecies, nspecies)``. Read-only.
    cation_indices:
        Integer indices of cation species. May be empty (returns input
        unmodified).
    e_index:
        Integer index of the electron species (``E``).

    Returns
    -------
    Tensor with the same shape; ``jacobian`` itself is not mutated.
    """
    if not cation_indices:
        return jacobian
    cation_idx = torch.tensor(
        cation_indices, dtype=torch.long, device=jacobian.device
    )
    delta = jacobian[..., e_index].clone()  # (ncol, nlyr, nspecies)
    out = jacobian.clone()
    out[..., cation_idx] = out[..., cation_idx] + delta.unsqueeze(-1)
    return out
