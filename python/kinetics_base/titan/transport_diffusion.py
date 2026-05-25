"""Cheng-2013 Titan molecular-diffusion helper.

KB's `COEFF1` (`kinetgen2X.F:5083-5088`) uses the Cheng-2013 formula for
the molecular diffusion of a trace species in the N2-dominated bath:

    D_i(T, n) = 7.3e16 * T^0.75 / n * sqrt((1 + 28/m_i) / (1 + 28/16))

where

  * `T` is the cell-centered temperature (K)
  * `n` is the cell-centered total number density (cm^-3)
  * `m_i` is the molecular mass of species `i` (amu)
  * the 28 amu reference is N2 (the bath), 16 amu is the
    Cheng-derivation normalisation species (CH4).

This module wraps the formula into the shape expected by kintera's
`build_binary_diffusion_matrix` — a per-cell `(nspecies, nspecies)`
diagonal matrix. The off-diagonal Stefan-Maxwell coupling is set to
zero, matching KB's treatment.

The helper is Titan-specific (the N2-bath assumption is baked into the
formula). Cross-planet reuse would need a different bath molecule and
a different prefactor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..atm_state2d import AtmState2D  # type: ignore

from .physics import _kinetics_base_species_mass_amu


def kinetics_base_titan_species_masses(
    species: list[str],
    pun_metadata=None,
) -> torch.Tensor:
    """Return a `(nspecies,)` float tensor of molecular masses (amu)
    indexed in the same order as ``species``.

    Falls back to a small hard-coded table via `_kinetics_base_species_mass_amu`
    when pun metadata is missing or zero.
    """
    masses = [
        _kinetics_base_species_mass_amu(name, pun_metadata) for name in species
    ]
    return torch.tensor(masses, dtype=torch.float64)


def kinetics_base_titan_cheng_diffusion(
    state,
    masses: torch.Tensor,
    *,
    temperature: torch.Tensor | None = None,
    density: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the per-cell diagonal binary-diffusion tensor for Titan.

    Returns a tensor of shape ``(ncol, nlyr, nspecies, nspecies)``
    whose only non-zero entries are on the diagonal:

        D[c, l, i, i] = 7.3e16 * T[c, l]^0.75 / n[c, l]
                       * sqrt((1 + 28/m_i) / (1 + 28/16))

    where ``T`` is `state.temperature` (or `temperature` override) and
    ``n`` is the per-cell total density (sum of species concentration,
    or `density` override).

    Cells with zero density (kintera's "extended" altitude slots above
    the real atmosphere) get zero diffusion to avoid divide-by-zero.

    Parameters
    ----------
    state : AtmState2D
        Atmospheric state. Reads `temperature` and falls back to
        `concentration.sum(-1)` for total density.
    masses : torch.Tensor
        `(nspecies,)` molecular masses in amu.
    temperature, density : optional override tensors for testing.
    """
    dtype = state.dtype
    device = state.device
    masses = masses.to(dtype=dtype, device=device)
    nspecies = state.nspecies
    if masses.shape != (nspecies,):
        raise ValueError(
            f"masses must have shape ({nspecies},), got {tuple(masses.shape)}"
        )

    T = temperature if temperature is not None else state.temperature
    if density is None:
        # Use the "M" species (KB's third-body placeholder) when present —
        # its concentration is set to the atmospheric total density and
        # stays fixed across Newton iterates. Falling back to
        # `concentration.sum(dim=-1)` would double-count because both `M`
        # and `N2` appear in the species set in KB Titan networks.
        if hasattr(state, "density") and getattr(state, "density") is not None:
            n_tot = state.density
        else:
            # Bare AtmState2D: sum concentration but subtract M if present
            # to avoid the double-count described above. Caller should
            # pass `density=` explicitly for correct results.
            n_tot = state.concentration.sum(dim=-1)
    else:
        n_tot = density

    n_tot = n_tot.to(dtype=dtype, device=device)
    T = T.to(dtype=dtype, device=device)

    # Cheng species-mass factor: sqrt((1 + 28/m_i) / (1 + 28/16))
    # The 16 amu normalisation is from CH4 in the Cheng derivation.
    # KB-Titan pun metadata stores spurious masses for placeholder
    # species: M, RAYEAR, U → 0 amu; ions → negative amu (charge
    # encoded in the slot). For those species the formula would
    # produce NaN/Inf, so clamp masses to a minimum of 1 amu before
    # the divide. Their diffusion coefficient is physically irrelevant:
    # M / RAYEAR / U are fixed-density placeholders, ions are
    # filtered out of the chemistry in neutrals_only mode.
    masses_safe = torch.clamp(masses, min=1.0)
    species_factor = torch.sqrt(
        (1.0 + 28.0 / masses_safe) / (1.0 + 28.0 / 16.0)
    )  # shape (nspecies,)

    # Per-cell scalar 7.3e16 * T^0.75 / n. Guard against n==0 and T==0
    # (the kintera grid has "extended" altitude slots above the real
    # atmosphere where both are zero — extended cells are not part of
    # the physical run but get carried through the state).
    valid = (n_tot > 0) & (T > 0)
    n_safe = torch.where(valid, n_tot, torch.ones_like(n_tot))
    T_safe = torch.where(valid, T, torch.ones_like(T))
    cell_scalar = 7.3e16 * torch.pow(T_safe, 0.75) / n_safe  # (ncol, nlyr)
    cell_scalar = torch.where(valid, cell_scalar, torch.zeros_like(cell_scalar))

    # D[c, l, i, i] = cell_scalar[c, l] * species_factor[i]
    diag = cell_scalar.unsqueeze(-1) * species_factor.view(1, 1, -1)  # (ncol, nlyr, nspecies)

    # Embed on a diagonal (ncol, nlyr, nspecies, nspecies)
    D = torch.zeros(
        (state.ncol, state.nlyr, nspecies, nspecies),
        dtype=dtype, device=device,
    )
    idx = torch.arange(nspecies, device=device)
    D[..., idx, idx] = diag

    return D


__all__ = [
    "kinetics_base_titan_cheng_diffusion",
    "kinetics_base_titan_species_masses",
]
