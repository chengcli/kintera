from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch
import pydisort

from .atm_state2d import AtmState2D

AVOGADRO = 6.02214076e23


@dataclass
class RadiativeTransferResult:
    wavelength: torch.Tensor
    optical_depth: torch.Tensor
    average_intensity: torch.Tensor
    actinic_flux: torch.Tensor


def compute_actinic_flux_disort(
    photo_chem,
    state: AtmState2D,
    top_flux: torch.Tensor,
    *,
    mu0: float | torch.Tensor = 1.0,
    phi0: float | torch.Tensor = 0.0,
    surface_albedo: float | torch.Tensor = 0.0,
    concentration_unit: str = "mol_m3",
    actinic_scale: float = 4.0 * pi,
    nstreams: int = 4,
) -> RadiativeTransferResult:
    photolysis = photo_chem.module("photolysis")
    wavelength = photolysis.buffer("wavelength").to(device=state.device, dtype=state.dtype)
    nwave = int(wavelength.numel())

    top_flux_2d = _as_wave_column(top_flux, nwave, state.ncol, dtype=state.dtype, device=state.device)
    sigma_total = _total_cross_section_by_species(photo_chem, state.temperature, wavelength)
    number_density = _to_number_density_cm3(state.concentration, concentration_unit)
    absorption = (number_density.unsqueeze(2) * sigma_total).sum(dim=-1)

    optical_depth = absorption * state.dx1f.view(1, state.nlyr, 1)

    op = pydisort.DisortOptions().flags("onlyfl,lamber")
    op.ds().nlyr = state.nlyr
    op.ds().nstr = nstreams
    op.ds().nmom = nstreams
    op.ds().nphase = 1
    op.nwave(nwave).ncol(state.ncol)
    ds = pydisort.Disort(op)

    prop = torch.zeros((nwave, state.ncol, state.nlyr, 2), dtype=state.dtype, device=state.device)
    prop[..., 0] = optical_depth.permute(2, 0, 1)
    ds.forward(
        prop,
        umu0=_as_column(mu0, state.ncol, dtype=state.dtype, device=state.device),
        phi0=_as_column(phi0, state.ncol, dtype=state.dtype, device=state.device),
        fbeam=top_flux_2d,
        albedo=_as_wave_column(surface_albedo, nwave, state.ncol, dtype=state.dtype, device=state.device),
    )

    gathered = ds.gather_flx()
    average_intensity_levels = gathered[..., pydisort.kIUAVG]
    average_intensity = 0.5 * (
        average_intensity_levels[:, :, :-1] + average_intensity_levels[:, :, 1:]
    )
    actinic_flux = actinic_scale * average_intensity

    return RadiativeTransferResult(
        wavelength=wavelength,
        optical_depth=optical_depth,
        average_intensity=average_intensity,
        actinic_flux=actinic_flux,
    )


def _total_cross_section_by_species(
    photo_chem,
    temperature: torch.Tensor,
    wavelength: torch.Tensor,
) -> torch.Tensor:
    photolysis = photo_chem.module("photolysis")
    reactions = photo_chem.options.photolysis().reactions()
    species = photo_chem.options.species()
    species_index = {name: idx for idx, name in enumerate(species)}
    ncol, nz = temperature.shape

    sigma = torch.zeros(
        (ncol, nz, wavelength.numel(), len(species)),
        dtype=temperature.dtype,
        device=temperature.device,
    )

    temp_flat = temperature.reshape(-1)
    for rxn_idx, reaction in enumerate(reactions):
        reactants = reaction.reactants()
        if len(reactants) != 1:
            raise ValueError("photolysis reactions must have exactly one absorber species")
        absorber = next(iter(reactants.keys()))
        absorber_idx = species_index[absorber]
        sigma_rxn = photolysis.interp_cross_section(rxn_idx, wavelength, temp_flat).select(-1, 0)
        sigma[:, :, :, absorber_idx] += sigma_rxn.reshape(ncol, nz, wavelength.numel())

    return sigma


def _to_number_density_cm3(concentration: torch.Tensor, concentration_unit: str) -> torch.Tensor:
    if concentration_unit == "mol_m3":
        return concentration * (AVOGADRO * 1.0e-6)
    if concentration_unit == "molecules_cm3":
        return concentration
    raise ValueError("concentration_unit must be 'mol_m3' or 'molecules_cm3'")


def _as_column(value: float | torch.Tensor, ncol: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.dim() == 0:
        return tensor.expand(ncol)
    if tensor.shape != (ncol,):
        raise ValueError("column vector must have shape (ncol,)")
    return tensor


def _as_wave_column(
    value: float | torch.Tensor,
    nwave: int,
    ncol: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.dim() == 0:
        return tensor.expand(nwave, ncol)
    if tensor.dim() == 1:
        if tensor.numel() == nwave:
            return tensor.unsqueeze(-1).expand(nwave, ncol)
        if tensor.numel() == ncol:
            return tensor.unsqueeze(0).expand(nwave, ncol)
    if tensor.shape != (nwave, ncol):
        raise ValueError("wave-column tensor must have shape (nwave, ncol)")
    return tensor
