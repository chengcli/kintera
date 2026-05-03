from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch
import pydisort

from .state import ColumnState1D

AVOGADRO = 6.02214076e23


@dataclass
class RadiativeTransferResult:
    wavelength: torch.Tensor
    optical_depth: torch.Tensor
    average_intensity: torch.Tensor
    actinic_flux: torch.Tensor


def compute_actinic_flux_disort(
    photo_chem,
    state: ColumnState1D,
    top_flux: torch.Tensor,
    *,
    mu0: float = 1.0,
    phi0: float = 0.0,
    surface_albedo: float = 0.0,
    concentration_unit: str = "mol_m3",
    actinic_scale: float = 4.0 * pi,
    nstreams: int = 4,
) -> RadiativeTransferResult:
    photolysis = photo_chem.module("photolysis")
    wavelength = photolysis.buffer("wavelength").to(
        device=state.device, dtype=state.dtype
    )
    nwave = int(wavelength.numel())

    top_flux = torch.as_tensor(top_flux, dtype=state.dtype, device=state.device)
    if top_flux.dim() != 1 or top_flux.numel() != nwave:
        raise ValueError("top_flux must have shape (nwave,)")

    sigma_total = _total_cross_section_by_species(photo_chem, state.temperature, wavelength)
    number_density = _to_number_density_cm3(state.concentration, concentration_unit)
    absorption = (number_density.unsqueeze(1) * sigma_total).sum(dim=-1)

    dz = state.z[1:] - state.z[:-1]
    cell_width = torch.empty_like(state.z)
    cell_width[0] = dz[0]
    if state.nz > 2:
        cell_width[1:-1] = 0.5 * (dz[1:] + dz[:-1])
    cell_width[-1] = dz[-1]
    optical_depth = absorption * cell_width.unsqueeze(-1)

    op = pydisort.DisortOptions().flags("onlyfl,lamber")
    op.ds().nlyr = state.nz
    op.ds().nstr = nstreams
    op.ds().nmom = nstreams
    op.ds().nphase = 1
    op.nwave(nwave).ncol(1)
    ds = pydisort.Disort(op)

    prop = torch.zeros((nwave, 1, state.nz, 2), dtype=state.dtype, device=state.device)
    prop[:, 0, :, 0] = optical_depth.transpose(0, 1)
    flux = ds.forward(
        prop,
        umu0=torch.tensor([mu0], dtype=state.dtype, device=state.device),
        phi0=torch.tensor([phi0], dtype=state.dtype, device=state.device),
        fbeam=top_flux[:, None],
        albedo=torch.full((nwave, 1), surface_albedo, dtype=state.dtype, device=state.device),
    )
    del flux

    gathered = ds.gather_flx()
    average_intensity_levels = gathered[..., pydisort.kIUAVG].squeeze(1)
    average_intensity = 0.5 * (
        average_intensity_levels[:, :-1] + average_intensity_levels[:, 1:]
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
    sigma = torch.zeros(
        (temperature.numel(), wavelength.numel(), len(species)),
        dtype=temperature.dtype,
        device=temperature.device,
    )

    for rxn_idx, reaction in enumerate(reactions):
        reactants = reaction.reactants()
        if len(reactants) != 1:
            raise ValueError("photolysis reactions must have exactly one absorber species")
        absorber = next(iter(reactants.keys()))
        absorber_idx = species_index[absorber]
        sigma_rxn = photolysis.interp_cross_section(rxn_idx, wavelength, temperature).sum(-1)
        sigma[:, :, absorber_idx] += sigma_rxn

    return sigma


def _to_number_density_cm3(
    concentration: torch.Tensor, concentration_unit: str
) -> torch.Tensor:
    if concentration_unit == "mol_m3":
        return concentration * (AVOGADRO * 1.0e-6)
    if concentration_unit == "molecules_cm3":
        return concentration
    raise ValueError("concentration_unit must be 'mol_m3' or 'molecules_cm3'")
