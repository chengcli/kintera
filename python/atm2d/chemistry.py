from __future__ import annotations

import torch


def build_chemistry_jacobian(
    kinetics,
    temperature: torch.Tensor,
    pressure: torch.Tensor,
    concentration: torch.Tensor,
    *,
    photo_inputs: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Build the species Jacobian from a `kintera.Kinetics` module.

    Returns a tensor with shape ``(ncol, nlyr, nspecies, nspecies)`` suitable
    for insertion into the diagonal of an implicit operator.
    """
    temp_2d, pressure_2d, conc_3d = _normalize_state_tensors(temperature, pressure, concentration)
    ncol, nz, nspecies = conc_3d.shape

    temp_flat = temp_2d.reshape(-1)
    pressure_flat = pressure_2d.reshape(-1)
    conc_flat = conc_3d.reshape(-1, nspecies)
    photo_flat = _flatten_photo_inputs(photo_inputs, ncol, nz) if photo_inputs is not None else None

    if photo_flat is None:
        rate, rc_ddc, rc_ddt = kinetics.forward(temp_flat, pressure_flat, conc_flat)
    else:
        rate, rc_ddc, rc_ddt = kinetics.forward(temp_flat, pressure_flat, conc_flat, photo_flat)
    cvol = torch.ones_like(temp_flat)
    jac_rxn = kinetics.jacobian(temp_flat, conc_flat, cvol, rate, rc_ddc, rc_ddt)
    stoich = kinetics.stoich.to(device=jac_rxn.device, dtype=jac_rxn.dtype)
    jac_species = torch.einsum("sr,brn->bsn", stoich, jac_rxn)
    return jac_species.reshape(ncol, nz, nspecies, nspecies)


def build_photochemistry_jacobian(
    photo_chem,
    temperature: torch.Tensor,
    concentration: torch.Tensor,
    actinic_flux: torch.Tensor,
) -> torch.Tensor:
    """Build the species Jacobian from a `kintera.PhotoChem` module.

    This helper explicitly calls
    ``photo_chem.module("photolysis").update_xs_diss_stacked(temp_flat)``
    before ``photo_chem.forward(...)``. Callers using ``PhotoChem.forward()``
    directly must do the same cache priming themselves.
    """
    temp_2d, _, conc_3d = _normalize_state_tensors(temperature, temperature, concentration)
    ncol, nz, nspecies = conc_3d.shape
    flux_3d = _normalize_actinic_flux(actinic_flux, ncol, nz, dtype=temp_2d.dtype, device=temp_2d.device)
    temp_flat = temp_2d.reshape(-1)
    photo_chem.module("photolysis").update_xs_diss_stacked(temp_flat)

    rate = photo_chem.forward(
        temp_flat,
        conc_3d.reshape(-1, nspecies),
        flux_3d.reshape(flux_3d.size(0), -1),
    )
    jac_rxn = photo_chem.jacobian(conc_3d.reshape(-1, nspecies), rate)
    stoich = photo_chem.stoich.to(device=jac_rxn.device, dtype=jac_rxn.dtype)
    jac_species = torch.einsum("sr,brn->bsn", stoich, jac_rxn)
    return jac_species.reshape(ncol, nz, nspecies, nspecies)


def _normalize_state_tensors(
    temperature: torch.Tensor,
    pressure: torch.Tensor,
    concentration: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    temp = torch.as_tensor(temperature)
    pres = torch.as_tensor(pressure, dtype=temp.dtype, device=temp.device)
    conc = torch.as_tensor(concentration, dtype=temp.dtype, device=temp.device)

    if temp.dim() == 1:
        temp = temp.unsqueeze(0)
    if pres.dim() == 1:
        pres = pres.unsqueeze(0)
    if conc.dim() == 2:
        conc = conc.unsqueeze(0)
    if temp.dim() != 2 or pres.dim() != 2 or conc.dim() != 3:
        raise ValueError("temperature/pressure must be 2D and concentration must be 3D")
    if temp.shape != pres.shape:
        raise ValueError("temperature and pressure must share shape (ncol, nz)")
    if conc.shape[:2] != temp.shape:
        raise ValueError("concentration must have shape (ncol, nz, nspecies)")
    return temp, pres, conc


def _normalize_actinic_flux(
    actinic_flux: torch.Tensor,
    ncol: int,
    nz: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    flux = torch.as_tensor(actinic_flux, dtype=dtype, device=device)
    if flux.dim() == 2:
        flux = flux.unsqueeze(1)
    if flux.dim() != 3:
        raise ValueError("actinic_flux must have shape (nwave, ncol, nz)")
    if flux.size(1) != ncol or flux.size(2) != nz:
        raise ValueError("actinic_flux must have shape (nwave, ncol, nz)")
    return flux


def _flatten_photo_inputs(
    photo_inputs: dict[str, torch.Tensor],
    ncol: int,
    nz: int,
) -> dict[str, torch.Tensor]:
    flattened: dict[str, torch.Tensor] = {}
    for key, value in photo_inputs.items():
        tensor = torch.as_tensor(value)
        if key == "actinic_flux":
            tensor = _normalize_actinic_flux(tensor, ncol, nz, dtype=tensor.dtype, device=tensor.device)
            flattened[key] = tensor.reshape(tensor.size(0), -1)
        elif key == "wavelength":
            flattened[key] = tensor
        else:
            flattened[key] = tensor
    return flattened
