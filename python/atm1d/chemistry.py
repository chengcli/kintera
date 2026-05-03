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
    if photo_inputs is None:
        rate, rc_ddc, rc_ddt = kinetics.forward(temperature, pressure, concentration)
    else:
        rate, rc_ddc, rc_ddt = kinetics.forward(
            temperature, pressure, concentration, photo_inputs
        )
    cvol = torch.ones_like(temperature)
    jac_rxn = kinetics.jacobian(temperature, concentration, cvol, rate, rc_ddc, rc_ddt)
    stoich = kinetics.stoich.to(device=jac_rxn.device, dtype=jac_rxn.dtype)
    return torch.einsum("sr,brn->bsn", stoich, jac_rxn)


def build_photochemistry_jacobian(
    photo_chem,
    temperature: torch.Tensor,
    concentration: torch.Tensor,
    actinic_flux: torch.Tensor,
) -> torch.Tensor:
    rate = photo_chem.forward(temperature, concentration, actinic_flux)
    jac_rxn = photo_chem.jacobian(concentration, rate)
    stoich = photo_chem.stoich.to(device=jac_rxn.device, dtype=jac_rxn.dtype)
    return torch.einsum("sr,brn->bsn", stoich, jac_rxn)
