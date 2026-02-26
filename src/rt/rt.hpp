#pragma once

#include <torch/torch.h>
#include <vector>

namespace kintera {

//! \param y               number density (nz, nspecies)  [cm^-3]
//! \param cross_sections  absorption cross-sections per absorber
//!                        (n_absorbers, nwave) [cm^2]
//! \param dz              layer thickness (nz,) [cm]
//! \param absorber_ids    indices into the species dimension of y
//! \return optical depth at each level (nz+1, nwave), where
//!         level 0 = TOA (tau=0) and level nz = surface
torch::Tensor compute_optical_depth(
    torch::Tensor y, torch::Tensor cross_sections,
    torch::Tensor dz, std::vector<int> absorber_ids);

//! \param stellar_flux  TOA stellar flux (nwave,) [photons cm^-2 s^-1 nm^-1]
//! \param tau           optical depth at levels (nz+1, nwave)
//! \param cos_zenith    cosine of the solar zenith angle
//! \return actinic flux at layer centers (nz, nwave)
torch::Tensor compute_actinic_flux(
    torch::Tensor stellar_flux, torch::Tensor tau, double cos_zenith);

//! \param y               number density (nz, nspecies)
//! \param cross_sections  (n_absorbers, nwave)
//! \param stellar_flux    (nwave,)
//! \param dz              (nz,)
//! \param cos_zenith      cosine of solar zenith angle
//! \param absorber_ids    species indices of absorbers
//! \return actinic flux at layer centers (nz, nwave)
torch::Tensor compute_actinic_flux(
    torch::Tensor y, torch::Tensor cross_sections,
    torch::Tensor stellar_flux, torch::Tensor dz,
    double cos_zenith, std::vector<int> absorber_ids);

}  // namespace kintera
