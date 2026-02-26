#pragma once

#include <torch/torch.h>

namespace kintera {

//! \param y      number density (nz, nspecies)
//! \param Kzz    eddy diffusion coefficient at interfaces (nz-1,)  [cm^2/s]
//! \param dzi    interface grid spacing (nz-1,) [cm]
//! \return       diffusion tendency dy/dt (nz, nspecies)
torch::Tensor diffusion_tendency(torch::Tensor y, torch::Tensor Kzz,
                                 torch::Tensor dzi);

//! Compute the tridiagonal coefficients A, B, C of the diffusion operator.
//!
//! The tendency is:  dy_j/dt = A_j * y_j + B_j * y_{j+1} + C_j * y_{j-1}
//! \param y      number density (nz, nspecies)
//! \param Kzz    eddy diffusion coefficient at interfaces (nz-1,)
//! \param dzi    interface grid spacing (nz-1,)
//! \return       (A, B, C) each of shape (nz,)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
diffusion_coefficients(torch::Tensor y, torch::Tensor Kzz, torch::Tensor dzi);

}  // namespace kintera
