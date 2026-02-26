//! @file diffusion.cpp
//! @brief Eddy diffusion (Kzz) transport module
//!
//! Implements the same conservative finite-volume discretization as
//! VULCAN's diffdf_no_mol: mixing-ratio diffusion on a non-uniform grid
//! with zero-flux boundary conditions.

#include "diffusion.hpp"

namespace kintera {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
diffusion_coefficients(torch::Tensor y, torch::Tensor Kzz,
                       torch::Tensor dzi) {
  int nz = y.size(0);
  auto opts = y.options();
  auto n_tot = y.sum(-1);  // (nz,)

  auto n_avg = (n_tot.slice(0, 1, nz) + n_tot.slice(0, 0, nz - 1)) / 2.0;

  auto D = Kzz * n_avg / dzi;

  auto A = torch::zeros({nz}, opts);
  auto B = torch::zeros({nz}, opts);
  auto C = torch::zeros({nz}, opts);

  A[0] = -D[0] / dzi[0] / n_tot[0];
  B[0] =  D[0] / dzi[0] / n_tot[1];

  A[nz - 1] = -D[nz - 2] / dzi[nz - 2] / n_tot[nz - 1];
  C[nz - 1] =  D[nz - 2] / dzi[nz - 2] / n_tot[nz - 2];

  if (nz > 2) {
    auto dzi_lo = dzi.slice(0, 0, nz - 2);  // dzi[j-1], j=1..nz-2
    auto dzi_hi = dzi.slice(0, 1, nz - 1);  // dzi[j],   j=1..nz-2
    auto dz_avg = (dzi_lo + dzi_hi) / 2.0;

    auto D_lo = D.slice(0, 0, nz - 2);  // D[j-1]
    auto D_hi = D.slice(0, 1, nz - 1);  // D[j]

    auto n_j   = n_tot.slice(0, 1, nz - 1);
    auto n_jp1 = n_tot.slice(0, 2, nz);
    auto n_jm1 = n_tot.slice(0, 0, nz - 2);

    A.slice(0, 1, nz - 1) = -(D_hi + D_lo) / (dz_avg * n_j);
    B.slice(0, 1, nz - 1) = D_hi / (dz_avg * n_jp1);

    C.slice(0, 1, nz - 1) = D_lo / (dz_avg * n_jm1);
  }

  return {A, B, C};
}

torch::Tensor diffusion_tendency(torch::Tensor y, torch::Tensor Kzz,
                                 torch::Tensor dzi) {
  int nz = y.size(0);
  auto [A, B, C] = diffusion_coefficients(y, Kzz, dzi);

  auto Au = A.unsqueeze(-1);
  auto Bu = B.unsqueeze(-1);
  auto Cu = C.unsqueeze(-1);

  auto tendency = Au * y;

  tendency.slice(0, 0, nz - 1) += Bu.slice(0, 0, nz - 1) * y.slice(0, 1, nz);
  tendency.slice(0, 1, nz) += Cu.slice(0, 1, nz) * y.slice(0, 0, nz - 1);

  return tendency;
}

}  // namespace kintera
