#pragma once

// torch
#include <torch/torch.h>

namespace kintera {

//! Single-step backward Euler implicit solve
/*!
 * Solves: Δc = (I/dt - S·J)^{-1} · S·r
 *
 * \param rate        reaction rates [mol/m³/s], shape (..., nreaction)
 * \param stoich      stoichiometry matrix, shape (nspecies, nreaction)
 * \param jacobian    Jacobian matrix, shape (..., nreaction, nspecies)
 * \param dt          time step [s]
 * \return concentration change [mol/m³], shape (..., nspecies)
 */
inline torch::Tensor evolve_implicit(torch::Tensor rate, torch::Tensor stoich,
                                     torch::Tensor jacobian, double dt) {
  auto nspecies = stoich.size(0);
  auto eye = torch::eye(nspecies, rate.options());
  auto SJ = stoich.matmul(jacobian);
  auto SR = stoich.matmul(rate.unsqueeze(-1)).squeeze(-1);
  return torch::linalg_solve(eye / dt - SJ, SR);
}

//! Multi-step subcycled backward Euler implicit solve
/*!
 * Divides the total timestep into nsubsteps smaller steps.
 * Each substep reuses the same rate and Jacobian (frozen coefficients).
 *
 * \param rate        reaction rates [mol/m³/s], shape (..., nreaction)
 * \param stoich      stoichiometry matrix, shape (nspecies, nreaction)
 * \param jacobian    Jacobian matrix, shape (..., nreaction, nspecies)
 * \param dt          total time step [s]
 * \param nsubsteps   number of substeps (default 1)
 * \return total concentration change [mol/m³], shape (..., nspecies)
 */
inline torch::Tensor evolve_implicit_subcycle(torch::Tensor rate,
                                              torch::Tensor stoich,
                                              torch::Tensor jacobian, double dt,
                                              int nsubsteps = 1) {
  if (nsubsteps <= 1) {
    return evolve_implicit(rate, stoich, jacobian, dt);
  }

  auto nspecies = stoich.size(0);
  auto eye = torch::eye(nspecies, rate.options());
  auto sub_dt = dt / nsubsteps;

  // Precompute S*J and factorize for reuse (same Jacobian across substeps)
  auto SJ = stoich.matmul(jacobian);
  auto SR = stoich.matmul(rate.unsqueeze(-1)).squeeze(-1);
  auto A = eye / sub_dt - SJ;

  // Accumulate total concentration change
  auto total_dc = torch::zeros_like(SR);

  for (int s = 0; s < nsubsteps; ++s) {
    auto dc = torch::linalg_solve(A, SR);
    total_dc += dc;
  }

  return total_dc;
}

}  // namespace kintera
