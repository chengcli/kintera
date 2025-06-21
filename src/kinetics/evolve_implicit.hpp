#pragma once

// torch
#include <torch/torch.h>

namespace kintera {

inline torch::Tensor evolve_implicit(torch::Tensor rate, torch::Tensor stoich,
                                     torch::Tensor jacobian, double dt) {
  auto nspecies = stoich.size(0);
  auto eye = torch::eye(nspecies, rate.options());
  auto SJ = stoich.matmul(jacobian);
  auto SR = stoich.matmul(rate.unsqueeze(-1)).squeeze(-1);
  return torch::linalg_solve(eye / dt - SJ, SR);
}

}  // namespace kintera
