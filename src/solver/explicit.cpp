#include "explicit.h"

#include <torch/torch.h>

#include <iostream>

namespace kintera {

// TODO: Use a solver base class

void explicit_solve(torch::Tensor& C, const torch::Tensor& rates,
                    const torch::Tensor& stoich_matrix, double dt,
                    double max_rel_change) {
  auto stoich_t = stoich_matrix.t();

  // Expand stoich_t to match the batch dimensions of rates
  std::vector<int64_t> expand_shape;
  for (int64_t i = 0; i < rates.dim() - 1; ++i) {
    expand_shape.push_back(rates.size(i));
  }
  expand_shape.push_back(stoich_t.size(0));
  expand_shape.push_back(stoich_t.size(1));

  auto stoich_expanded = stoich_t.expand(expand_shape);
  auto rates_reshaped = rates.unsqueeze(-1);
  auto dC = torch::matmul(stoich_expanded, rates_reshaped).squeeze(-1);
  auto abs_rel_changes = (dC * dt).abs() / (C + 1e-30);
  auto max_change = abs_rel_changes.max().item<double>();

  // Limit timestep if necessary
  double actual_dt = dt;
  if (max_change > max_rel_change) {
    std::cout << "Limiting timestep from " << dt << " to "
              << dt * (max_rel_change / max_change) << std::endl;
    actual_dt = dt * (max_rel_change / max_change);
  }

  C += dC * actual_dt;

  // Ensure non-negativity; should be ensured, prevent numerical errors
  C.clamp_(0);
}

}  // namespace kintera
