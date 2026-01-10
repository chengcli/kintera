//! @file jacobian_photolysis.cpp
//! @brief Jacobian computation for photolysis reactions

#include "jacobian.hpp"

namespace kintera {

torch::Tensor jacobian_photolysis(torch::Tensor rate, torch::Tensor stoich,
                                  torch::Tensor conc, torch::Tensor rc_ddC,
                                  torch::optional<torch::Tensor> rc_ddT) {
  // For first-order photolysis: A + hν -> products
  // Rate law: d[A]/dt = -k[A], where k is the photolysis rate constant
  // Jacobian: J[i,j] = d(d[i]/dt)/d[j] = stoich[i,r] * k[r] * order[j,r] / [j]

  auto batch_shape = rate.sizes().vec();
  batch_shape.pop_back();
  int nreaction = rate.size(-1);
  int nspecies = stoich.size(0);

  // Reactant order: negative stoichiometry clamped to positive
  auto order = (-stoich).clamp_min(0.0);

  auto output_shape = batch_shape;
  output_shape.push_back(nspecies);
  output_shape.push_back(nspecies);
  auto jacobian = torch::zeros(output_shape, rate.options());

  if (batch_shape.empty()) {
    // Non-batched case
    auto stoich_rate = stoich * rate;
    jacobian = torch::matmul(stoich_rate, order.t());
    jacobian = jacobian / conc.clamp_min(1e-30).unsqueeze(-2);
  } else {
    // Batched case
    std::vector<int64_t> stoich_shape = batch_shape;
    stoich_shape.push_back(nspecies);
    stoich_shape.push_back(nreaction);

    auto stoich_rate = stoich.expand(stoich_shape) * rate.unsqueeze(-2);
    jacobian = torch::matmul(stoich_rate, order.t());
    jacobian = jacobian / conc.clamp_min(1e-30).unsqueeze(-2);
  }

  // Add concentration-dependent rate contributions if present
  if (rc_ddC.defined() && rc_ddC.numel() > 0) {
    jacobian = jacobian + torch::matmul(stoich, rc_ddC.transpose(-1, -2));
  }

  return jacobian;
}

}  // namespace kintera
