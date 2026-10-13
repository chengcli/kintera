#pragma once

#include <torch/torch.h>

namespace kintera {

//! Surface extent for signed gas stoichiometry, with implicit first
//! derivatives. conc: (..., nspecies, nreaction), nu: (nspecies, nreaction),
//! logkc: (..., nreaction). Inputs must already be broadcast to these shapes.
class EvaporationExtent : public torch::autograd::Function<EvaporationExtent> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor conc, torch::Tensor nu,
                               torch::Tensor logkc);
  static std::vector<torch::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      std::vector<torch::Tensor> grad_outputs);
};

}  // namespace kintera
