#pragma once

// torch
#include <torch/torch.h>

// kintera
#include "nucleation.hpp"

namespace kintera {

class LogSVPFunc : public torch::autograd::Function<LogSVPFunc> {
 public:
  static constexpr bool is_traceable = true;

  static void init(std::vector<Nucleation> const& react) { _react = react; }

  static torch::Tensor grad(torch::Tensor const& temp);

  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor const& temp);

  static std::vector<torch::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      std::vector<torch::Tensor> grad_outputs);

 private:
  static std::vector<Nucleation> _react;
};

}  // namespace kintera
