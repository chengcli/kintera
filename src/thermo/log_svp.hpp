#pragma once

// torch
#include <torch/torch.h>

// kintera
#include "nucleation.hpp"

namespace kintera {

class LogSVPFunc : public torch::autograd::Function<LogSVPFunc> {
 public:
  static constexpr bool is_traceable = true;

  static void init(NucleationOptions const& op) {
    _logsvp = op.logsvp();
    _logsvp_ddT = op.logsvp_ddT();
  }

  static torch::Tensor grad(torch::Tensor const& temp, bool expanded = false);

  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor const& temp,
                               bool expanded = false);

  static std::vector<torch::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      std::vector<torch::Tensor> grad_outputs);

 private:
  static std::vector<user_func1> _logsvp;
  static std::vector<user_func1> _logsvp_ddT;
};

}  // namespace kintera
