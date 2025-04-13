#pragma once

// torch
#include <torch/torch.h>

// kintera
#include <kintera/add_arg.h>

#include <kintera/reaction.hpp>

namespace kintera {

using func1_t = std::function<torch::Tensor(torch::Tensor)>;

struct Nucleation {
  Nucleation() = default;
  Nucleation(std::string const& equation, std::string const& name,
             std::map<std::string, double> const& params = {});

  torch::Tensor eval_func(torch::Tensor tem) const;
  torch::Tensor eval_logf_ddT(torch::Tensor tem) const;

  ADD_ARG(double, min_tem) = 0.0;
  ADD_ARG(double, max_tem) = 3000.;
  ADD_ARG(Reaction, reaction);
  ADD_ARG(func1_t, func);
  ADD_ARG(func1_t, logf_ddT);
};

}  // namespace kintera
