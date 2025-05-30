#pragma once

// yaml
#include <yaml-cpp/yaml.h>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/reaction.hpp>

#include "thermo.h"

// arg
#include <kintera/add_arg.h>

namespace kintera {

struct Nucleation {
  Nucleation() = default;
  static Nucleation from_yaml(const YAML::Node& node);

  torch::Tensor eval_func(torch::Tensor tem) const;
  torch::Tensor eval_func_ddT(torch::Tensor tem) const;

  ADD_ARG(double, minT) = 0.0;
  ADD_ARG(double, maxT) = 3000.;
  ADD_ARG(Reaction, reaction);
  ADD_ARG(user_func1, func);
  ADD_ARG(user_func1, func_ddT);
};

}  // namespace kintera

#undef ADD_ARG
