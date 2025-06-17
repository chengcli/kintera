#pragma once

// torch
#include <torch/torch.h>

// kintera
#include <kintera/reaction.hpp>
#include <kintera/utils/func1.hpp>

// arg
#include <kintera/add_arg.h>

namespace YAML {
class Node;
}

namespace kintera {

struct Nucleation {
  Nucleation() = default;
  static Nucleation from_yaml(const YAML::Node& node);

  ADD_ARG(double, minT) = 0.0;
  ADD_ARG(double, maxT) = 3000.;
  ADD_ARG(Reaction, reaction);
  ADD_ARG(user_func1, func);
  ADD_ARG(user_func1, func_ddT);
};

}  // namespace kintera

#undef ADD_ARG
