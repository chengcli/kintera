#pragma once

// C/C++
#include <set>

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

struct NucleationOptions {
  static NucleationOptions from_yaml(const YAML::Node& node);
  NucleationOptions() = default;

  ADD_ARG(std::vector<Reaction>, reactions) = {};
  ADD_ARG(std::vector<double>, minT) = {};
  ADD_ARG(std::vector<double>, maxT) = {};
  ADD_ARG(std::vector<user_func1>, logsvp) = {};
  ADD_ARG(std::vector<user_func1>, logsvp_ddT) = {};
};

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, NucleationOptions op);

}  // namespace kintera

#undef ADD_ARG
