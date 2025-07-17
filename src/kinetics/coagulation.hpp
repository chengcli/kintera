#pragma once

// C/C++
#include <set>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include "arrhenius.hpp"

namespace kintera {

struct CoagulationOptions : public ArrheniusOptions {
  CoagulationOptions() = default;
  CoagulationOptions(const ArrheniusOptions& arrhenius)
      : ArrheniusOptions(arrhenius) {}
  void report(std::ostream& os) const { ArrheniusOptions::report(os); }
};

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        CoagulationOptions op);

}  // namespace kintera
