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

struct CoagulationOptionsImpl final : public ArrheniusOptionsImpl {
  static std::shared_ptr<CoagulationOptionsImpl> create() {
    return std::make_shared<CoagulationOptionsImpl>();
  }
  static std::shared_ptr<CoagulationOptionsImpl> from_yaml(
      const YAML::Node& node);

  std::string name() const override { return "coagulation"; }
  CoagulationOptionsImpl() = default;
  CoagulationOptionsImpl(const ArrheniusOptionsImpl& arrhenius)
      : ArrheniusOptionsImpl(arrhenius) {}

  void report(std::ostream& os) const { ArrheniusOptionsImpl::report(os); }
};
using CoagulationOptions = std::shared_ptr<CoagulationOptionsImpl>;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set,
                        CoagulationOptions op);

}  // namespace kintera
