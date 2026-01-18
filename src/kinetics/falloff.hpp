#pragma once

// C/C++
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/kintera_formatter.hpp>
#include <kintera/reaction.hpp>

// arg
#include <kintera/add_arg.h>

namespace YAML {
class Node;
}

namespace kintera {

enum class FalloffType { None = 0, Troe = 1, SRI = 2 };

inline std::string falloff_type_to_string(FalloffType type) {
  switch (type) {
    case FalloffType::Troe:
      return "Troe";
    case FalloffType::SRI:
      return "SRI";
    default:
      return "none";
  }
}

inline FalloffType string_to_falloff_type(const std::string& str) {
  if (str == "Troe") return FalloffType::Troe;
  if (str == "SRI") return FalloffType::SRI;
  return FalloffType::None;
}

// Falloff/three-body reaction options
// Three-body: k = k0 * [M]
// Falloff: k = k0*[M] / (1 + k0*[M]/k_inf) * F
struct FalloffOptionsImpl {
  static std::shared_ptr<FalloffOptionsImpl> create() {
    return std::make_shared<FalloffOptionsImpl>();
  }

  static std::shared_ptr<FalloffOptionsImpl> from_yaml(
      const YAML::Node& node,
      std::shared_ptr<FalloffOptionsImpl> derived_type_ptr = nullptr);

  virtual std::string name() const { return "falloff"; }
  virtual ~FalloffOptionsImpl() = default;

  void report(std::ostream& os) const {
    os << "* reactions = " << fmt::format("{}", reactions()) << "\n"
       << "* Tref = " << Tref() << " K\n"
       << "* nreactions = " << reactions().size() << "\n";
  }

  ADD_ARG(double, Tref) = 300.0;
  ADD_ARG(std::string, units) = "molecule,cm,s";
  ADD_ARG(std::vector<Reaction>, reactions) = {};

  // Low-pressure Arrhenius: k0 = A * (T/Tref)^b * exp(-Ea_R/T)
  ADD_ARG(std::vector<double>, k0_A) = {};
  ADD_ARG(std::vector<double>, k0_b) = {};
  ADD_ARG(std::vector<double>, k0_Ea_R) = {};

  // High-pressure Arrhenius (1e100 for three-body)
  ADD_ARG(std::vector<double>, kinf_A) = {};
  ADD_ARG(std::vector<double>, kinf_b) = {};
  ADD_ARG(std::vector<double>, kinf_Ea_R) = {};

  ADD_ARG(std::vector<int>, falloff_types) = {};
  ADD_ARG(std::vector<bool>, is_three_body) = {};

  // Troe: F_cent = (1-A)*exp(-T/T3) + A*exp(-T/T1) + exp(-T2/T)
  ADD_ARG(std::vector<double>, troe_A) = {};
  ADD_ARG(std::vector<double>, troe_T3) = {};
  ADD_ARG(std::vector<double>, troe_T1) = {};
  ADD_ARG(std::vector<double>, troe_T2) = {};

  // SRI: F_cent = D * (A*exp(-B/T) + exp(-T/C))^E
  ADD_ARG(std::vector<double>, sri_A) = {};
  ADD_ARG(std::vector<double>, sri_B) = {};
  ADD_ARG(std::vector<double>, sri_C) = {};
  ADD_ARG(std::vector<double>, sri_D) = {};
  ADD_ARG(std::vector<double>, sri_E) = {};

  // Per-reaction third-body efficiencies
  ADD_ARG(std::vector<Composition>, efficiencies) = {};
};
using FalloffOptions = std::shared_ptr<FalloffOptionsImpl>;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, FalloffOptions op);

}  // namespace kintera

#undef ADD_ARG
