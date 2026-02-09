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

//! Options for Troe falloff reactions: k = k_Lindemann * F_Troe
struct TroeFalloffOptionsImpl {
  static std::shared_ptr<TroeFalloffOptionsImpl> create() {
    return std::make_shared<TroeFalloffOptionsImpl>();
  }

  static std::shared_ptr<TroeFalloffOptionsImpl> from_yaml(
      const YAML::Node& node,
      std::shared_ptr<TroeFalloffOptionsImpl> derived_type_ptr = nullptr);

  virtual std::string name() const { return "falloff"; }
  virtual ~TroeFalloffOptionsImpl() = default;

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

  // High-pressure Arrhenius: kinf = A * (T/Tref)^b * exp(-Ea_R/T)
  ADD_ARG(std::vector<double>, kinf_A) = {};
  ADD_ARG(std::vector<double>, kinf_b) = {};
  ADD_ARG(std::vector<double>, kinf_Ea_R) = {};

  // Troe parameters: F_cent = (1-A)*exp(-T/T3) + A*exp(-T/T1) + exp(-T2/T)
  ADD_ARG(std::vector<double>, troe_A) = {};
  ADD_ARG(std::vector<double>, troe_T3) = {};
  ADD_ARG(std::vector<double>, troe_T1) = {};
  ADD_ARG(std::vector<double>, troe_T2) = {};  // 0.0 for 3-param, non-zero for 4-param

  // Per-reaction third-body efficiencies
  ADD_ARG(std::vector<Composition>, efficiencies) = {};
};
using TroeFalloffOptions = std::shared_ptr<TroeFalloffOptionsImpl>;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, TroeFalloffOptions op);

class TroeFalloffImpl : public torch::nn::Cloneable<TroeFalloffImpl> {
 public:
  //! Low-pressure Arrhenius parameters, shape (nreaction,)
  torch::Tensor k0_A;
  torch::Tensor k0_b;
  torch::Tensor k0_Ea_R;

  //! High-pressure Arrhenius parameters, shape (nreaction,)
  torch::Tensor kinf_A;
  torch::Tensor kinf_b;
  torch::Tensor kinf_Ea_R;

  //! Efficiency matrix: efficiency[i][j] = efficiency of species j for reaction i
  //! Shape: (nreaction, nspecies)
  //! Default efficiency = 1.0 if species not in efficiency map
  torch::Tensor efficiency_matrix;

  //! Troe parameters, shape (nreaction,)
  torch::Tensor troe_A;
  torch::Tensor troe_T3;
  torch::Tensor troe_T1;
  torch::Tensor troe_T2;

  //! options with which this `TroeFalloffImpl` was constructed
  TroeFalloffOptions options;

  //! Constructor to initialize the layer
  TroeFalloffImpl() : options(TroeFalloffOptionsImpl::create()) {}
  explicit TroeFalloffImpl(TroeFalloffOptions const& options_);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! Compute the rate constant
  /*!
   * \param T temperature [K], shape (...)
   * \param P pressure [Pa], shape (...)
   * \param C concentration [mol/m^3], shape (..., nspecies)
   * \param other additional parameters
   * \return reaction rate constant in (mol, m, s), shape (..., nreaction)
   */
  torch::Tensor forward(torch::Tensor T, torch::Tensor P, torch::Tensor C,
                        std::map<std::string, torch::Tensor> const& other);

 private:
  torch::Tensor compute_k0(torch::Tensor T) const;
  torch::Tensor compute_kinf(torch::Tensor T) const;
  torch::Tensor compute_falloff_factor(torch::Tensor T, torch::Tensor Pr) const;
};
TORCH_MODULE(TroeFalloff);

}  // namespace kintera

#undef ADD_ARG
