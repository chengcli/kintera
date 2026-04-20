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
#include <kintera/kintera_formatter.hpp>
#include <kintera/reaction.hpp>

// arg
#include <kintera/add_arg.h>

namespace YAML {
class Node;
}

namespace kintera {

//! Options for three-body (Lindemann) reaction rate constants
/*!
 * Standard Lindemann falloff formula:
 *   k_eff(T, P) = k0(T) * [M] / (1 + k0(T) * [M] / kinf(T))
 *
 * where:
 *   k0(T) = A0 * T^b0 * exp(-Ea_R0 / T)     (low-pressure limit)
 *   kinf(T) = Ainf * T^binf * exp(-Ea_Rinf / T)  (high-pressure limit)
 *   [M] = P / (R * T)  [mol/m^3]
 *
 * Input parameters are in CGS (molecule, cm, s) and converted to SI
 * internally. k0 is converted as termolecular (order = sum_stoich + 1),
 * kinf as bimolecular (order = sum_stoich).
 */
struct ThreeBodyOptionsImpl {
  static std::shared_ptr<ThreeBodyOptionsImpl> create() {
    return std::make_shared<ThreeBodyOptionsImpl>();
  }
  static std::shared_ptr<ThreeBodyOptionsImpl> from_yaml(
      const YAML::Node& node);

  std::string name() const { return "three-body"; }

  std::shared_ptr<ThreeBodyOptionsImpl> clone() const {
    return std::make_shared<ThreeBodyOptionsImpl>(*this);
  }

  void report(std::ostream& os) const {
    os << "* reactions = " << fmt::format("{}", reactions()) << "\n"
       << "* A0 = " << fmt::format("{}", A0()) << "\n"
       << "* b0 = " << fmt::format("{}", b0()) << "\n"
       << "* Ea_R0 = " << fmt::format("{}", Ea_R0()) << " K\n"
       << "* Ainf = " << fmt::format("{}", Ainf()) << "\n"
       << "* binf = " << fmt::format("{}", binf()) << "\n"
       << "* Ea_Rinf = " << fmt::format("{}", Ea_Rinf()) << " K\n";
  }

  ADD_ARG(std::vector<Reaction>, reactions) = {};

  //! Low-pressure (termolecular) Arrhenius parameters
  ADD_ARG(std::vector<double>, A0) = {};
  ADD_ARG(std::vector<double>, b0) = {};
  ADD_ARG(std::vector<double>, Ea_R0) = {};

  //! High-pressure (bimolecular) Arrhenius parameters
  ADD_ARG(std::vector<double>, Ainf) = {};
  ADD_ARG(std::vector<double>, binf) = {};
  ADD_ARG(std::vector<double>, Ea_Rinf) = {};
};
using ThreeBodyOptions = std::shared_ptr<ThreeBodyOptionsImpl>;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, ThreeBodyOptions op);

class ThreeBodyImpl : public torch::nn::Cloneable<ThreeBodyImpl> {
 public:
  //! Low-pressure Arrhenius parameters (SI)
  torch::Tensor A0;
  torch::Tensor b0;
  torch::Tensor Ea_R0;

  //! High-pressure Arrhenius parameters (SI)
  torch::Tensor Ainf;
  torch::Tensor binf;
  torch::Tensor Ea_Rinf;

  ThreeBodyOptions options;

  ThreeBodyImpl() : options(ThreeBodyOptionsImpl::create()) {}
  explicit ThreeBodyImpl(ThreeBodyOptions const& options_);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! Compute effective bimolecular rate constant via Lindemann falloff
  /*!
   * \param T temperature [K], shape (...)
   * \param P pressure [Pa], shape (...)
   * \param C concentration [mol/m^3], shape (..., nspecies)
   * \param other additional parameters
   * \return effective rate constant [mol, m, s], shape (..., nreaction)
   */
  torch::Tensor forward(torch::Tensor T, torch::Tensor P, torch::Tensor C,
                        std::map<std::string, torch::Tensor> const& other);
};
TORCH_MODULE(ThreeBody);

}  // namespace kintera
