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

//! Options for three-body reactions: k = k0 * [M]_eff
struct ThreeBodyOptionsImpl {
  static std::shared_ptr<ThreeBodyOptionsImpl> create() {
    return std::make_shared<ThreeBodyOptionsImpl>();
  }

  static std::shared_ptr<ThreeBodyOptionsImpl> from_yaml(
      const YAML::Node& node,
      std::shared_ptr<ThreeBodyOptionsImpl> derived_type_ptr = nullptr);

  virtual std::string name() const { return "three-body"; }
  virtual ~ThreeBodyOptionsImpl() = default;

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

  // Per-reaction third-body efficiencies
  ADD_ARG(std::vector<Composition>, efficiencies) = {};
};
using ThreeBodyOptions = std::shared_ptr<ThreeBodyOptionsImpl>;

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, ThreeBodyOptions op);

class ThreeBodyImpl : public torch::nn::Cloneable<ThreeBodyImpl> {
 public:
  //! Low-pressure Arrhenius parameters, shape (nreaction,)
  torch::Tensor k0_A;
  torch::Tensor k0_b;
  torch::Tensor k0_Ea_R;

  //! Efficiency matrix: efficiency[i][j] = efficiency of species j for reaction i
  //! Shape: (nreaction, nspecies)
  //! Default efficiency = 1.0 if species not in efficiency map
  torch::Tensor efficiency_matrix;

  //! options with which this `ThreeBodyImpl` was constructed
  ThreeBodyOptions options;

  //! Constructor to initialize the layer
  ThreeBodyImpl() : options(ThreeBodyOptionsImpl::create()) {}
  explicit ThreeBodyImpl(ThreeBodyOptions const& options_);
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
};
TORCH_MODULE(ThreeBody);

}  // namespace kintera

#undef ADD_ARG
