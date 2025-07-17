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

//! Options to initialize all reaction rate constants
struct ArrheniusOptions {
  static ArrheniusOptions from_yaml(const YAML::Node& node,
                                    std::string const& other_type = "");
  virtual ~ArrheniusOptions() = default;
  void report(std::ostream& os) const {
    os << "* reactions = " << fmt::format("{}", reactions()) << "\n"
       << "* Tref = " << Tref() << " K\n"
       << "* units = " << units() << "\n"
       << "* A = " << fmt::format("{}", A()) << "\n"
       << "* b = " << fmt::format("{}", b()) << "\n"
       << "* Ea_R = " << fmt::format("{}", Ea_R()) << " K\n"
       << "* E4_R = " << fmt::format("{}", E4_R()) << "\n";
  }

  // reference temperature
  ADD_ARG(double, Tref) = 300.0;

  // units
  ADD_ARG(std::string, units) = "molecule,cm,s";

  //! reactions
  ADD_ARG(std::vector<Reaction>, reactions) = {};

  //! Pre-exponential factor.
  //! actual units depend on the reaction order
  ADD_ARG(std::vector<double>, A) = {};

  //! Dimensionless temperature exponent
  ADD_ARG(std::vector<double>, b) = {};

  //! Activation energy in K
  ADD_ARG(std::vector<double>, Ea_R) = {};

  //! Additional 4th parameter in the rate expression
  ADD_ARG(std::vector<double>, E4_R) = {};
};

void add_to_vapor_cloud(std::set<std::string>& vapor_set,
                        std::set<std::string>& cloud_set, ArrheniusOptions op);

class ArrheniusImpl : public torch::nn::Cloneable<ArrheniusImpl> {
 public:
  //! pre-exponential factor [molecule, cm, s], shape (nreaction,)
  torch::Tensor A;

  //! temperature exponent, shape (nreaction,)
  torch::Tensor b;

  //! activation energy [K], shape (nreaction,)
  torch::Tensor Ea_R;

  //! additional 4th parameter in the rate expression, shape (nreaction,)
  torch::Tensor E4_R;

  //! options with which this `ArrheniusImpl` was constructed
  ArrheniusOptions options;

  //! Constructor to initialize the layer
  ArrheniusImpl() = default;
  explicit ArrheniusImpl(ArrheniusOptions const& options_);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! Compute the rate constant
  /*!
   * \param T temperature [K], shape (...)
   * \param P pressure [pa], shape (...)
   * \param C concentration [mol/m^3], shape (..., nspecies)
   * \param other additional parameters
   * \return reaction rate constant in (mol, m, s), (..., nreaction)
   */
  torch::Tensor forward(torch::Tensor T, torch::Tensor P, torch::Tensor C,
                        std::map<std::string, torch::Tensor> const& other);
};
TORCH_MODULE(Arrhenius);

}  // namespace kintera

#undef ADD_ARG
