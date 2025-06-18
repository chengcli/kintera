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
#include <kintera/reaction.hpp>

// arg
#include <kintera/add_arg.h>

namespace YAML {
class Node;
}

namespace kintera {

//! Options to initialize all reaction rate constants
struct ArrheniusOptions {
  static ArrheniusOptions from_yaml(const YAML::Node& node);
  virtual ~ArrheniusOptions() = default;

  //! reactions
  ADD_ARG(std::vector<Reaction>, reactions) = {};

  //! Pre-exponential factor. The unit system is (mol, cm, s);
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
  //! log pre-exponential factor [ln(mol, m, s)], shape (nreaction,)
  torch::Tensor logA;

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

  //! Compute the log rate constant
  /*!
   * \param T temperature [K], shape (...)
   * \param P pressure [pa], shape (...)
   * \param other additional parameters
   * \return log reaction rate constant in ln(mol, m, s), (..., nreaction)
   */
  torch::Tensor forward(torch::Tensor T, torch::Tensor P,
                        std::map<std::string, torch::Tensor> const& other);
};
TORCH_MODULE(Arrhenius);

}  // namespace kintera

#undef ADD_ARG
