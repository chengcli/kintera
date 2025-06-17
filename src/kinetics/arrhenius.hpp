#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// arg
#include <kintera/add_arg.h>

namespace YAML {
class Node;
}

namespace kintera {

//! Options to initialize all reaction rate constants
struct ArrheniusOptions {
  static ArrheniusOptions from_yaml(const YAML::Node& node);
  static ArrheniusOptions from_map(
      const std::vector<std::map<std::string, std::string>>& param);

  //! Pre-exponential factor. The unit system is (mol, m, s);
  //! actual units depend on the reaction order
  ADD_ARG(std::vector<double>, A) = {};

  //! Dimensionless temperature exponent
  ADD_ARG(std::vector<double>, b) = {};

  //! Activation energy in K
  ADD_ARG(std::vector<double>, Ea_R) = {};

  //! Additional 4th parameter in the rate expression
  ADD_ARG(std::vector<double>, E4_R) = {};
};

class ArrheniusImpl : public torch::nn::Cloneable<ArrheniusImpl> {
 public:
  //! log pre-exponential factor, shape (nreaction,)
  torch::Tensor logA;

  //! temperature exponent, shape (nreaction,)
  torch::Tensor b;

  //! activation energy in K, shape (nreaction,)
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
