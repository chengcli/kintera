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
struct EvaporationOptions {
  static EvaporationOptions from_yaml(const YAML::Node& node);

  // reference temperature
  ADD_ARG(double, Tref) = 300.0;

  // reference pressure
  ADD_ARG(double, Pref) = 1.e5;

  //! Diffusivity [cm^2/s] at reference temperature and pressure
  ADD_ARG(std::vector<double>, diff_c) = {};

  //! Diffusivity temperature exponent
  ADD_ARG(std::vector<double>, diff_T) = {};

  //! Diffusivity pressure exponent
  ADD_ARG(std::vector<double>, diff_P) = {};

  //! Molar volume [cm^3/mol]
  ADD_ARG(std::vector<double>, vm) = {};

  //! Particle diameter [cm]
  ADD_ARG(std::vector<double>, diameter) = {};
}

class EvaporationImpl : public torch::nn::Cloneable<EvaporationImpl> {
 public:
  //! diffusivity, shape (nreaction,)
  torch::Tensor log_diff_c,
      diff_T,  // temperature exponent
      diff_P;  // pressure exponent

  //! molar volume, shape (nreaction,)
  torch::Tensor log_vm;

  //! particle diameter, shape (nreaction,)
  torch::Tensor log_diameter;

  //! options with which this `EvaporationImpl` was constructed
  EvaporationOptions options;

  //! Constructor to initialize the layer
  EvaporationImpl() = default;
  explicit EvaporationImpl(EvaporationOptions const& options_);
  void reset() override;
  void pretty_print(std::ostream& os) const override;

  //! Compute the log rate constant
  /*!
   * \param T temperature [K], shape (...)
   * \param P pressure [pa], shape (...)
   * \param other additional parameters, e.g., concentration
   * \return log rate constant in ln(mol, m, s), shape (..., nreaction)
   */
  torch::Tensor forward(torch::Tensor T, torch::Tensor P,
                        std::map<std::string, torch::Tensor> const& other);
};
TORCH_MODULE(Evaporation);

}  // namespace kintera

#undef ADD_ARG
