#pragma once

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

namespace kintera {

struct KineticRateOptions {
  ADD_ARG(std::vector<std::string>, species) = {};
  ADD_ARG(std::vector<Reaction>, reactions) = {};
};

class KineticRateImpl : public torch::nn::Cloneable<KineticRateImpl> {
 public:
  //! activity order matrix, shape (nreaction, nspecies)
  torch::Tensor order;

  //! stoichiometry matrix, shape (nreaction, nspecies)
  torch::Tensor stoich;

  //! options with which this `KineticRateImpl` was constructed
  KineticRateOptions options;

  //! Constructor to initialize the layer
  KineticRateImpl() = default;
  explicit KineticRateImpl(const KineticRateOptions& options_);
  void reset() override;

  //! Compute kinetic rate of reactions
  /*!
   * \param conc concentration [kmol/m^3], shape (ncol, nlyr, nspecies)
   * \param log_rate_constant log rate constant in ln(kmol, m, s),
   *        shape (ncol, nlyr, nreaction)
   * \return kinetic rate of reactions [kmol/m^3/s],
   *        shape (ncol, nlyr, nreaction)
   */
  torch::Tensor forward(torch::Tensor conc, torch::Tensor log_rate_constant);
};

TORCH_MODULE(KineticRate);

}  // namespace kintera
