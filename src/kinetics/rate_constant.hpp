#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include "arrhenius.hpp"

// arg
#include <kintera/add_arg.h>

namespace kintera {

struct RateConstantOptions {
  ADD_ARG(std::vector<std::string>, types) = {};
  ADD_ARG(std::string, reaction_file) = "";
};

class RateConstantImpl : public torch::nn::Cloneable<RateConstantImpl> {
 public:
  //! start reaction index of each reaction type
  std::vector<int> rxn_id_start;

  //! end reaction index of each reaction type (exclusive)
  std::vector<int> rxn_id_end;

  //! options with which this `RateConstantImpl` was constructed
  RateConstantOptions options;

  //! submodule: evaluate reaction rate constants
  std::vector<torch::nn::AnyModule> eval_rate_constants;

  //! Constructor to initialize the layer
  RateConstantImpl() = default;
  explicit RateConstantImpl(const RateConstantOptions& options_);
  void reset() override;

  //! Compute reaction rate constant
  /*!
   * \param T temperature [K], shape (ncol, nlyr)
   * \param other other parameters
   * \return log rate constant in ln(kmol, m, s), shape (ncol, nlyr, nreaction)
   */
  torch::Tensor forward(torch::Tensor T,
                        std::map<std::string, torch::Tensor> const& other);
};

TORCH_MODULE(RateConstant);

}  // namespace kintera
