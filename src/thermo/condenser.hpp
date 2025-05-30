#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// kintera
#include "nucleation.hpp"

// arg
#include <kintera/add_arg.h>

namespace kintera {

struct CondenserOptions {
  static CondenserOptions from_yaml(std::string const& filename);
  CondenserOptions() = default;

  ADD_ARG(std::vector<Nucleation>, react);
  ADD_ARG(std::vector<std::string>, species);

  ADD_ARG(int, ngas) = 1;
  ADD_ARG(int, max_iter) = 10;
  ADD_ARG(bool, enable_boiling) = true;
};

class CondenserYImpl : public torch::nn::Cloneable<CondenserYImpl> {
 public:
  //! options with which this `CondenserY` was constructed
  CondenserOptions options;

  //! stoichiometry matrix
  torch::Tensor stoich;

  CondenserYImpl() = default;
  explicit CondenserYImpl(const CondenserOptions& options_);
  void reset() override;

  int species_index(std::string const& name) const {
    auto it =
        std::find(options.species().begin(), options.species().end(), name);
    if (it == options.species().end()) {
      throw std::runtime_error("species not found");
    }
    return std::distance(options.species().begin(), it);
  }

  torch::Tensor forward(torch::Tensor temp, torch::Tensor intEng,
                        torch::Tensor conc);
};
TORCH_MODULE(CondenserY);

class CondenserXImpl : public torch::nn::Cloneable<CondenserXImpl> {
 public:
  //! options with which this `CondenserX` was constructed
  CondenserOptions options;

  //! stoichiometry matrix
  torch::Tensor stoich;

  CondenserXImpl() = default;
  explicit CondenserXImpl(const CondenserOptions& options_);
  void reset() override;

  int species_index(std::string const& name) const {
    auto it =
        std::find(options.species().begin(), options.species().end(), name);
    if (it == options.species().end()) {
      throw std::runtime_error("species not found");
    }
    return std::distance(options.species().begin(), it);
  }

  torch::Tensor forward(torch::Tensor temp, torch::Tensor pres,
                        torch::Tensor xfrac);
};
TORCH_MODULE(CondenserX);

}  // namespace kintera
