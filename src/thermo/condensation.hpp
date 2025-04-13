#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>

// kintera
#include "nucleation.hpp"

namespace kintera {

struct CondensationOptions {
  static CondensationOptions from_yaml(std::string const& filename);
  CondensationOptions() = default;

  ADD_ARG(int, max_iter) = 10;
  ADD_ARG(std::vector<Nucleation>, react);
  ADD_ARG(std::vector<std::string>, species);
  ADD_ARG(bool, enable_boiling) = true;
};

class CondensationImpl : public torch::nn::Cloneable<CondensationImpl> {
 public:
  //! options with which this `Condensation` was constructed
  CondensationOptions options;

  //! stoichiometry matrix
  torch::Tensor stoich;

  CondensationImpl() = default;
  explicit CondensationImpl(const CondensationOptions& options_);
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
                        torch::Tensor conc, torch::Tensor intEng_RT,
                        torch::Tensor cv_R,
                        torch::optional<torch::Tensor> krate = torch::nullopt);

  torch::Tensor equilibrate_tp(torch::Tensor temp, torch::Tensor pres,
                               torch::Tensor xfrac, int ngas) const;
};
TORCH_MODULE(Condensation);

}  // namespace kintera
