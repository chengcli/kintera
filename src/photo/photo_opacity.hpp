#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
// clang-format off
#include <configure.h>
#include <add_arg.h>
// clang-format on

#include "reaction.hpp"
#include "xsection/xsection.hpp"

namespace kintera {

struct PhotoOpacityOptions {
  ADD_ARG(std::vector<Reaction>, reactions) = {};
  ADD_ARG(std::vector<std::string>, species);
};

class PhotoOpacityImpl : public torch::nn::Cloneable<PhotoOpacityImpl> {
 public:
  //! interpolated photo x-section [cm^2 molecule^-1] at common wavelengths
  std::vector<Xsection> xsections;

  //! options with which this `PhotoOpacityImpl` was constructed
  PhotoOpacityOptions options;

  //! Constructor to initialize the layer
  PhotoOpacityImpl() = default;
  explicit PhotoOpacityImpl(PhotoOpacityOptions const& options_);
  void reset() override;

  //! Get photolysis optical properties
  //! \param conc mole concentration [molecules cm^-3], (ncol, nlyr, nspecies)
  //! \param temp temperature [K], (ncol, nlyr)
  //! \return photolysis opacities [m^-1], (nwave, ncol, nlyr)
  torch::Tensor forward(torch::Tensor conc,
                        torch::optional<torch::Tensor> temp = torch::nullopt);
};
TORCH_MODULE(PhotoOpacity);

}  // namespace kintera
