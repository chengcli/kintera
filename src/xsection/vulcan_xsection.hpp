#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <configure.h>

#include "xsection.hpp"

// arg
#include <kintera/add_arg.h>

namespace kintera {

struct VulcanXsectionOptions {
  ADD_ARG(std::string, cross_file) = "ch4.txt";
  ADD_ARG(std::vector<std::string>, branches) = { "CH4" };
  ADD_ARG(std::vector<std::string>, species);
};

class VulcanXsectionImpl : public torch::nn::Cloneable<VulcanXsectionImpl>,
                           protected XsectionImpl {
 public:
  //! wavelength [nm]
  //! (nwave,)
  torch::Tensor kwave;

  //! photo x-section [cm^2/molecule]
  //! (nwave, nbranch)
  torch::Tensor kdata;

  //! options with which this `VulcanXsectionImpl` was constructed
  VulcanXsectionOptions options;

  //! Constructor to initialize the layer
  VulcanXsectionImpl() = default;
  explicit VulcanXsectionImpl(S8RTOptions const& options_);
  void reset() override;

  //! Get effective stoichiometric coefficients
  //! \param wave wavelength [nm], (nwave, ncol, nlyr)
  //! \param actinic flux [photons nm^-1], (nwave, ncol, nlyr)
  //! \param temp temperature [K], (ncol, nlyr)
  //! \return effective stoichiometric coeff, (ncol, nlyr, nspecies)
  torch::Tensor forward(torch::Tensor wave, torch::Tensor aflux,
                        torch::optional<torch::Tensor> temp = torch::nullopt);
};
TORCH_MODULE(VulcanXsection);

}  // namespace kintera

#undef ADD_ARG
