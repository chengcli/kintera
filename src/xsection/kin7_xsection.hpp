#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// arg
#include <kintera/add_arg.h>

namespace kintera {

//! Options to initialize `Kin7Xsection`
struct Kin7XsectionOptions {
  ADD_ARG(std::string, cross_file) = "ch4.dat2";
  ADD_ARG(std::vector<std::string>, branches) = {
    "CH4:1",
    "CH3:1 H:1",
    "(1)CH2:1 H2:1",
    "(3)CH2:1 H:2",
    "CH:1 H2:1 H:1"
  };
  ADD_ARG(std::vector<std::string>, species);
  ADD_ARG(int, reaction_id) = 0;
};

//! `Kin7Xsection` is a module to compute photolysis cross-sections
//! and the effective stoichiometric coefficients for the photolysis reaction.
//! The data format is the same as that of the KINETICS7 cross-section file.
class Kin7XsectionImpl : public torch::nn::Cloneable<Kin7XsectionImpl> {
 public:
  //! wavelength [nm]
  //! (nwave,)
  torch::Tensor kwave;

  //! photo x-section [cm^2 molecule^-1]
  //! (nwave, nbranch)
  torch::Tensor kdata;

  //! stoichiometric coefficients of each branch
  //! (nbranch, nspecies)
  torch::Tensor stoich;

  //! options with which this `Kin7XsectionImpl` was constructed
  Kin7XsectionOptions options;

  //! Constructor to initialize the layer
  Kin7XsectionImpl() = default;
  explicit Kin7XsectionImpl(Kin7XsectionOptions const& options_);
  void reset() override;

  //! Get effective stoichiometric coefficients for photo-dissociation
  /*!
   * \param wave wavelength [nm]
   *        (nwave, ncol, nlyr)
   *
   * \param aflux actinic flux [photons nm^-1]
   *        (nwave, ncol, nlyr)
   *
   * \param kcross (out) total photo x-section [cm^2 molecule^-1]
   *        (nreaction, nwave, ncol, nlyr)
   *
   * \param temp temperature [K]
   *        (ncol, nlyr)
   *
   * \return effective stoichiometric coefficients
   *         (ncol, nlyr, nspecies)
   */
  torch::Tensor forward(torch::Tensor wave, torch::Tensor aflux,
                        torch::optional<torch::Tensor> kcross = torch::nullopt,
                        torch::optional<torch::Tensor> temp = torch::nullopt);

 protected:
  // This allows type erasure with default arguments
  FORWARD_HAS_DEFAULT_ARGS({2, torch::nn::AnyValue(torch::nullopt)},
                           {3, torch::nn::AnyValue(torch::nullopt)})
};
TORCH_MODULE(Kin7Xsection);

}  // namespace kintera

#undef ADD_ARG
