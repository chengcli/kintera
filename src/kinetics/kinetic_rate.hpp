#pragma once

// torch
#include <torch/nn/cloneable.h>
#include <torch/nn/functional.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/common.h>
#include <torch/nn/modules/container/any.h>

// kintera
#include <kintera/species.hpp>

#include "arrhenius.hpp"
#include "coagulation.hpp"
#include "evaporation.hpp"

// arg
#include <kintera/add_arg.h>

namespace kintera {

struct KineticRateOptions : public SpeciesThermo {
  static KineticRateOptions from_yaml(std::string const& filename);

  KineticRateOptions() = default;

  std::vector<Reaction> reactions() const;

  ADD_ARG(double, Tref) = 298.15;
  ADD_ARG(double, Pref) = 101325.0;

  ADD_ARG(ArrheniusOptions, arrhenius) = {};
  ADD_ARG(CoagulationOptions, coagulation) = {};
  ADD_ARG(EvaporationOptions, evaporation) = {};
  ADD_ARG(bool, evolve_temperature) = false;
};

class KineticRateImpl : public torch::nn::Cloneable<KineticRateImpl> {
 public:
  //! stoichiometry matrix, shape (nspecies, nreaction)
  torch::Tensor stoich;

  //! log rate constant in ln(mol, m, s), shape (..., nreaction)
  torch::Tensor logrc_ddT;

  //! rate constant evaluator
  std::vector<torch::nn::AnyModule> rce;

  //! options with which this `KineticRateImpl` was constructed
  KineticRateOptions options;

  //! Constructor to initialize the layer
  KineticRateImpl() = default;
  explicit KineticRateImpl(const KineticRateOptions& options_);
  void reset() override;

  //! Compute kinetic rate of reactions
  /*!
   * \param temp temperature [K], shape (...)
   * \param pres pressure [Pa], shape (...)
   * \param conc concentration [mol/m^3], shape (..., nspecies)
   * \return kinetic rate of reactions [mol/(m^3 s)], shape (..., nreaction)
   */
  torch::Tensor forward(torch::Tensor temp, torch::Tensor pres,
                        torch::Tensor conc);
};

TORCH_MODULE(KineticRate);

}  // namespace kintera
