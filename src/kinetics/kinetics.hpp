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

struct KineticsOptions : public SpeciesThermo {
  static KineticsOptions from_yaml(std::string const& filename);
  KineticsOptions() = default;
  void report(std::ostream& os) const {
    os << "* Tref = " << Tref() << " K\n"
       << "* Pref = " << Pref() << " Pa\n"
       << "* evolve_temperature = " << (evolve_temperature() ? "true" : "false")
       << "\n";
  }

  std::vector<Reaction> reactions() const;

  ADD_ARG(double, Tref) = 298.15;
  ADD_ARG(double, Pref) = 101325.0;

  ADD_ARG(ArrheniusOptions, arrhenius);
  ADD_ARG(CoagulationOptions, coagulation);
  ADD_ARG(EvaporationOptions, evaporation);

  ADD_ARG(bool, evolve_temperature) = false;
};

class KineticsImpl : public torch::nn::Cloneable<KineticsImpl> {
 public:
  //! stoichiometry matrix, shape (nspecies, nreaction)
  torch::Tensor stoich;

  //! rate constant evaluator
  std::vector<torch::nn::AnyModule> rc_evaluator;

  //! options with which this `KineticsImpl` was constructed
  KineticsOptions options;

  //! Constructor to initialize the layer
  KineticsImpl() = default;
  explicit KineticsImpl(const KineticsOptions& options_);
  void reset() override;

  torch::Tensor jacobian(torch::Tensor temp, torch::Tensor conc,
                         torch::Tensor cvol, torch::Tensor rate,
                         torch::Tensor rc_ddC,
                         torch::optional<torch::Tensor> rc_ddT) const;

  //! Compute kinetic rate of reactions
  /*!
   * \param temp    temperature [K], shape (...)
   * \param pres    pressure [Pa], shape (...)
   * \param conc    concentration [mol/m^3], shape (..., nspecies)
   * \return        (1) kinetic rate of reactions [mol/(m^3 s)],
   *                    shape (..., nreaction)
   *                (2) rate constant derivative with respect to concentration
   *                    [1/s] shape (..., nspecies, nreaction)
   *                (3) optional: rate constant derivative with respect to
   *                    temperature [mol/(m^3 K s], shape (..., nreaction)
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
  forward(torch::Tensor temp, torch::Tensor pres, torch::Tensor conc);

 private:
  // used in evaluating jacobian
  std::vector<int> _nreactions;

  void _jacobian_mass_action(torch::Tensor temp, torch::Tensor conc,
                             torch::Tensor cvol, torch::Tensor rate,
                             torch::optional<torch::Tensor> logrc_ddT,
                             int begin, int end, torch::Tensor& out) const;

  void _jacobian_evaporation(torch::Tensor temp, torch::Tensor conc,
                             torch::Tensor cvol, torch::Tensor rate,
                             torch::optional<torch::Tensor> logrc_ddT,
                             int begin, int end, torch::Tensor& out) const;
};

TORCH_MODULE(Kinetics);

}  // namespace kintera
