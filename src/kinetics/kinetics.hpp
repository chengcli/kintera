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
#include "lindemann_falloff.hpp"
#include "photolysis.hpp"
#include "sri_falloff.hpp"
#include "three_body.hpp"
#include "troe_falloff.hpp"

// arg
#include <kintera/add_arg.h>

namespace kintera {

struct KineticsOptionsImpl final : public SpeciesThermoImpl {
  static std::shared_ptr<KineticsOptionsImpl> create() {
    auto op = std::make_shared<KineticsOptionsImpl>();
    op->arrhenius() = ArrheniusOptionsImpl::create();
    op->coagulation() = CoagulationOptionsImpl::create();
    op->evaporation() = EvaporationOptionsImpl::create();
    op->three_body() = ThreeBodyOptionsImpl::create();
    op->lindemann_falloff() = LindemannFalloffOptionsImpl::create();
    op->troe_falloff() = TroeFalloffOptionsImpl::create();
    op->sri_falloff() = SRIFalloffOptionsImpl::create();
    op->photolysis() = PhotolysisOptionsImpl::create();
    return op;
  }

  static std::shared_ptr<KineticsOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);
  static std::shared_ptr<KineticsOptionsImpl> from_yaml(
      YAML::Node const& config, bool verbose = false);

  void report(std::ostream& os) const {
    os << "-- kinetics options --\n";
    os << "* Tref = " << Tref() << " K\n"
       << "* Pref = " << Pref() << " Pa\n"
       << "* offset_zero = " << (offset_zero() ? "true" : "false") << "\n"
       << "* evolve_temperature = " << (evolve_temperature() ? "true" : "false")
       << "\n";
  }

  std::vector<Reaction> reactions() const;

  ADD_ARG(double, Tref) = 298.15;
  ADD_ARG(double, Pref) = 101325.0;

  ADD_ARG(ArrheniusOptions, arrhenius);
  ADD_ARG(CoagulationOptions, coagulation);
  ADD_ARG(EvaporationOptions, evaporation);
  ADD_ARG(ThreeBodyOptions, three_body);
  ADD_ARG(LindemannFalloffOptions, lindemann_falloff);
  ADD_ARG(TroeFalloffOptions, troe_falloff);
  ADD_ARG(SRIFalloffOptions, sri_falloff);
  ADD_ARG(PhotolysisOptions, photolysis);

  ADD_ARG(bool, evolve_temperature) = false;
  ADD_ARG(bool, verbose) = false;
  ADD_ARG(bool, offset_zero) = false;
};
using KineticsOptions = std::shared_ptr<KineticsOptionsImpl>;

class KineticsImpl : public torch::nn::Cloneable<KineticsImpl> {
 public:
  //! Create and register a `KineticsImpl` module
  /*!
   * This function registers the created module as a submodule
   * of the given parent module `p`.
   *
   * \param[in] opts  options for constructing the `KineticsImpl`
   * \param[in] p     parent module for registering the created module
   * \return          created `KineticsImpl` module
   */
  static std::shared_ptr<KineticsImpl> create(
      KineticsOptions const& opts, torch::nn::Module* p,
      std::string const& name = "kinetics");

  //! stoichiometry matrix, shape (nspecies, nreaction)
  torch::Tensor stoich;

  //! rate constant evaluator
  std::vector<torch::nn::AnyModule> rc_evaluator;

  //! options with which this `KineticsImpl` was constructed
  KineticsOptions options;

  // --- reverse reaction data ---
  //! 1.0 for reversible reactions, 0.0 otherwise, shape (nreaction,)
  torch::Tensor rev_mask;
  //! product-only stoichiometry (positive values), shape (nspecies, nreaction)
  torch::Tensor prod_stoich;
  //! net mole change per reaction, shape (nreaction,)
  torch::Tensor dn;
  //! NASA-9 coefficients, shape (nspecies, 9)
  torch::Tensor nasa9_coeffs_low, nasa9_coeffs_high;
  //! NASA-9 mid-temperature, shape (nspecies,)
  torch::Tensor nasa9_Tmid;
  //! whether any reversible reactions exist
  bool has_reversible_ = false;
  //! cached raw rate constants from last forward() call (before mass-action)
  mutable torch::Tensor last_kf_;

  //! Constructor to initialize the layer
  KineticsImpl() : options(KineticsOptionsImpl::create()) {}
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

  //! Compute kinetic rate with extra data (e.g. actinic flux for photolysis)
  std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
  forward(torch::Tensor temp, torch::Tensor pres, torch::Tensor conc,
          std::map<std::string, torch::Tensor> const& extra);

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
