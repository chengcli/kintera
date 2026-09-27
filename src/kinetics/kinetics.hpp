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
#include "kb_falloff.hpp"
#include "lindemann_falloff.hpp"
#include "sri_falloff.hpp"
#include "three_body.hpp"
#include "troe_falloff.hpp"

// arg
#include <kintera/add_arg.h>

namespace kintera {

struct KineticsOptionsImpl final : public SpeciesThermoImpl {
  static std::shared_ptr<KineticsOptionsImpl> create() {
    return std::make_shared<KineticsOptionsImpl>();
  }

  static std::shared_ptr<KineticsOptionsImpl> from_yaml(
      std::string const& filename, bool verbose = false);
  static std::shared_ptr<KineticsOptionsImpl> from_yaml(
      YAML::Node const& config, bool verbose = false);

  std::shared_ptr<KineticsOptionsImpl> clone() const {
    auto op = std::make_shared<KineticsOptionsImpl>(*this);
    if (arrhenius()) op->arrhenius() = arrhenius()->clone();
    if (coagulation()) op->coagulation() = coagulation()->clone();
    if (evaporation()) op->evaporation() = evaporation()->clone();
    if (three_body()) op->three_body() = three_body()->clone();
    if (lindemann_falloff())
      op->lindemann_falloff() = lindemann_falloff()->clone();
    if (troe_falloff()) op->troe_falloff() = troe_falloff()->clone();
    if (sri_falloff()) op->sri_falloff() = sri_falloff()->clone();
    if (kb_falloff()) op->kb_falloff() = kb_falloff()->clone();
    return op;
  }

  static std::shared_ptr<KineticsOptionsImpl> from_kinetics_base(
      std::string const& master_input_path,
      std::string const& photo_catalog_path = "",
      std::string const& cross_dir = "", bool verbose = false);

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

  ADD_ARG(ArrheniusOptions, arrhenius) = ArrheniusOptionsImpl::create();
  ADD_ARG(CoagulationOptions, coagulation) = CoagulationOptionsImpl::create();
  ADD_ARG(EvaporationOptions, evaporation) = EvaporationOptionsImpl::create();
  ADD_ARG(ThreeBodyOptions, three_body) = ThreeBodyOptionsImpl::create();
  ADD_ARG(LindemannFalloffOptions,
          lindemann_falloff) = LindemannFalloffOptionsImpl::create();
  ADD_ARG(TroeFalloffOptions, troe_falloff) = TroeFalloffOptionsImpl::create();
  ADD_ARG(SRIFalloffOptions, sri_falloff) = SRIFalloffOptionsImpl::create();
  ADD_ARG(KBFalloffOptions, kb_falloff) = KBFalloffOptionsImpl::create();
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
  //! 1.0 for reversible reactions, 0.0 otherwise, shape (nreaction_orig,)
  torch::Tensor rev_mask;
  //! product-only stoichiometry (positive values), shape (nspecies,
  //! nreaction_orig)
  torch::Tensor prod_stoich;
  //! reactant-only stoichiometry (positive values), shape (nspecies,
  //! nreaction_orig)
  torch::Tensor react_stoich;
  //! net mole change per reaction, shape (nreaction_orig,)
  torch::Tensor dn;
  //! whether any reversible reactions exist
  bool has_reversible_ = false;
  //! cached raw rate constants from last forward() call (before mass-action)
  mutable torch::Tensor last_kf_;

  // --- split forward/reverse data ---
  //! number of reversible reactions
  int n_reversible_ = 0;
  //! number of original reactions (before augmentation)
  int n_reactions_orig_ = 0;
  //! indices of reversible reactions in the original reaction list
  torch::Tensor rev_indices_;

  //! Constructor to initialize the layer
  KineticsImpl() : options(KineticsOptionsImpl::create()) {}
  explicit KineticsImpl(const KineticsOptions& options_);
  void reset() override;

  //! Compute the reaction-space Jacobian of mass-action rates.
  /*!
   * \param temp   temperature [K], shape (...)
   * \param conc   species concentrations [mol/m^3], shape (..., nspecies)
   * \param cvol   volumetric heat capacity [J/(m^3 K)], shape (...)
   * \param rate   reaction rates after mass-action multiplication
   *               [mol/(m^3 s)], shape (..., nreaction_aug)
   * \param rc_ddC derivative of the raw rate constants with respect to
   *               concentration, shape (..., nspecies, nreaction_aug)
   * \param rc_ddT optional derivative of the raw rate constants with respect
   *               to temperature [mol/(m^3 K s)], shape (..., nreaction_aug)
   * \return       reaction-space Jacobian d(rate_i)/dC_j,
   *               shape (..., nreaction_aug, nspecies)
   *
   * Here `nreaction_aug` is the number of forward reactions plus any
   * appended reverse reactions when reversible chemistry is enabled.
   */
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
  forward(torch::Tensor temp, torch::Tensor pres, torch::Tensor conc,
          std::map<std::string, torch::Tensor> const& extra = {});

 private:
  // used in evaluating jacobian
  std::vector<int> _nreactions;
  void _jacobian_mass_action(torch::Tensor temp, torch::Tensor conc,
                             torch::Tensor cvol, torch::Tensor rate,
                             torch::optional<torch::Tensor> logrc_ddT,
                             int begin, int end, torch::Tensor& out) const;
};

TORCH_MODULE(Kinetics);

}  // namespace kintera
