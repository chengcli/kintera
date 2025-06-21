// kintera
#include "kinetic_rate.hpp"

#include <kintera/constants.h>

#include <kintera/thermo/eval_uhs.hpp>

namespace kintera {

KineticRateImpl::KineticRateImpl(const KineticRateOptions& options_)
    : options(options_) {
  populate_thermo(options);
  reset();
}

void KineticRateImpl::reset() {
  auto species = options.species();
  auto nspecies = species.size();

  check_dimensions(options);

  // change internal energy offset to T = 0
  for (int i = 0; i < options.uref_R().size(); ++i) {
    options.uref_R()[i] -= options.cref_R()[i] * options.Tref();
  }

  // change entropy offset to T = 0
  for (int i = 0; i < options.vapor_ids().size(); ++i) {
    options.sref_R()[i] -=
        (options.cref_R()[i] + 1) * log(options.Tref()) - log(options.Pref());
  }

  // set cloud entropy offset to 0 (not used)
  for (int i = options.vapor_ids().size(); i < options.sref_R().size(); ++i) {
    options.sref_R()[i] = 0.;
  }

  auto reactions = options.reactions();
  // order = register_buffer("order",
  //     torch::zeros({nspecies, nreaction}), torch::kFloat64);
  stoich = register_buffer(
      "stoich",
      torch::zeros({(int)nspecies, (int)reactions.size()}, torch::kFloat64));

  for (int j = 0; j < reactions.size(); ++j) {
    auto const& r = reactions[j];
    for (int i = 0; i < species.size(); ++i) {
      auto it = r.reactants().find(species[i]);
      if (it != r.reactants().end()) {
        stoich[i][j] = -it->second;
      }
      it = r.products().find(species[i]);
      if (it != r.products().end()) {
        stoich[i][j] = it->second;
      }
    }
  }

  _nreactions.clear();

  // register Arrhenius rates
  rc_evaluator.push_back(torch::nn::AnyModule(Arrhenius(options.arrhenius())));
  register_module("arrhenius", rc_evaluator.back().ptr());
  _nreactions.push_back(options.arrhenius().reactions().size());

  // register Coagulation rates
  rc_evaluator.push_back(
      torch::nn::AnyModule(Arrhenius(options.coagulation())));
  register_module("coagulation", rc_evaluator.back().ptr());
  _nreactions.push_back(options.coagulation().reactions().size());

  // register Evaporation rates
  rc_evaluator.push_back(
      torch::nn::AnyModule(Evaporation(options.evaporation())));
  register_module("evaporation", rc_evaluator.back().ptr());
  _nreactions.push_back(options.evaporation().reactions().size());
}

torch::Tensor KineticRateImpl::jacobian(
    torch::Tensor temp, torch::Tensor conc, torch::Tensor cvol,
    torch::Tensor rate, torch::optional<torch::Tensor> logrc_ddT) const {
  auto vec = temp.sizes().vec();
  vec.push_back(stoich.size(0));
  vec.push_back(stoich.size(1));

  auto stoich_local = (-stoich).clamp_min(0.0).t();

  // forward reaction mask
  auto jacobian = stoich_local * rate.unsqueeze(-1) / conc.unsqueeze(-2);

  // add temperature derivative if provided
  if (logrc_ddT.has_value()) {
    auto intEng = eval_intEng_R(temp, conc, options) * constants::Rgas;
    jacobian -= rate.unsqueeze(-1) * logrc_ddT.value().unsqueeze(-1) *
                intEng.unsqueeze(-2) / cvol.unsqueeze(-1).unsqueeze(-1);
  }

  /*int nmass_action = _nreactions[0] + _nreactions[1];
  _jacobian_mass_action(temp, conc, cvol,
                        rate.narrow(-1, 0, nmass_action),
                        logrc_ddT,
                        0, nmass_action,
                        jacobian.narrow(-2, 0, nmass_action));

  int nevaporation = _nreactions[2];
  _jacobian_evaporation(temp, conc, cvol,
                        rate.narrow(-1, nmass_action, nevaporation),
                        logrc_ddT,
                        nmass_action, nmass_action + nevaporation,
                        jacobian.narrow(-2, nmass_action, nevaporation));*/

  return jacobian;
}

std::pair<torch::Tensor, torch::optional<torch::Tensor>>
KineticRateImpl::forward(torch::Tensor temp, torch::Tensor pres,
                         torch::Tensor conc) {
  // prepare data
  std::map<std::string, torch::Tensor> other = {};
  other["conc"] = conc;

  // batch dimensions
  auto vec1 = temp.sizes().vec();

  vec1.push_back(stoich.size(1));
  auto result = torch::empty(vec1, temp.options());

  torch::optional<torch::Tensor> logrc_ddT;

  // track rate constant derivative
  if (options.evolve_temperature()) {
    logrc_ddT = torch::empty(vec1, temp.options());
  }

  int first = 0;
  for (int i = 0; i < rc_evaluator.size(); ++i) {
    // no reaction, skip
    if (_nreactions[i] == 0) continue;

    torch::Tensor logr;

    if (options.evolve_temperature()) {
      vec1.back() = _nreactions[i];
      auto temp1 = temp.unsqueeze(-1).expand(vec1);
      temp1.requires_grad_(true);

      logr = rc_evaluator[i].forward(temp1, pres, other);
      logr.backward(torch::ones_like(logr));

      logrc_ddT.value().narrow(-1, first, _nreactions[i]) = temp1.grad();
    } else {
      logr = rc_evaluator[i].forward(temp, pres, other);
    }

    // mark reactants
    auto sm = stoich.narrow(1, first, _nreactions[i]).clamp_max(0.).abs();

    std::vector<int64_t> vec2(temp.dim(), 1);
    vec2.push_back(sm.size(0));
    vec2.push_back(sm.size(1));

    // TODO(cli) Take care of zero or negative concentrations
    // sanitize concentration
    // logr += conc.log().unsqueeze(-2).matmul(sm.view(vec2)).squeeze(-2);

    result.narrow(-1, first, _nreactions[i]) = logr.exp();
    result.narrow(-1, first, _nreactions[i]) *=
        conc.unsqueeze(-1).pow(sm.view(vec2)).prod(-2);
    first += _nreactions[i];
  }

  return std::make_pair(result, logrc_ddT);
}

}  // namespace kintera
