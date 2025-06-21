// kintera
#include "kinetic_rate.hpp"

#include <kintera/utils/check_resize.hpp>

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

  nreactions_.clear();

  // register Arrhenius rates
  rc_evaluator.push_back(torch::nn::AnyModule(Arrhenius(options.arrhenius())));
  register_module("arrhenius", rc_evaluator.back().ptr());
  nreactions_.push_back(options.arrhenius().reactions().size());

  // register Coagulation rates
  rc_evaluator.push_back(
      torch::nn::AnyModule(Arrhenius(options.coagulation())));
  register_module("coagulation", rc_evaluator.back().ptr());
  nreactions_.push_back(options.coagulation().reactions().size());

  // register Evaporation rates
  rc_evaluator.push_back(
      torch::nn::AnyModule(Evaporation(options.evaporation())));
  register_module("evaporation", rc_evaluator.back().ptr());
  nreactions_.push_back(options.evaporation().reactions().size());
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
    if (nreactions_[i] == 0) continue;

    torch::Tensor logr;

    if (options.evolve_temperature()) {
      vec1.back() = nreactions_[i];
      auto temp1 = temp.unsqueeze(-1).expand(vec1);
      temp1.requires_grad_(true);

      logr = rc_evaluator[i].forward(temp1, pres, other);
      logr.backward(torch::ones_like(logr));

      logrc_ddT.value().narrow(-1, first, nreactions_[i]) = temp1.grad();
    } else {
      logr = rc_evaluator[i].forward(temp, pres, other);
    }

    // mark reactants
    auto sm = stoich.narrow(1, first, nreactions_[i]).clamp_max(0.).abs();

    std::vector<int64_t> vec2(temp.dim(), 1);
    vec2.push_back(sm.size(0));
    vec2.push_back(sm.size(1));

    // TODO(cli) Take care of zero or negative concentrations
    logr += conc.log().unsqueeze(-2).matmul(sm.view(vec2)).squeeze(-2);

    result.narrow(-1, first, nreactions_[i]) = logr.exp();
    first += nreactions_[i];
  }

  return std::make_pair(result, logrc_ddT);
}

}  // namespace kintera
