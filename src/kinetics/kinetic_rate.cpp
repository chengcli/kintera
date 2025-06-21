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

  // register Arrhenius rates
  rc_evaluator.push_back(torch::nn::AnyModule(Arrhenius(options.arrhenius())));
  register_module("arrhenius", rc_evaluator.back().ptr());

  // register Coagulation rates
  rc_evaluator.push_back(
      torch::nn::AnyModule(Arrhenius(options.coagulation())));
  register_module("coagulation", rc_evaluator.back().ptr());

  // register Evaporation rates
  rc_evaluator.push_back(
      torch::nn::AnyModule(Evaporation(options.evaporation())));
  register_module("evaporation", rc_evaluator.back().ptr());
}

std::pair<torch::Tensor, torch::optional<torch::Tensor>>
KineticRateImpl::forward(torch::Tensor temp, torch::Tensor pres,
                         torch::Tensor conc) {
  // prepare data
  std::map<std::string, torch::Tensor> other = {};
  other["conc"] = conc;

  // batch dimensions
  auto vec = temp.sizes().vec();
  vec.push_back(stoich.size(1));

  auto result = torch::empty(vec, temp.options());

  torch::optional<torch::Tensor> logrc_ddT;

  // track rate constant derivative
  if (options.evolve_temperature()) {
    temp.requires_grad_(true);
    logrc_ddT = torch::empty(vec, temp.options());
  }

  int first = 0;
  for (auto rc : rc_evaluator) {
    auto logr = rc.forward(temp, pres, other);
    int nreactions = logr.size(-1);

    if (options.evolve_temperature()) {
      auto identity = torch::eye(nreactions, logr.options());
      logr.backward(identity);
      logrc_ddT.value().narrow(1, first, nreactions) =
          temp.grad().unsqueeze(-1);
    }

    // mark reactants
    auto sm = stoich.narrow(1, first, nreactions).clamp_max(0.).abs();

    std::vector<int64_t> vec2(temp.dim(), 1);
    vec2.push_back(sm.size(0));
    vec2.push_back(sm.size(1));

    logr += conc.log().unsqueeze(-2).matmul(sm.view(vec2)).squeeze(-2);
    result.narrow(1, first, nreactions) = logr.exp();

    first += nreactions;
  }

  if (options.evolve_temperature()) {
    temp.requires_grad_(false);
  }

  return std::make_pair(result, logrc_ddT);
}

}  // namespace kintera
