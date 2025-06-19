// kintera
#include "kinetic_rate.hpp"

#include <kintera/utils/check_resize.hpp>

namespace kintera {

KineticRateImpl::KineticRateImpl(const KineticRateOptions& options_)
    : options(options_) {
  reset();
}

void KineticRateImpl::reset() {
  auto reactions = options.reactions();
  auto species = options.species();

  // order = register_buffer("order",
  //     torch::zeros({nspecies, nreaction}), torch::kFloat64);
  stoich = register_buffer(
      "stoich",
      torch::zeros({species.size(), reactions.size()}, torch::kFloat64));

  for (int j = 0; j < reactions.size(); ++j) {
    for (int i = 0; i < species.size(); ++i) {
      auto it = reactions[j].reactants().find(species[i]);
      if (it != reactions[j].reactants().end()) {
        stoich[i][j] = -it->second;
      }
      it = reactions[j].products().find(species[i]);
      if (it != reactions[j].products().end()) {
        stoich[i][j] = it->second;
      }
    }
  }

  // placeholder for log rate constant
  logrc_ddT = register_buffer("logrc_ddT", torch::zeros({1}, torch::kFloat64));

  // register Arrhenius rates
  rce.push_back(torch::nn::AnyModule(Arrhenius(options.arrhenius())));
  register_module("arrhenius", rce.back().ptr());

  // register Coagulation rates
  rce.push_back(torch::nn::AnyModule(Arrhenius(options.coagulation())));
  register_module("coagulation", rce.back().ptr());

  // register Evaporation rates
  rce.push_back(torch::nn::AnyModule(Evaporation(options.evaporation())));
  register_module("evaporation", rce.back().ptr());
}

torch::Tensor KineticRateImpl::forward(torch::Tensor temp, torch::Tensor pres,
                                       torch::Tensor conc) {
  // compute Arrhenius rate constants
  std::map<std::string, torch::Tensor> other = {};
  other["conc"] = conc;

  // batch dimensions
  auto vec = temp.sizes().vec();
  vec.push_back(stoich.size(1));

  auto result = torch::empty(vec, temp.options());

  // track rate constant derivative
  if (options.evolve_temperature()) {
    temp.requires_grad_(true);
  }

  logrc_ddT.set_(check_resize(logrc_ddT, vec, temp.options()));

  int first = 0;
  for (int i = 0; i < rce.size(); ++i) {
    auto logr = rce[i].forward(temp, pres, other);
    int nreactions = logr.size(-1);

    if (options.evolve_temperature()) {
      auto identity = torch::eye(nreactions, logr.options());
      logr.backward(identity);
      logrc_ddT.narrow(1, first, nreactions) = temp.grad().unsqueeze(-1);
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

  return result;
}

}  // namespace kintera
