// kintera
#include "kinetic_rate.hpp"

#include <kintera/utils/check_resize.hpp>

namespace kintera {

KineticRateImpl::KineticRateImpl(const KineticRateOptions& options_)
    : options(options_) {
  auto nspecies = options.species().size();

  // populate higher-order thermodynamic functions
  while (options.intEng_R_extra().size() < nspecies) {
    options.intEng_R_extra().push_back(nullptr);
  }

  while (options.entropy_R_extra().size() < nspecies) {
    options.entropy_R_extra().push_back(nullptr);
  }

  while (options.cv_R_extra().size() < nspecies) {
    options.cv_R_extra().push_back(nullptr);
  }

  while (options.cp_R_extra().size() < nspecies) {
    options.cp_R_extra().push_back(nullptr);
  }

  while (options.czh().size() < nspecies) {
    options.czh().push_back(nullptr);
  }

  while (options.czh_ddC().size() < nspecies) {
    options.czh_ddC().push_back(nullptr);
  }

  reset();
}

void KineticRateImpl::reset() {
  auto reactions = options.reactions();
  auto species = options.species();
  auto nspecies = species.size();

  TORCH_CHECK(options.cref_R().size() == nspecies,
              "cref_R size = ", options.cref_R().size(),
              ". Expected = ", nspecies);

  TORCH_CHECK(options.uref_R().size() == nspecies,
              "uref_R size = ", options.uref_R().size(),
              ". Expected = ", nspecies);

  TORCH_CHECK(options.sref_R().size() == nspecies,
              "sref_R size = ", options.sref_R().size(),
              ". Expected = ", nspecies);

  // change internal energy offset to T = 0
  for (int i = 0; i < options.uref_R().size(); ++i) {
    options.uref_R()[i] -= options.cref_R()[i] * options.Tref();
  }

  // change entropy offset to T = 0
  for (int i = 0; i < options.vapor_ids().size(); ++i) {
    options.sref_R()[i] -=
        (options.cref_R()[i] + 1) * log(options.Tref()) - log(options.Pref());
  }

  // order = register_buffer("order",
  //     torch::zeros({nspecies, nreaction}), torch::kFloat64);
  stoich = register_buffer(
      "stoich",
      torch::zeros({species.size(), reactions.size()}, torch::kFloat64));

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
