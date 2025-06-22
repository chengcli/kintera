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
    torch::Tensor rate, torch::Tensor rc_ddC,
    torch::optional<torch::Tensor> rc_ddT) const {
  auto vec = temp.sizes().vec();
  vec.push_back(stoich.size(0));
  vec.push_back(stoich.size(1));

  auto stoich_local = (-stoich).clamp_min(0.0).t();

  // forward reaction mask
  auto jacobian = stoich_local * rate.unsqueeze(-1) / conc.unsqueeze(-2);

  // add temperature derivative if provided
  if (rc_ddT.has_value()) {
    auto intEng = eval_intEng_R(temp, conc, options) * constants::Rgas;
    jacobian -= rate.unsqueeze(-1) * rc_ddT.value().unsqueeze(-1) *
                intEng.unsqueeze(-2) / cvol.unsqueeze(-1).unsqueeze(-1);
  }

  return jacobian;
}

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
KineticRateImpl::forward(torch::Tensor temp, torch::Tensor pres,
                         torch::Tensor conc) {
  // prepare data
  std::map<std::string, torch::Tensor> other = {};

  // dimension of reaction rate constants
  auto vec1 = temp.sizes().vec();
  vec1.push_back(stoich.size(1));
  auto result = torch::empty(vec1, temp.options());

  // dimension of rate constant derivatives
  auto vec2 = conc.sizes().vec();
  vec2.push_back(stoich.size(1));
  auto rc_ddC = torch::empty(vec2, conc.options());

  // optional temperature derivative
  torch::optional<torch::Tensor> rc_ddT;

  // track rate constant derivative
  if (options.evolve_temperature()) {
    rc_ddT = torch::empty(vec1, temp.options());
  }

  int first = 0;
  for (int i = 0; i < rc_evaluator.size(); ++i) {
    // no reaction, skip
    if (_nreactions[i] == 0) continue;

    std::cout << "i = " << i << ", first = " << first
              << ", nreactions = " << _nreactions[i] << std::endl;

    other["stoich"] = stoich.narrow(1, first, _nreactions[i]);

    torch::Tensor rate;

    vec2.back() = _nreactions[i];
    auto conc1 = conc.unsqueeze(-1).expand(vec2);
    conc1.requires_grad_(true);

    if (options.evolve_temperature()) {
      vec1.back() = _nreactions[i];
      auto temp1 = temp.unsqueeze(-1).expand(vec1);
      temp1.requires_grad_(true);

      rate = rc_evaluator[i].forward(temp1, pres, conc1, other);
      rate.backward(torch::ones_like(rate));

      rc_ddC.narrow(-1, first, _nreactions[i]) = conc1.grad();
      rc_ddT.value().narrow(-1, first, _nreactions[i]) = temp1.grad();

      std::cout << "first = " << first << std::endl;
      std::cout << "rate = " << rate << std::endl;
    } else {
      rate = rc_evaluator[i].forward(temp, pres, conc1, other);
      rate.backward(torch::ones_like(rate));

      rc_ddC.narrow(-1, first, _nreactions[i]) = conc1.grad();
    }

    result.narrow(-1, first, _nreactions[i]) = rate;
    first += _nreactions[i];
  }

  // mark reactants
  auto sm = stoich.clamp_max(0.).abs();
  result *= conc.unsqueeze(-1).pow(sm).prod(-2);

  return std::make_tuple(result, rc_ddC, rc_ddT);
}

}  // namespace kintera
