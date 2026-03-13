// kintera
#include "kinetics.hpp"

#include <kintera/constants.h>

#include <kintera/thermo/eval_uhs.hpp>
#include <kintera/thermo/nasa9.hpp>
#include "lindemann_falloff.hpp"
#include "sri_falloff.hpp"
#include "three_body.hpp"
#include "troe_falloff.hpp"

namespace kintera {

extern std::vector<std::array<double, 9>> species_nasa9_low;
extern std::vector<std::array<double, 9>> species_nasa9_high;
extern std::vector<double> species_nasa9_Tmid;
extern bool species_has_nasa9;

std::shared_ptr<KineticsImpl> KineticsImpl::create(KineticsOptions const& opts,
                                                   torch::nn::Module* p,
                                                   std::string const& name) {
  TORCH_CHECK(p, "[Kinetics] Parent module is null");
  TORCH_CHECK(opts, "[Kinetics] Options pointer is null");

  return p->register_module(name, Kinetics(opts));
}

KineticsImpl::KineticsImpl(const KineticsOptions& options_)
    : options(options_) {
  populate_thermo(options);
  reset();
}

void KineticsImpl::reset() {
  auto species = options->species();
  auto nspecies = species.size();

  if (options->verbose()) {
    std::cout << "[Kinetics] initializing with species: "
              << fmt::format("{}", species) << std::endl;
  }

  check_dimensions(options);

  if (!options->offset_zero()) {
    // change internal energy offset to T = 0
    for (int i = 0; i < options->uref_R().size(); ++i) {
      options->uref_R()[i] -= options->cref_R()[i] * options->Tref();
    }

    // change entropy offset to T = 1, P = 1
    for (int i = 0; i < options->vapor_ids().size(); ++i) {
      auto Tref = std::max(options->Tref(), 1.);
      auto Pref = std::max(options->Pref(), 1.);
      options->sref_R()[i] -=
          (options->cref_R()[i] + 1) * log(Tref) - log(Pref);
    }

    // set cloud entropy offset to 0 (not used)
    for (int i = options->vapor_ids().size(); i < options->sref_R().size();
         ++i) {
      options->sref_R()[i] = 0.;
    }
    options->offset_zero(true);
  }

  if (options->verbose()) {
    std::cout << "[Kinetics] species uref_R (K) at T = 0: "
              << fmt::format("{}", options->uref_R()) << std::endl;
    std::cout << "[Kinetics] species sref_R (dimensionless): "
              << fmt::format("{}", options->sref_R()) << std::endl;
  }

  auto reactions = options->reactions();
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

  if (options->verbose()) {
    std::cout << "[Kinetics] stoichiometry matrix:\n" << stoich << std::endl;
  }

  _nreactions.clear();

  // register Arrhenius rates
  rc_evaluator.push_back(torch::nn::AnyModule(Arrhenius(options->arrhenius())));
  register_module("arrhenius", rc_evaluator.back().ptr());
  _nreactions.push_back(options->arrhenius()->reactions().size());

  if (options->verbose()) {
    std::cout << "[Kinetics] registered "
              << options->arrhenius()->reactions().size()
              << " Arrhenius reactions" << std::endl;
  }

  // register Coagulation rates
  rc_evaluator.push_back(
      torch::nn::AnyModule(Arrhenius(options->coagulation())));
  register_module("coagulation", rc_evaluator.back().ptr());
  _nreactions.push_back(options->coagulation()->reactions().size());

  if (options->verbose()) {
    std::cout << "[Kinetics] registered "
              << options->coagulation()->reactions().size()
              << " Coagulation reactions" << std::endl;
  }

  // register Evaporation rates
  rc_evaluator.push_back(
      torch::nn::AnyModule(Evaporation(options->evaporation())));
  register_module("evaporation", rc_evaluator.back().ptr());
  _nreactions.push_back(options->evaporation()->reactions().size());

  if (options->verbose()) {
    std::cout << "[Kinetics] registered "
              << options->evaporation()->reactions().size()
              << " Evaporation reactions" << std::endl;
  }

  // register Three-Body rates
  if (options->three_body()->reactions().size() > 0) {
    rc_evaluator.push_back(
        torch::nn::AnyModule(ThreeBody(options->three_body())));
    register_module("three_body", rc_evaluator.back().ptr());
    _nreactions.push_back(options->three_body()->reactions().size());

    if (options->verbose()) {
      std::cout << "[Kinetics] registered "
                << options->three_body()->reactions().size()
                << " Three-Body reactions" << std::endl;
    }
  }

  // register Lindemann Falloff rates
  if (options->lindemann_falloff()->reactions().size() > 0) {
    rc_evaluator.push_back(
        torch::nn::AnyModule(LindemannFalloff(options->lindemann_falloff())));
    register_module("lindemann_falloff", rc_evaluator.back().ptr());
    _nreactions.push_back(options->lindemann_falloff()->reactions().size());

    if (options->verbose()) {
      std::cout << "[Kinetics] registered "
                << options->lindemann_falloff()->reactions().size()
                << " Lindemann Falloff reactions" << std::endl;
    }
  }

  // register Troe Falloff rates
  if (options->troe_falloff()->reactions().size() > 0) {
    rc_evaluator.push_back(
        torch::nn::AnyModule(TroeFalloff(options->troe_falloff())));
    register_module("troe_falloff", rc_evaluator.back().ptr());
    _nreactions.push_back(options->troe_falloff()->reactions().size());

    if (options->verbose()) {
      std::cout << "[Kinetics] registered "
                << options->troe_falloff()->reactions().size()
                << " Troe Falloff reactions" << std::endl;
    }
  }

  // register SRI Falloff rates
  if (options->sri_falloff()->reactions().size() > 0) {
    rc_evaluator.push_back(
        torch::nn::AnyModule(SRIFalloff(options->sri_falloff())));
    register_module("sri_falloff", rc_evaluator.back().ptr());
    _nreactions.push_back(options->sri_falloff()->reactions().size());

    if (options->verbose()) {
      std::cout << "[Kinetics] registered "
                << options->sri_falloff()->reactions().size()
                << " SRI Falloff reactions" << std::endl;
    }
  }

  // register Photolysis rates
  rc_evaluator.push_back(
      torch::nn::AnyModule(Photolysis(options->photolysis())));
  register_module("photolysis", rc_evaluator.back().ptr());
  _nreactions.push_back(options->photolysis()->reactions().size());

  if (options->verbose()) {
    std::cout << "[Kinetics] registered "
              << options->photolysis()->reactions().size()
              << " Photolysis reactions" << std::endl;
  }

  // --- Build reverse reaction metadata ---
  has_reversible_ = false;
  int nrxn = reactions.size();

  auto rev_mask_data = torch::zeros({nrxn}, torch::kFloat64);
  auto prod_stoich_data =
      torch::zeros({(int)nspecies, nrxn}, torch::kFloat64);
  auto dn_data = torch::zeros({nrxn}, torch::kFloat64);

  for (int j = 0; j < nrxn; ++j) {
    auto const& r = reactions[j];
    if (r.reversible()) {
      has_reversible_ = true;
      rev_mask_data[j] = 1.0;
    }
    for (int i = 0; i < (int)species.size(); ++i) {
      auto it = r.products().find(species[i]);
      if (it != r.products().end()) {
        prod_stoich_data[i][j] = it->second;
      }
    }
    double dn_val = 0.0;
    for (auto const& p : r.products()) dn_val += p.second;
    for (auto const& p : r.reactants()) dn_val -= p.second;
    dn_data[j] = dn_val;
  }

  rev_mask = register_buffer("rev_mask", rev_mask_data);
  prod_stoich = register_buffer("prod_stoich", prod_stoich_data);
  dn = register_buffer("dn", dn_data);

  if (species_has_nasa9 && has_reversible_) {
    auto low = torch::zeros({(int)nspecies, 9}, torch::kFloat64);
    auto high = torch::zeros({(int)nspecies, 9}, torch::kFloat64);
    auto tmid = torch::zeros({(int)nspecies}, torch::kFloat64);

    for (int i = 0; i < (int)nspecies; ++i) {
      int gid = options->vapor_ids()[i];
      for (int k = 0; k < 9; ++k) {
        low[i][k] = species_nasa9_low[gid][k];
        high[i][k] = species_nasa9_high[gid][k];
      }
      tmid[i] = species_nasa9_Tmid[gid];
    }

    nasa9_coeffs_low = register_buffer("nasa9_coeffs_low", low);
    nasa9_coeffs_high = register_buffer("nasa9_coeffs_high", high);
    nasa9_Tmid = register_buffer("nasa9_Tmid", tmid);

    if (options->verbose()) {
      std::cout << "[Kinetics] loaded NASA-9 thermo data for "
                << nspecies << " species" << std::endl;
    }
  }

  if (options->verbose()) {
    int n_rev = rev_mask_data.sum().item<int>();
    std::cout << "[Kinetics] " << n_rev << " reversible reactions out of "
              << nrxn << " total" << std::endl;
  }
}

torch::Tensor KineticsImpl::jacobian(
    torch::Tensor temp, torch::Tensor conc, torch::Tensor cvol,
    torch::Tensor rate, torch::Tensor rc_ddC,
    torch::optional<torch::Tensor> rc_ddT) const {
  auto react_st = (-stoich).clamp_min(0.0).t();  // (nrxn, nspecies)
  auto concp_fwd =
      conc.unsqueeze(-2).pow(react_st).prod(-1, /*keepdim=*/true);

  if (!has_reversible_ || !nasa9_coeffs_low.defined()) {
    auto jac =
        concp_fwd * rc_ddC.transpose(-1, -2) +
        react_st * rate.unsqueeze(-1) / conc.unsqueeze(-2).clamp_min(1e-20);

    if (rc_ddT.has_value()) {
      auto intEng = eval_intEng_R(temp, conc, options) * constants::Rgas;
      jac -= concp_fwd * rc_ddT.value().unsqueeze(-1) *
             intEng.unsqueeze(-2) /
             cvol.unsqueeze(-1).unsqueeze(-1);
    }
    return jac;
  }

  // Compute rate_f and rate_r directly using cached rate constants (k_f)
  auto g_RT = nasa9_gibbs_RT(temp, nasa9_coeffs_low, nasa9_coeffs_high,
                              nasa9_Tmid);
  auto delta_g_RT = torch::matmul(g_RT, stoich);
  auto log_standconc =
      torch::log(torch::tensor(options->Pref(), temp.options()) /
                 (constants::Rgas * temp));
  auto Kc = (-delta_g_RT + dn * log_standconc.unsqueeze(-1)).exp();
  constexpr double Kc_min = 1e-60;
  auto kc_valid = (Kc > Kc_min).to(torch::kFloat64);

  auto sm = stoich.clamp_max(0.).abs();
  auto conc_react = conc.unsqueeze(-1).pow(sm).prod(-2);
  auto conc_prod = conc.unsqueeze(-1).pow(prod_stoich).prod(-2);

  auto rate_f = last_kf_ * conc_react;
  auto rate_r = rev_mask * kc_valid *
                (last_kf_ / Kc.clamp_min(Kc_min)) * conc_prod;

  auto prod_st = prod_stoich.t();  // (nrxn, nspecies)
  auto inv_conc = 1.0 / conc.unsqueeze(-2).clamp_min(1e-20);

  auto jac = react_st * rate_f.unsqueeze(-1) * inv_conc -
             prod_st * rate_r.unsqueeze(-1) * inv_conc +
             concp_fwd * rc_ddC.transpose(-1, -2);

  if (rc_ddT.has_value()) {
    auto intEng = eval_intEng_R(temp, conc, options) * constants::Rgas;
    jac -= concp_fwd * rc_ddT.value().unsqueeze(-1) *
           intEng.unsqueeze(-2) /
           cvol.unsqueeze(-1).unsqueeze(-1);
  }

  return jac;
}

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
KineticsImpl::forward(torch::Tensor temp, torch::Tensor pres,
                      torch::Tensor conc) {
  return forward(temp, pres, conc, {});
}

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
KineticsImpl::forward(torch::Tensor temp, torch::Tensor pres,
                      torch::Tensor conc,
                      std::map<std::string, torch::Tensor> const& extra) {
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
  if (options->evolve_temperature()) {
    rc_ddT = torch::empty(vec1, temp.options());
  }

  // other data passed to rate constant evaluator (includes extra for photolysis)
  std::map<std::string, torch::Tensor> other(extra.begin(), extra.end());
  int first = 0;
  for (int i = 0; i < rc_evaluator.size(); ++i) {
    // no reaction, skip
    if (_nreactions[i] == 0) continue;

    other["stoich"] = stoich.narrow(1, first, _nreactions[i]);

    torch::Tensor rate;

    vec2.back() = _nreactions[i];
    auto conc1 = conc.unsqueeze(-1).expand(vec2).clone();
    conc1.requires_grad_(true);

    if (options->evolve_temperature()) {
      vec1.back() = _nreactions[i];
      auto temp1 = temp.unsqueeze(-1).expand(vec1);
      temp1.requires_grad_(true);

      rate = rc_evaluator[i].forward(temp1, pres, conc1, other);

      rate.backward(torch::ones_like(rate));

      if (conc1.grad().defined()) {
        rc_ddC.narrow(-1, first, _nreactions[i]) = conc1.grad();
      } else {
        rc_ddC.narrow(-1, first, _nreactions[i]).fill_(0.);
      }

      if (temp1.grad().defined()) {
        rc_ddT.value().narrow(-1, first, _nreactions[i]) = temp1.grad();
      } else {
        rc_ddT.value().narrow(-1, first, _nreactions[i]).fill_(0.);
      }
    } else {
      rate = rc_evaluator[i].forward(temp, pres, conc1, other);

      if (rate.requires_grad()) {
        rate.backward(torch::ones_like(rate));
      }

      if (conc1.grad().defined()) {
        rc_ddC.narrow(-1, first, _nreactions[i]) = conc1.grad();
      } else {
        rc_ddC.narrow(-1, first, _nreactions[i]).fill_(0.);
      }
    }

    result.narrow(-1, first, _nreactions[i]) = rate;
    first += _nreactions[i];
  }

  // cache raw rate constants for use in jacobian()
  last_kf_ = result.detach().clone();

  // mass-action: multiply rate constant by product of reactant concentrations
  auto sm = stoich.clamp_max(0.).abs();
  auto rate_f = result * conc.unsqueeze(-1).pow(sm).prod(-2);

  // compute reverse rates for reversible reactions
  if (has_reversible_ && nasa9_coeffs_low.defined()) {
    auto g_RT = nasa9_gibbs_RT(temp, nasa9_coeffs_low, nasa9_coeffs_high,
                                nasa9_Tmid);

    auto delta_g_RT = torch::matmul(g_RT, stoich);

    auto log_standconc =
        torch::log(torch::tensor(options->Pref(), temp.options()) /
                   (constants::Rgas * temp));
    auto Kc = (-delta_g_RT + dn * log_standconc.unsqueeze(-1)).exp();
    constexpr double Kc_min = 1e-60;
    auto kc_valid = (Kc > Kc_min).to(torch::kFloat64);

    auto conc_prod = conc.unsqueeze(-1).pow(prod_stoich).prod(-2);
    auto rate_r = rev_mask * kc_valid *
                  (result / Kc.clamp_min(Kc_min)) * conc_prod;

    rate_f = rate_f - rate_r;
  }

  return std::make_tuple(rate_f.detach(), rc_ddC, rc_ddT);
}

}  // namespace kintera
