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
  n_reactions_orig_ = reactions.size();

  // Detect reversible reactions early for stoich augmentation
  std::vector<int64_t> rev_idx_vec;
  for (int j = 0; j < n_reactions_orig_; ++j) {
    if (reactions[j].reversible()) {
      rev_idx_vec.push_back(j);
    }
  }
  n_reversible_ = rev_idx_vec.size();
  has_reversible_ = n_reversible_ > 0;
  if (has_reversible_) {
    rev_indices_ = torch::tensor(rev_idx_vec, torch::kLong);
  }

  // Build base stoichiometry
  auto stoich_base =
      torch::zeros({(int)nspecies, n_reactions_orig_}, torch::kFloat64);
  for (int j = 0; j < n_reactions_orig_; ++j) {
    auto const& r = reactions[j];
    for (int i = 0; i < (int)species.size(); ++i) {
      auto it = r.reactants().find(species[i]);
      if (it != r.reactants().end()) {
        stoich_base[i][j] = -it->second;
      }
      it = r.products().find(species[i]);
      if (it != r.products().end()) {
        stoich_base[i][j] += it->second;
      }
    }
  }

  // Augment stoich with negated columns for reversible reactions
  if (has_reversible_) {
    auto rev_cols = -stoich_base.index_select(1, rev_indices_);
    stoich = register_buffer("stoich", torch::cat({stoich_base, rev_cols}, 1));
  } else {
    stoich = register_buffer("stoich", stoich_base);
  }

  if (options->verbose()) {
    std::cout << "[Kinetics] stoichiometry matrix:\n" << stoich << std::endl;
    if (has_reversible_) {
      std::cout << "[Kinetics] augmented with " << n_reversible_
                << " reverse reaction columns" << std::endl;
    }
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
  int nrxn = n_reactions_orig_;

  auto rev_mask_data = torch::zeros({nrxn}, torch::kFloat64);
  auto prod_stoich_data = torch::zeros({(int)nspecies, nrxn}, torch::kFloat64);
  auto react_stoich_data = torch::zeros({(int)nspecies, nrxn}, torch::kFloat64);
  auto dn_data = torch::zeros({nrxn}, torch::kFloat64);

  for (int j = 0; j < nrxn; ++j) {
    auto const& r = reactions[j];
    if (r.reversible()) {
      rev_mask_data[j] = 1.0;
    }
    for (int i = 0; i < (int)species.size(); ++i) {
      auto it = r.products().find(species[i]);
      if (it != r.products().end()) {
        prod_stoich_data[i][j] = it->second;
      }
      it = r.reactants().find(species[i]);
      if (it != r.reactants().end()) {
        react_stoich_data[i][j] = it->second;
      }
    }
    double dn_val = 0.0;
    for (auto const& p : r.products()) dn_val += p.second;
    for (auto const& p : r.reactants()) dn_val -= p.second;
    dn_data[j] = dn_val;
  }

  rev_mask = register_buffer("rev_mask", rev_mask_data);
  prod_stoich = register_buffer("prod_stoich", prod_stoich_data);
  react_stoich = register_buffer("react_stoich", react_stoich_data);
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
      std::cout << "[Kinetics] loaded NASA-9 thermo data for " << nspecies
                << " species" << std::endl;
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
  // Tiny floor just above IEEE 754 min-normal to avoid 0/0 in rate/conc.
  // Must match the floor in forward() for consistency.
  auto conc_safe = conc.clamp_min(1e-300);

  // Build augmented reactant stoichiometry: for forward reactions use
  // react_stoich, for reverse reactions use prod_stoich (products of the
  // original forward reaction become reactants of the reverse).
  torch::Tensor react_st_full;
  if (has_reversible_) {
    auto rev_indices = rev_indices_.to(prod_stoich.device(), torch::kLong);
    auto prod_stoich_rev = prod_stoich.index_select(1, rev_indices);
    react_st_full = torch::cat({react_stoich, prod_stoich_rev}, 1);
  } else {
    react_st_full = react_stoich;
  }
  auto react_st = react_st_full.t();  // (nrxn_aug, nspecies)
  auto concp_fwd =
      conc_safe.unsqueeze(-2).pow(react_st).prod(-1, /*keepdim=*/true);

  auto jac = concp_fwd * rc_ddC.transpose(-1, -2) +
             react_st * rate.unsqueeze(-1) / conc_safe.unsqueeze(-2);

  if (rc_ddT.has_value()) {
    auto intEng = eval_intEng_R(temp, conc, options) * constants::Rgas;
    jac -= concp_fwd * rc_ddT.value().unsqueeze(-1) * intEng.unsqueeze(-2) /
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
  int nrxn_orig = n_reactions_orig_;

  // dimension of reaction rate constants (original reactions only)
  auto vec1 = temp.sizes().vec();
  vec1.push_back(nrxn_orig);
  auto result = torch::empty(vec1, temp.options());

  // dimension of rate constant derivatives (original reactions only)
  auto vec2 = conc.sizes().vec();
  vec2.push_back(nrxn_orig);
  auto rc_ddC = torch::empty(vec2, conc.options());

  // optional temperature derivative
  torch::optional<torch::Tensor> rc_ddT;

  if (options->evolve_temperature()) {
    rc_ddT = torch::empty(vec1, temp.options());
  }

  // Use base stoich (first nrxn_orig columns) for evaluator dispatch
  auto stoich_base = stoich.narrow(1, 0, nrxn_orig);

  std::map<std::string, torch::Tensor> other(extra.begin(), extra.end());
  int first = 0;
  for (int i = 0; i < (int)rc_evaluator.size(); ++i) {
    if (_nreactions[i] == 0) continue;

    other["stoich"] = stoich_base.narrow(1, first, _nreactions[i]);

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

  last_kf_ = result.detach().clone();

  // Tiny floor just above IEEE 754 min-normal so that pow() and the
  // Jacobian's rate/conc ratio stay finite.  A large floor (e.g. 1e-20)
  // inflates rates of trace-species reactions and destroys the Jacobian
  // conditioning via the reverse-reaction rc_ddC / Kc amplification.
  auto conc_safe = conc.clamp_min(1e-300);

  // mass-action forward: k_f * product(C_reactant^stoich)
  auto rate_f = result * conc_safe.unsqueeze(-1).pow(react_stoich).prod(-2);

  // Split forward/reverse: augment rates with separate reverse entries
  if (has_reversible_ && nasa9_coeffs_low.defined()) {
    auto rev_indices = rev_indices_.to(result.device(), torch::kLong);
    auto g_RT =
        nasa9_gibbs_RT(temp, nasa9_coeffs_low, nasa9_coeffs_high, nasa9_Tmid);
    auto delta_g_RT = torch::matmul(g_RT, stoich_base);
    auto log_standconc =
        torch::log(torch::tensor(options->Pref(), temp.options()) /
                   (constants::Rgas * temp));
    auto Kc = (-delta_g_RT + dn * log_standconc.unsqueeze(-1)).exp();

    // Extract only reversible reactions
    auto Kc_rev = Kc.index_select(-1, rev_indices);
    auto kf_rev = result.index_select(-1, rev_indices);
    auto prod_stoich_rev = prod_stoich.index_select(1, rev_indices);
    auto conc_prod_rev = conc_safe.unsqueeze(-1).pow(prod_stoich_rev).prod(-2);

    // Reverse rate = (k_f / Kc) * product(C_product^stoich), no clamp
    auto kr_rev = kf_rev / Kc_rev.clamp_min(1e-250);
    auto rate_rev = kr_rev * conc_prod_rev;

    // Augment rates: [forward_all, reverse_reversible]
    rate_f = torch::cat({rate_f, rate_rev}, -1);

    // Augment rc_ddC: reverse rate constant derivative = d(k_f)/d[C] / Kc
    auto rc_ddC_fwd_rev = rc_ddC.index_select(-1, rev_indices);
    auto rc_ddC_rev = rc_ddC_fwd_rev / Kc_rev.clamp_min(1e-250).unsqueeze(-2);
    rc_ddC = torch::cat({rc_ddC, rc_ddC_rev}, -1);

    if (rc_ddT.has_value()) {
      auto rc_ddT_fwd_rev = rc_ddT.value().index_select(-1, rev_indices);
      rc_ddT = torch::cat({rc_ddT.value(), rc_ddT_fwd_rev}, -1);
    }
  }

  return std::make_tuple(rate_f.detach(), rc_ddC, rc_ddT);
}

}  // namespace kintera
