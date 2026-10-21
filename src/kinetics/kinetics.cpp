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

std::shared_ptr<KineticsImpl> KineticsImpl::create(KineticsOptions const& opts,
                                                   torch::nn::Module* p,
                                                   std::string const& name) {
  TORCH_CHECK(p, "[Kinetics] Parent module is null");
  TORCH_CHECK(opts, "[Kinetics] Options pointer is null");

  return p->register_module(name, Kinetics(opts));
}

KineticsImpl::KineticsImpl(const KineticsOptions& options_)
    : options(options_) {
  TORCH_CHECK(options, "[Kinetics] Options pointer is null");
  populate_thermo(options);
  reset();
}

void KineticsImpl::reset() {
  // a sub-option set to None means no reactions of that type
  if (!options->arrhenius()) options->arrhenius(ArrheniusOptionsImpl::create());
  if (!options->coagulation())
    options->coagulation(CoagulationOptionsImpl::create());
  if (!options->evaporation())
    options->evaporation(EvaporationOptionsImpl::create());
  if (!options->three_body())
    options->three_body(ThreeBodyOptionsImpl::create());
  if (!options->lindemann_falloff())
    options->lindemann_falloff(LindemannFalloffOptionsImpl::create());
  if (!options->troe_falloff())
    options->troe_falloff(TroeFalloffOptionsImpl::create());
  if (!options->sri_falloff())
    options->sri_falloff(SRIFalloffOptionsImpl::create());
  if (!options->kb_falloff())
    options->kb_falloff(KBFalloffOptionsImpl::create());

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

  // register KB Falloff rates
  if (options->kb_falloff() && options->kb_falloff()->reactions().size() > 0) {
    rc_evaluator.push_back(
        torch::nn::AnyModule(KBFalloff(options->kb_falloff())));
    register_module("kb_falloff", rc_evaluator.back().ptr());
    _nreactions.push_back(options->kb_falloff()->reactions().size());

    if (options->verbose()) {
      std::cout << "[Kinetics] registered "
                << options->kb_falloff()->reactions().size()
                << " KB Falloff reactions" << std::endl;
    }
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
  auto orders = react_stoich_data.clone();
  evaporation_start_ = options->arrhenius()->reactions().size() +
                       options->coagulation()->reactions().size();
  evaporation_count_ = options->evaporation()->reactions().size();
  if (evaporation_count_ > 0) {
    auto ngas = options->vapor_ids().size();
    auto col = orders.narrow(1, evaporation_start_, evaporation_count_);
    col.narrow(0, 0, ngas).zero_();
    auto signed_col =
        stoich_base.narrow(1, evaporation_start_, evaporation_count_);
    auto condensed = signed_col.narrow(0, ngas, nspecies - ngas);
    TORCH_CHECK(
        condensed.lt(0).sum(0).eq(1).all().item<bool>() &&
            (condensed.eq(0) | condensed.eq(-1)).all().item<bool>(),
        "Evaporation requires one condensed reactant with coefficient 1 "
        "and no condensed products");
  }
  forward_orders = register_buffer("forward_orders", orders);
  dn = register_buffer("dn", dn_data);

  if (options->verbose()) {
    int n_rev = rev_mask_data.sum().item<int>();
    std::cout << "[Kinetics] " << n_rev << " reversible reactions out of "
              << nrxn << " total" << std::endl;
  }
}

torch::Tensor KineticsImpl::jacobian(
    torch::Tensor temp, torch::Tensor conc, torch::Tensor cvol,
    torch::Tensor rate, torch::Tensor rc_ddC,
    torch::optional<torch::Tensor> rc_ddT,
    torch::optional<torch::Tensor> rate_constants) const {
  // Tiny floor just above IEEE 754 min-normal to avoid 0/0 in rate/conc.
  // Must match the floor in forward() for consistency.
  auto conc_safe =
      conc.clamp_min(conc.scalar_type() == torch::kFloat32 ? 1e-30 : 1e-300);

  // Build augmented reactant stoichiometry: for forward reactions use
  // react_stoich, for reverse reactions use prod_stoich (products of the
  // original forward reaction become reactants of the reverse).
  torch::Tensor react_st_full;
  if (has_reversible_) {
    auto rev_indices = rev_indices_.to(prod_stoich.device(), torch::kLong);
    auto prod_stoich_rev = prod_stoich.index_select(1, rev_indices);
    react_st_full = torch::cat({forward_orders, prod_stoich_rev}, 1);
  } else {
    react_st_full = forward_orders;
  }
  auto react_st = react_st_full.t();  // (nrxn_aug, nspecies)
  auto concp_fwd =
      conc_safe.unsqueeze(-2).pow(react_st).prod(-1, /*keepdim=*/true);

  // Assemble the evaporation concentration factors before autograd can save
  // them. Never overwrite a tensor used by a previous multiplication.
  auto reaction_ids = torch::arange(
      react_st.size(0), forward_orders.options().dtype(torch::kLong));
  auto evaporation_rows =
      reaction_ids.ge(evaporation_start_) &
      reaction_ids.lt(evaporation_start_ + evaporation_count_);
  auto ordinary_order = torch::where(evaporation_rows.unsqueeze(-1),
                                     torch::zeros_like(react_st), react_st);
  // Avoid evaluating discarded 0/C terms, whose backward can overflow at
  // trace/zero concentrations even when the corresponding row is replaced.
  auto denominator = torch::where(ordinary_order.ne(0), conc_safe.unsqueeze(-2),
                                  torch::ones_like(conc_safe).unsqueeze(-2));
  auto mass_derivative = ordinary_order * rate.unsqueeze(-1) / denominator;
  if (evaporation_count_ > 0) {
    auto order =
        forward_orders.narrow(1, evaporation_start_, evaporation_count_).t();
    auto solid = (conc.unsqueeze(-2) * order).sum(-1, true);
    auto cp = solid.clamp_min(0.);
    torch::Tensor k;
    if (rate_constants.has_value()) {
      TORCH_CHECK(rate_constants->sizes() ==
                      rate.narrow(-1, 0, n_reactions_orig_).sizes(),
                  "rate_constants must match the forward reaction rate shape");
      k = rate_constants->narrow(-1, evaporation_start_, evaporation_count_);
    } else {
      TORCH_CHECK(solid.ne(0).all().item<bool>(),
                  "Evaporation at zero solid abundance requires explicit "
                  "rate_constants from forward_with_rate_constants()");
      auto denominator =
          torch::where(solid.gt(0), solid, torch::ones_like(solid));
      k = rate.narrow(-1, evaporation_start_, evaporation_count_) /
          denominator.squeeze(-1);
    }
    auto rows = torch::arange(evaporation_start_,
                              evaporation_start_ + evaporation_count_,
                              forward_orders.options().dtype(torch::kLong));
    concp_fwd = concp_fwd.index_copy(-2, rows, cp);
    mass_derivative = mass_derivative.index_copy(
        -2, rows, k.unsqueeze(-1) * order * solid.ge(0));
  }
  auto jac = concp_fwd * rc_ddC.transpose(-1, -2) + mass_derivative;

  if (rc_ddT.has_value()) {
    auto intEng = eval_intEng_R(temp, conc, options) * constants::Rgas;
    jac -= concp_fwd * rc_ddT.value().unsqueeze(-1) * intEng.unsqueeze(-2) /
           cvol.unsqueeze(-1).unsqueeze(-1);
  }
  return jac;
}

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
KineticsImpl::forward(torch::Tensor temp, torch::Tensor pres,
                      torch::Tensor conc,
                      std::map<std::string, torch::Tensor> const& extra) {
  auto [rate, dc, dt, kf] =
      forward_with_rate_constants(temp, pres, conc, extra);
  return std::make_tuple(rate, dc, dt);
}

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>,
           torch::Tensor>
KineticsImpl::forward_with_rate_constants(
    torch::Tensor temp, torch::Tensor pres, torch::Tensor conc,
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
  other["gas_mask"] =
      torch::arange(conc.size(-1), stoich.options())
          .lt(static_cast<int64_t>(options->vapor_ids().size()));
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

  // Optional external rate-constant override for selected reactions. A tensor
  // `extra["kf_override"]` of shape (..., nreaction_orig) whose FINITE entries
  // replace the computed rate constant (use NaN where no override applies).
  // The overridden reactions' rate-constant derivative rc_ddC is zeroed: the
  // supplied value is treated as constant in concentration, so the mass-action
  // and Jacobian are still assembled by the core engine while a caller injects
  // bespoke rates (e.g. KB UPDATE_CHEMB). Forward-only (irreversible) use.
  {
    auto it = extra.find("kf_override");
    if (it != extra.end() && it->second.numel() > 0) {
      auto ovr = it->second.to(result.dtype());
      auto ovr_mask = torch::isfinite(ovr);
      result = torch::where(ovr_mask, ovr, result);
      rc_ddC = torch::where(ovr_mask.unsqueeze(-2), torch::zeros_like(rc_ddC),
                            rc_ddC);
    }
  }

  // Tiny floor just above IEEE 754 min-normal so that pow() and the
  // Jacobian's rate/conc ratio stay finite.  A large floor (e.g. 1e-20)
  // inflates rates of trace-species reactions and destroys the Jacobian
  // conditioning via the reverse-reaction rc_ddC / Kc amplification.
  auto conc_safe =
      conc.clamp_min(conc.scalar_type() == torch::kFloat32 ? 1e-30 : 1e-300);

  // mass-action forward: k_f * product(C_reactant^stoich)
  auto rate_f = result * conc_safe.unsqueeze(-1).pow(forward_orders).prod(-2);
  if (evaporation_count_ > 0) {
    auto orders =
        forward_orders.narrow(1, evaporation_start_, evaporation_count_);
    auto cp = (conc.unsqueeze(-1) * orders).sum(-2).clamp_min(0.);
    rate_f.narrow(-1, evaporation_start_, evaporation_count_)
        .copy_(result.narrow(-1, evaporation_start_, evaporation_count_) * cp);
  }

  // Split forward/reverse: augment rates with separate reverse entries
  if (has_reversible_ && options->has_nasa9()) {
    auto rev_indices = rev_indices_.to(result.device(), torch::kLong);
    auto nasa9_coeffs_low = options->nasa9_coeffs_low_tensor(temp.options());
    auto nasa9_coeffs_high = options->nasa9_coeffs_high_tensor(temp.options());
    auto nasa9_Tmid = options->nasa9_Tmid_tensor(temp.options());
    torch::Tensor h_RT;
    auto g_RT = nasa9_gibbs_RT(temp, nasa9_coeffs_low, nasa9_coeffs_high,
                               nasa9_Tmid, &h_RT);
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

    // dk_r/dT = (dk_f/dT)/Kc - k_r dlnKc/dT; Kc held where its clamp is active
    if (rc_ddT.has_value()) {
      auto dlnKc =
          ((torch::matmul(h_RT, stoich_base) - dn) / temp.unsqueeze(-1))
              .index_select(-1, rev_indices);
      auto rc_ddT_rev = rc_ddT.value().index_select(-1, rev_indices) /
                            Kc_rev.clamp_min(1e-250) -
                        kr_rev * torch::where(Kc_rev > 1e-250, dlnKc,
                                              torch::zeros_like(dlnKc));
      rc_ddT = torch::cat({rc_ddT.value(), rc_ddT_rev}, -1);
    }
  }

  return std::make_tuple(rate_f.detach(), rc_ddC, rc_ddT, result.detach());
}

}  // namespace kintera
