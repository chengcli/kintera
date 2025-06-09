// kintera
#include <kintera/constants.h>

#include "eval_uh.hpp"
#include "thermo.hpp"
#include "thermo_dispatch.hpp"

namespace kintera {

ThermoYImpl::ThermoYImpl(const ThermoOptions &options_) : options(options_) {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  if (options.mu_ratio().empty()) {
    options.mu_ratio() = std::vector<double>(nvapor + ncloud, 1.);
  }

  if (options.cref_R().empty()) {
    options.cref_R() = std::vector<double>(nvapor + ncloud, 5. / 2.);
  }

  if (options.uref_R().empty()) {
    options.uref_R() = std::vector<double>(nvapor + ncloud, 0.);
  }

  // populate higher-order thermodynamic functions
  while (options.intEng_R_extra().size() < options.species().size()) {
    options.intEng_R_extra().push_back(nullptr);
  }

  while (options.cv_R_extra().size() < options.species().size()) {
    options.cv_R_extra().push_back(nullptr);
  }

  while (options.cp_R_extra().size() < options.species().size()) {
    options.cp_R_extra().push_back(nullptr);
  }

  while (options.compress_z().size() <= nvapor) {
    options.compress_z().push_back(nullptr);
  }

  reset();
}

void ThermoYImpl::reset() {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  TORCH_CHECK(options.mu_ratio().size() == nvapor + ncloud,
              "mu_ratio size mismatch");
  TORCH_CHECK(options.cref_R().size() == nvapor + ncloud,
              "cref_R size mismatch");
  TORCH_CHECK(options.uref_R().size() == nvapor + ncloud,
              "uref_R size mismatch");

  mu_ratio_m1 = register_buffer(
      "mu_ratio_m1", 1. / torch::tensor(options.mu_ratio(), torch::kFloat64));
  mu_ratio_m1 -= 1.;

  auto cv_R = torch::tensor(options.cref_R(), torch::kFloat64);
  auto uref_R = torch::tensor(options.uref_R(), torch::kFloat64);

  // J/mol/K -> J/kg/K
  cv_ratio_m1 = register_buffer(
      "cv_ratio_m1", cv_R * (options.gammad() - 1.) * (mu_ratio_m1 + 1.));
  cv_ratio_m1 -= 1.;

  u0_R = register_buffer("u0_R",
                         (uref_R - cv_R * options.Tref()) * (mu_ratio_m1 + 1.));

  // populate stoichiometry matrix
  int nspecies = options.species().size();
  int nreact = options.react().size();

  stoich = register_buffer("stoich",
                           torch::zeros({nspecies, nreact}, torch::kFloat64));

  for (int j = 0; j < options.react().size(); ++j) {
    auto const &r = options.react()[j];
    for (int i = 0; i < options.species().size(); ++i) {
      auto it = r.reaction().reactants().find(options.species()[i]);
      if (it != r.reaction().reactants().end()) {
        stoich[i][j] = -it->second;
      }
      it = r.reaction().products().find(options.species()[i]);
      if (it != r.reaction().products().end()) {
        stoich[i][j] = it->second;
      }
    }
  }
}

torch::Tensor ThermoYImpl::f_eps(torch::Tensor yfrac, int n) const {
  auto nvapor = options.vapor_ids().size();
  auto ncloud = options.cloud_ids().size();

  auto yu = yfrac.narrow(0, n, nvapor).unfold(0, nvapor, 1);
  return 1. + yu.matmul(mu_ratio_m1.narrow(0, 0, nvapor)).squeeze(0) -
         yfrac.narrow(0, n + nvapor, ncloud).sum(0);
}

torch::Tensor ThermoYImpl::f_sig(torch::Tensor yfrac, int n) const {
  int ny = options.vapor_ids().size() + options.cloud_ids().size();
  auto yu = yfrac.narrow(0, n, ny).unfold(n, ny, 1);
  return 1. + yu.matmul(cv_ratio_m1).squeeze(0);
}

torch::Tensor ThermoYImpl::compute(std::string ab,
                                   std::initializer_list<torch::Tensor> args,
                                   torch::optional<torch::Tensor> out) const {
  if (ab == "C->Y") {
    out = _conc_to_yfrac(*args.begin(), out);
  } else if (ab == "Y->X") {
    out = _yfrac_to_xfrac(*args.begin());
  } else if (ab == "DY->C") {
    out = _yfrac_to_conc(*args.begin(), *(args.begin() + 1));
  } else if (ab == "DY->cv") {
    out =
        _cv_mean(*args.begin(), *(args.begin() + 1), *(args.begin() + 2), out);
  } else if (ab == "DPY->U") {
    out = _pres_to_intEng(*args.begin(), *(args.begin() + 1),
                          *(args.begin() + 2));
  } else if (ab == "DUY->P") {
    out = _intEng_to_pres(*args.begin(), *(args.begin() + 1),
                          *(args.begin() + 2));
  } else if (ab == "DPY->T") {
    out =
        _pres_to_temp(*args.begin(), *(args.begin() + 1), *(args.begin() + 2));
  } else if (ab == "DTY->P") {
    out =
        _temp_to_pres(*args.begin(), *(args.begin() + 1), *(args.begin() + 2));
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }

  return out.value_or(torch::Tensor());
}

torch::Tensor ThermoYImpl::forward(torch::Tensor rho, torch::Tensor intEng,
                                   torch::Tensor yfrac) {
  auto yfrac0 = yfrac.clone();
  auto conc = _yfrac_to_conc(rho, yfrac);

  // initial guess
  auto pres = _intEng_to_pres(rho, intEng, yfrac);
  auto temp = _pres_to_temp(rho, pres, yfrac);

  // dimensional expanded cv and u0 array
  auto u0 = torch::zeros({1 + (int)options.uref_R().size()}, conc.options());
  auto cv = torch::zeros({1 + (int)options.cref_R().size()}, conc.options());

  u0.narrow(0, 1, options.uref_R().size()) = u0_R * constants::Rgas;
  cv.narrow(0, 1, options.cref_R().size()) =
      constants::Rgas * torch::tensor(options.cref_R(), conc.options());
  cv[0] = constants::Rgas / (options.gammad() - 1.);

  // prepare data
  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(false)
          .declare_static_shape(conc.sizes(), /*squash_dims=*/{conc.dim() - 1})
          .add_output(conc)
          .add_owned_output(temp.unsqueeze(-1))
          .add_owned_input(intEng.unsqueeze(-1))
          .add_input(stoich)
          .add_input(u0)
          .add_input(cv)
          .build();

  // prepare svp function
  user_func1 *logsvp_func = new user_func1[options.react().size()];
  for (int i = 0; i < options.react().size(); ++i) {
    logsvp_func[i] = options.react()[i].func();
  }

  // prepare svp function derivatives
  user_func1 *logsvp_func_ddT = new user_func1[options.react().size()];
  for (int i = 0; i < options.react().size(); ++i) {
    logsvp_func_ddT[i] = options.react()[i].func_ddT();
  }

  // call the equilibrium solver
  at::native::call_equilibrate_uv(
      conc.device().type(), iter, logsvp_func, logsvp_func_ddT,
      options.intEng_R_extra().data(), options.cv_R_extra().data(),
      options.ftol(), options.max_iter());

  delete[] logsvp_func;
  delete[] logsvp_func_ddT;

  _conc_to_yfrac(conc, yfrac);
  return yfrac - yfrac0;
}

torch::Tensor ThermoYImpl::_yfrac_to_xfrac(torch::Tensor yfrac) const {
  int ny = yfrac.size(0);
  TORCH_CHECK(ny == options.vapor_ids().size() + options.cloud_ids().size(),
              "mass fraction size mismatch");

  auto vec = yfrac.sizes().vec();
  for (int i = 0; i < vec.size() - 1; ++i) {
    vec[i] = yfrac.size(i + 1);
  }
  vec.back() = ny + 1;

  auto xfrac = torch::empty(vec, yfrac.options());

  // (ny, ...) -> (..., ny + 1)
  int ndim = yfrac.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  xfrac.narrow(-1, 1, ny) = yfrac.permute(vec) * (mu_ratio_m1 + 1.);
  auto sum = 1. + yfrac.permute(vec).matmul(mu_ratio_m1);
  xfrac.narrow(-1, 1, ny) /= sum.unsqueeze(-1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, ny).sum(-1);
  return xfrac;
}

torch::Tensor ThermoYImpl::_yfrac_to_conc(torch::Tensor rho,
                                          torch::Tensor yfrac) const {
  auto nvapor = options.vapor_ids().size();
  auto ncloud = options.cloud_ids().size();

  auto vec = yfrac.sizes().vec();
  vec.erase(vec.begin());
  vec.push_back(1 + nvapor + ncloud);

  auto result = torch::empty(vec, yfrac.options());

  // (ny, ...) -> (..., ny + 1)
  int ndim = yfrac.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  auto rhod = rho * (1. - yfrac.sum(0));
  result.select(-1, 0) = rhod;
  result.narrow(-1, 1, nvapor + ncloud) =
      rho.unsqueeze(-1) * yfrac.permute(vec) * (mu_ratio_m1 + 1.);
  return result / (constants::Rgas / options.Rd());
}

torch::Tensor ThermoYImpl::_pres_to_intEng(torch::Tensor rho,
                                           torch::Tensor pres,
                                           torch::Tensor yfrac) const {
  auto vec = yfrac.sizes().vec();
  for (int n = 1; n < vec.size(); ++n) vec[n] = 1;

  auto yu0 = options.Rd() * rho * (yfrac * u0_R.view(vec)).sum(0);
  return yu0 + pres * f_sig(yfrac) / f_eps(yfrac) / (options.gammad() - 1.);
}

torch::Tensor ThermoYImpl::_intEng_to_pres(torch::Tensor rho,
                                           torch::Tensor intEng,
                                           torch::Tensor yfrac) const {
  auto vec = yfrac.sizes().vec();
  for (int n = 1; n < vec.size(); ++n) vec[n] = 1;

  auto yu0 = options.Rd() * rho * (yfrac * u0_R.view(vec)).sum(0);
  return (options.gammad() - 1.) * (intEng - yu0) * f_eps(yfrac) / f_sig(yfrac);
}

torch::Tensor ThermoYImpl::_conc_to_yfrac(
    torch::Tensor conc, torch::optional<torch::Tensor> out) const {
  int ny = conc.size(-1) - 1;

  auto vec = conc.sizes().vec();
  for (int i = 0; i < vec.size() - 1; ++i) {
    vec[i + 1] = conc.size(i);
  }
  vec[0] = ny;

  torch::Tensor yfrac;
  if (out.has_value()) {
    TORCH_CHECK(out->sizes() == vec, "Output tensor size mismatch");
    yfrac = out.value();
  } else {
    yfrac = torch::empty(vec, conc.options());
  }

  // (..., ny + 1) -> (ny, ...)
  int ndim = conc.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  yfrac.permute(vec) = conc.narrow(-1, 1, ny) / (mu_ratio_m1 + 1.);

  auto sum = conc.sum(-1) -
             conc.narrow(-1, 1, ny).matmul(mu_ratio_m1 / (mu_ratio_m1 + 1.));
  yfrac /= sum.unsqueeze(0);
  return yfrac;
}

torch::Tensor ThermoYImpl::_cv_mean(torch::Tensor rho, torch::Tensor pres,
                                    torch::Tensor yfrac,
                                    torch::optional<torch::Tensor> out) const {
  int ny = yfrac.size(-1);
  auto temp = compute("DPY->T", {rho, pres, yfrac});
  auto conc = _yfrac_to_conc(rho, yfrac);
  auto cv1 = eval_cv_R(temp, conc, options) * options.Rd();
  auto cvd = cv1.select(-1, 0);

  int ndim = yfrac.dim();
  std::vector<int64_t> vec(ndim);
  for (int i = 0; i < ndim - 1; ++i) vec[i] = i + 1;
  vec[ndim - 1] = 0;
  auto cvy =
      (cv1.narrow(-1, 1, ny) * (1. + mu_ratio_m1) - 1.) * yfrac.permute(vec);

  if (out.has_value()) {
    out = cvd + cvy.sum(-1);
    return out.value();
  } else {
    return cvd + cvy.sum(-1);
  }
}

}  // namespace kintera
