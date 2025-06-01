// kintera
#include <kintera/constants.h>

#include "thermo.hpp"

namespace kintera {

void call_equilibrate_uv_cpu(at::TensorIterator &iter,
                             user_func1 const *logsvp_func,
                             user_func1 const *logsvp_func_ddT,
                             user_func1 const *intEng_extra,
                             user_func1 const *intEng_extra_ddT,
                             double logsvp_eps, int max_iter);

ThermoYImpl::ThermoYImpl(const ThermoOptions& options_) : options(options_) {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  if (options.mu_ratio().empty()) {
    options.mu_ratio() = std::vector<double>(nvapor + ncloud, 1.);
  }

  if (options.cref_R().empty()) {
    options.cref_R() = std::vector<double>(nvapor + ncloud, 5./2.);
  }

  if (options.uref_R().empty()) {
    options.uref_R() = std::vector<double>(nvapor + ncloud, 0.);
  }

  // populate higher-order internal energy and cv functions
  while (options.intEng_extra().size() < options.species().size()) {
    options.intEng_extra().push_back(nullptr);
  }

  while (options.cv_extra().size() < options.species().size()) {
    options.cv_extra().push_back(nullptr);
  }

  reset();
}

void ThermoYImpl::reset() {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  TORCH_CHECK(options.mu_ratio().size() == nvapor + ncloud,
              "mu_ratio size mismatch");
  TORCH_CHECK(options.cref_R().size() == nvapor + ncloud, "cref_R size mismatch");
  TORCH_CHECK(options.uref_R().size() == nvapor + ncloud, "uref_R size mismatch");

  mu_ratio_m1 = register_buffer(
      "mu_ratio_m1", 1. / torch::tensor(options.mu_ratio(), torch::kFloat64));
  mu_ratio_m1 -= 1.;

  auto cv_R = torch::tensor(options.cref_R(), torch::kFloat64);
  auto uref_R = torch::tensor(options.uref_R(), torch::kFloat64);

  // J/mol/K -> J/kg/K
  cv_ratio_m1 = register_buffer("cv_ratio_m1", 
      cv_R * (options.gammad() - 1.) * (mu_ratio_m1 + 1.));
  cv_ratio_m1 -= 1.;

  u0_R = register_buffer("u0_R", (uref_R - cv_R * options.Tref()) * (mu_ratio_m1 + 1.));

  // populate stoichiometry matrix
  int nspecies = options.species().size();
  int nreact = options.react().size();

  stoich = register_buffer("stoich",
                           torch::zeros({nspecies, nreact}, torch::kFloat64));

  for (int j = 0; j < options.react().size(); ++j) {
    auto const& r = options.react()[j];
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

torch::Tensor ThermoYImpl::f_eps(torch::Tensor yfrac) const {
  auto nvapor = options.vapor_ids().size();
  auto ncloud = options.cloud_ids().size();

  auto yu = yfrac.narrow(0, 0, nvapor).unfold(0, nvapor, 1);
  return 1. + yu.matmul(mu_ratio_m1.narrow(0, 0, nvapor)).squeeze(0) -
         yfrac.narrow(0, nvapor, ncloud).sum(0);
}

torch::Tensor ThermoYImpl::f_sig(torch::Tensor yfrac) const {
  int nmass = options.vapor_ids().size() + options.cloud_ids().size();
  auto yu = yfrac.narrow(0, 0, nmass).unfold(0, nmass, 1);
  return 1. + yu.matmul(cv_ratio_m1).squeeze(0);
}

torch::Tensor ThermoYImpl::compute(std::string ab, ...) const {
  va_list args;
  torch::Tensor result;
  va_start(args, ab);

  if (ab == "C->Y") {
    auto conc = va_arg(args, torch::Tensor);
    result = _conc_to_yfrac(conc);
  } else if (ab == "Y->X") {
    auto yfrac = va_arg(args, torch::Tensor);
    result = _yfrac_to_xfrac(yfrac);
  } else if (ab == "DY->C") {
    auto rho = va_arg(args, torch::Tensor);
    auto yfrac = va_arg(args, torch::Tensor);
    result = _yfrac_to_conc(rho, yfrac);
  } else if (ab == "DPY->U") {
    auto rho = va_arg(args, torch::Tensor);
    auto pres = va_arg(args, torch::Tensor);
    auto yfrac = va_arg(args, torch::Tensor);
    result = _pres_to_intEng(rho, pres, yfrac);
  } else if (ab == "DUY->P") {
    auto rho = va_arg(args, torch::Tensor);
    auto intEng = va_arg(args, torch::Tensor);
    auto yfrac = va_arg(args, torch::Tensor);
    result = _intEng_to_pres(rho, intEng, yfrac);
  } else if (ab == "DPY->T") {
    auto rho = va_arg(args, torch::Tensor);
    auto pres = va_arg(args, torch::Tensor);
    auto yfrac = va_arg(args, torch::Tensor);
    result = _pres_to_temp(rho, pres, yfrac);
  } else if (ab == "DTY->P") {
    auto rho = va_arg(args, torch::Tensor);
    auto temp = va_arg(args, torch::Tensor);
    auto yfrac = va_arg(args, torch::Tensor);
    result = _temp_to_pres(rho, temp, yfrac);
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }

  va_end(args);
  return result;
}

torch::Tensor ThermoYImpl::forward(torch::Tensor rho, torch::Tensor intEng,
                                   torch::Tensor yfrac) {
  auto yfrac0 = yfrac.clone();
  auto conc = _yfrac_to_conc(rho, yfrac);

  // initial guess
  auto pres = _intEng_to_pres(rho, intEng, yfrac);
  auto temp = _pres_to_temp(rho, pres, yfrac);

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
        .add_owned_input(torch::tensor(options.uref_R(), conc.options()))
        .add_owned_input(torch::tensor(options.cref_R(), conc.options()))
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
  if (conc.is_cpu()) {
    call_equilibrate_uv_cpu(iter, logsvp_func, logsvp_func_ddT,
                            options.intEng_extra().data(),
                            options.cv_extra().data(),
                            options.ftol(), options.max_iter());
  } else if (conc.is_cuda()) {
    TORCH_CHECK(false, "CUDA support not implemented yet");
  } else {
    TORCH_CHECK(false, "Unsupported tensor type");
  }

  delete[] logsvp_func;
  delete[] logsvp_func_ddT;

  yfrac = _conc_to_yfrac(conc);
  return yfrac - yfrac0;
}

torch::Tensor ThermoYImpl::_yfrac_to_xfrac(torch::Tensor yfrac) const {
  int nmass = yfrac.size(0);
  TORCH_CHECK(nmass == options.vapor_ids().size() + options.cloud_ids().size(),
              "mass fraction size mismatch");

  auto vec = yfrac.sizes().vec();
  for (int i = 0; i < vec.size() - 1; ++i) {
    vec[i] = yfrac.size(i + 1);
  }
  vec.back() = nmass + 1;

  auto xfrac = torch::empty(vec, yfrac.options());

  // (nmass, ...) -> (..., nmass + 1)
  int ndim = yfrac.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  xfrac.narrow(-1, 1, nmass) = yfrac.permute(vec) * (mu_ratio_m1 + 1.);
  auto sum = 1. + yfrac.permute(vec).matmul(mu_ratio_m1);
  xfrac.narrow(-1, 1, nmass) /= sum.unsqueeze(-1);
  xfrac.select(-1, 0) = 1. - xfrac.narrow(-1, 1, nmass).sum(-1);
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

  // (nmass, ...) -> (..., nmass + 1)
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

torch::Tensor ThermoYImpl::_pres_to_intEng(torch::Tensor rho, torch::Tensor pres,
                                           torch::Tensor yfrac) const {
  auto vec = yfrac.sizes().vec();
  for (int n = 1; n < vec.size(); ++n) vec[n] = 1;

  auto yu0 = options.Rd() * rho * (yfrac * u0_R.view(vec)).sum(0);
  return yu0 + pres * f_sig(yfrac) / f_eps(yfrac) / (options.gammad() - 1.);
}

torch::Tensor ThermoYImpl::_intEng_to_pres(torch::Tensor rho, torch::Tensor intEng,
                                           torch::Tensor yfrac) const {
  auto vec = yfrac.sizes().vec();
  for (int n = 1; n < vec.size(); ++n) vec[n] = 1;

  auto yu0 = options.Rd() * rho * (yfrac * u0_R.view(vec)).sum(0);
  return (options.gammad() - 1.) * (intEng - yu0) * f_eps(yfrac) / f_sig(yfrac);
}

// FIXME
torch::Tensor ThermoYImpl::_conc_to_yfrac(torch::Tensor conc) const {
  return conc;
}

}  // namespace kintera
