// kintera
#include <kintera/constants.h>

#include <kintera/utils/check_resize.hpp>

#include "eval_uhs.hpp"
#include "thermo.hpp"
#include "thermo_dispatch.hpp"

namespace kintera {

ThermoYImpl::ThermoYImpl(const ThermoOptions &options_)
    : options(std::move(options_)) {
  // populate higher-order thermodynamic functions
  auto nspecies = options.species().size();

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

void ThermoYImpl::reset() {
  auto nspecies = options.species().size();

  TORCH_CHECK(options.mu_ratio().size() == nspecies,
              "mu_ratio size = ", options.mu_ratio().size(),
              ". Expected = ", nspecies);

  TORCH_CHECK(options.cref_R().size() == nspecies,
              "cref_R size = ", options.cref_R().size(),
              ". Expected = ", nspecies);

  TORCH_CHECK(options.uref_R().size() == nspecies,
              "uref_R size = ", options.uref_R().size(),
              ". Expected = ", nspecies);

  TORCH_CHECK(options.sref_R().size() == nspecies,
              "sref_R size = ", options.sref_R().size(),
              ". Expected = ", nspecies);

  auto mud = constants::Rgas / options.Rd();
  inv_mu = register_buffer(
      "inv_mu",
      1. / (mud * torch::tensor(options.mu_ratio(), torch::kFloat64)));

  // change internal energy offset to T = 0
  for (int i = 0; i < options.uref_R().size(); ++i) {
    options.uref_R()[i] -= options.cref_R()[i] * options.Tref();
  }

  // change entropy offset to T = 0
  for (int i = 0; i < options.vapor_ids().size(); ++i) {
    options.sref_R()[i] -=
        (options.cref_R()[i] + 1) * log(options.Tref()) - log(options.Pref());
  }

  auto cv_R = torch::tensor(options.cref_R(), torch::kFloat64);
  auto uref_R = torch::tensor(options.uref_R(), torch::kFloat64);

  // J/kg/K
  cv0 = register_buffer("cv0", cv_R * constants::Rgas * inv_mu);

  // J/kg
  u0 = register_buffer("u0", uref_R * constants::Rgas * inv_mu);

  // populate stoichiometry matrix
  auto species = options.species();
  int nspecies = species.size();
  int nreact = options.react().size();

  stoich = register_buffer("stoich",
                           torch::zeros({nspecies, nreact}, torch::kFloat64));

  for (int j = 0; j < nreact; ++j) {
    auto const &r = options.react()[j];
    for (int i = 0; i < nspecies; ++i) {
      auto it = r.reaction().reactants().find(species[i]);
      if (it != r.reaction().reactants().end()) {
        stoich[i][j] = -it->second;
      }
      it = r.reaction().products().find(species[i]);
      if (it != r.reaction().products().end()) {
        stoich[i][j] = it->second;
      }
    }
  }

  // populate buffers
  _D = register_buffer("D", torch::empty({0}));
  _P = register_buffer("P", torch::empty({0}));
  _Y = register_buffer("Y", torch::empty({0}));
  _X = register_buffer("X", torch::empty({0}));
  _V = register_buffer("V", torch::empty({0}));
  _T = register_buffer("T", torch::empty({0}));
  _U = register_buffer("U", torch::empty({0}));
  _S = register_buffer("S", torch::empty({0}));
  _F = register_buffer("F", torch::empty({0}));
  _cv = register_buffer("cv", torch::empty({0}));
}

torch::Tensor const &ThermoYImpl::compute(
    std::string ab, std::initializer_list<torch::Tensor> args) {
  if (ab == "V->Y") {
    _V.set_(*args.begin());
    _ivol_to_yfrac(_V, _Y);
    return _Y;
  } else if (ab == "Y->X") {
    _Y.set_(*args.begin());
    _yfrac_to_xfrac(_Y, _X);
    return _X;
  } else if (ab == "DY->V") {
    _D.set_(*args.begin());
    _Y.set_(*(args.begin() + 1));
    _yfrac_to_ivol(_D, _Y, _V);
    return _V;
  } else if (ab == "PV->T") {
    _P.set_(*args.begin());
    _V.set_(*(args.begin() + 1));
    _pres_to_temp(_P, _V, _T);
    return _T;
  } else if (ab == "VT->cv") {
    _V.set_(*args.begin());
    _T.set_(*(args.begin() + 1));
    _cv_vol(_V, _T, _cv);
    return _cv;
  } else if (ab == "VT->U") {
    _V.set_(*args.begin());
    _T.set_(*(args.begin() + 1));
    _intEng_vol(_V, _T, _U);
    return _U;
  } else if (ab == "VU->T") {
    _V.set_(*args.begin());
    _U.set_(*(args.begin() + 1));
    _intEng_to_temp(_V, _U, _T);
    return _T;
  } else if (ab == "VT->P") {
    _V.set_(*args.begin());
    _T.set_(*(args.begin() + 1));
    _temp_to_pres(_V, _T, _P);
    return _P;
  } else if (ab == "PVT->S") {
    _P.set_(*args.begin());
    _V.set_(*(args.begin() + 1));
    _T.set_(*(args.begin() + 2));
    _entropy_vol(_P, _V, _T, _S);
    return _S;
  } else if (ab == "TUS->F") {
    _T.set_(*args.begin());
    _U.set_(*(args.begin() + 1));
    _S.set_(*(args.begin() + 2));
    _F.set_(_U - _T * _S);
    return _F;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

torch::Tensor ThermoYImpl::forward(torch::Tensor rho, torch::Tensor intEng,
                                   torch::Tensor &yfrac,
                                   torch::optional<torch::Tensor> diag) {
  auto yfrac0 = yfrac.clone();
  auto ivol = compute("DY->V", {rho, yfrac});
  auto vec = ivol.sizes().vec();

  // |reactions| x |reactions| weight matrix
  vec[ivol.dim() - 1] = options.react().size() * options.react().size();
  auto gain = torch::empty(vec, ivol.options());

  // diagnostic array
  vec[ivol.dim() - 1] = 1;
  if (!diag.has_value()) {
    diag = torch::zeros(vec, ivol.options());
  }

  // initial guess
  auto &temp = compute("VU->T", {ivol, intEng});
  auto pres = compute("VT->P", {ivol, temp});
  auto conc = ivol * inv_mu;

  // prepare data
  auto iter =
      at::TensorIteratorConfig()
          .resize_outputs(false)
          .check_all_same_dtype(false)
          .declare_static_shape(conc.sizes(), /*squash_dims=*/{conc.dim() - 1})
          .add_output(gain)
          .add_output(diag.value())
          .add_output(conc)
          .add_owned_output(temp.unsqueeze(-1))
          .add_owned_input(intEng.unsqueeze(-1))
          .add_input(stoich)
          .add_owned_input(u0 / inv_mu)   // J/kg -> J/mol
          .add_owned_input(cv0 / inv_mu)  // J(kg K) -> J/(mol K)
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

  ivol = conc / inv_mu;
  yfrac = compute("V->Y", {ivol});

  vec[ivol.dim() - 1] = options.react().size();
  vec.push_back(options.react().size());
  return gain.view(vec);
}

void ThermoYImpl::_ivol_to_yfrac(torch::Tensor ivol, torch::Tensor &out) const {
  int ny = ivol.size(-1) - 1;
  TORCH_CHECK(ny + 1 == options.vapor_ids().size() + options.cloud_ids().size(),
              "mass fraction size mismatch");

  auto vec = ivol.sizes().vec();
  for (int i = 0; i < vec.size() - 1; ++i) {
    vec[i + 1] = ivol.size(i);
  }
  vec[0] = ny;

  out.set_(check_resize(out, vec, ivol.options()));

  // (..., ny + 1) -> (ny, ...)
  int ndim = ivol.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  out.permute(vec) = ivol.narrow(-1, 1, ny) / ivol.sum(-1, /*keepdim=*/true);
}

void ThermoYImpl::_yfrac_to_xfrac(torch::Tensor yfrac,
                                  torch::Tensor &out) const {
  int ny = yfrac.size(0);
  TORCH_CHECK(ny + 1 == options.vapor_ids().size() + options.cloud_ids().size(),
              "mass fraction size mismatch");

  auto vec = yfrac.sizes().vec();
  for (int i = 0; i < vec.size() - 1; ++i) {
    vec[i] = yfrac.size(i + 1);
  }
  vec.back() = ny + 1;

  out.set_(check_resize(out, vec, yfrac.options()));

  // (ny, ...) -> (..., ny + 1)
  int ndim = yfrac.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  auto mud = constants::Rgas / options.Rd();
  out.narrow(-1, 1, ny) = yfrac.permute(vec) * inv_mu.narrow(0, 1, ny) * mud;

  auto sum = 1. + yfrac.permute(vec).matmul(mud * inv_mu.narrow(0, 1, ny) - 1.);
  out.narrow(-1, 1, ny) /= sum.unsqueeze(-1);
  out.select(-1, 0) = 1. - out.narrow(-1, 1, ny).sum(-1);
}

void ThermoYImpl::_yfrac_to_ivol(torch::Tensor rho, torch::Tensor yfrac,
                                 torch::Tensor &out) const {
  int ny = yfrac.size(0);
  TORCH_CHECK(ny + 1 == options.vapor_ids().size() + options.cloud_ids().size(),
              "mass fraction size mismatch");

  auto vec = yfrac.sizes().vec();
  vec.erase(vec.begin());
  vec.push_back(1 + ny);

  out.set_(check_resize(out, vec, yfrac.options()));

  // (ny, ...) -> (..., ny + 1)
  int ndim = yfrac.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  out.select(-1, 0) = rho * (1. - yfrac.sum(0));
  out.narrow(-1, 1, ny) = (rho.unsqueeze(-1) * yfrac.permute(vec));
}

void ThermoYImpl::_pres_to_temp(torch::Tensor pres, torch::Tensor ivol,
                                torch::Tensor &out) const {
  int ngas = options.vapor_ids().size();

  // kg/m^3 -> mol/m^3
  auto conc_gas = (ivol * inv_mu).narrow(-1, 0, ngas);

  out.set_(pres / (conc_gas.sum(-1) * constants::Rgas));
  int iter = 0;
  while (iter++ < options.max_iter()) {
    auto cz = eval_czh(out, conc_gas, options);
    auto func = out * (cz * conc_gas).sum(-1) - pres / constants::Rgas;
    auto cv_R = eval_cv_R(out, conc_gas, options);
    auto cp_R = eval_cp_R(out, conc_gas, options);
    auto temp_pre = out.clone();
    out += func / ((cp_R - cv_R) * conc_gas).sum(-1);
    if ((1. - temp_pre / out).abs().max().item<double>() < options.ftol()) {
      break;
    }
  }

  if (iter >= options.max_iter()) {
    TORCH_WARN("ThermoYImpl::_pres_to_temp: max iterations reached");
  }
}

void ThermoYImpl::_cv_vol(torch::Tensor ivol, torch::Tensor temp,
                          torch::Tensor &out) const {
  // kg/m^3 -> mol/m^3
  auto conc = ivol * inv_mu;
  auto cv = eval_cv_R(temp, conc, options) * constants::Rgas;
  out.set_((cv * conc).sum(-1));
}

void ThermoYImpl::_intEng_to_temp(torch::Tensor ivol, torch::Tensor intEng,
                                  torch::Tensor &out) const {
  // kg/m^3 -> mol/m^3
  auto u0_sum = (ivol * u0).sum(-1);
  auto cv0_sum = (ivol * cv0).sum(-1);
  auto conc = ivol * inv_mu;

  out.set_((intEng - u0_sum) / cv0_sum);
  int iter = 0;
  while (iter++ < options.max_iter()) {
    auto u = eval_intEng_R(out, conc, options) * constants::Rgas;
    auto cv = eval_cv_R(out, conc, options) * constants::Rgas;
    auto temp_pre = out.clone();
    out += (intEng - (u * conc).sum(-1)) / (cv * conc).sum(-1);
    if ((1. - temp_pre / out).abs().max().item<double>() < options.ftol()) {
      break;
    }
  }

  if (iter >= options.max_iter()) {
    TORCH_WARN("ThermoYImpl::_intEng_to_temp: max iterations reached");
  }
}

void ThermoYImpl::_temp_to_pres(torch::Tensor ivol, torch::Tensor temp,
                                torch::Tensor &out) const {
  int ngas = options.vapor_ids().size();

  // kg/m^3 -> mol/m^3
  auto conc_gas = (ivol * inv_mu).narrow(-1, 0, ngas);
  auto cz = eval_czh(temp, conc_gas, options);
  out.set_(constants::Rgas * temp * (cz * conc_gas).sum(-1));
}

}  // namespace kintera
