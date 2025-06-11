// kintera
#include <kintera/constants.h>

#include <kintera/utils/check_resize.hpp>

#include "eval_uh.hpp"
#include "thermo.hpp"
#include "thermo_dispatch.hpp"

namespace kintera {

ThermoYImpl::ThermoYImpl(const ThermoOptions &options_) : options(options_) {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  if (options.mu_ratio().empty()) {
    options.mu_ratio() = std::vector<double>(1 + nvapor + ncloud, 1.);
  }

  if (options.cref_R().empty()) {
    options.cref_R() = std::vector<double>(1 + nvapor + ncloud, 5. / 2.);
  }

  if (options.uref_R().empty()) {
    options.uref_R() = std::vector<double>(1 + nvapor + ncloud, 0.);
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

  while (options.czh().size() < options.species().size()) {
    options.czh().push_back(nullptr);
  }

  while (options.czh_ddC().size() < options.species().size()) {
    options.czh_ddC().push_back(nullptr);
  }

  reset();
}

void ThermoYImpl::reset() {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  TORCH_CHECK(options.mu_ratio().size() == 1 + nvapor + ncloud,
              "mu_ratio size mismatch");
  TORCH_CHECK(options.cref_R().size() == 1 + nvapor + ncloud,
              "cref_R size mismatch");
  TORCH_CHECK(options.uref_R().size() == 1 + nvapor + ncloud,
              "uref_R size mismatch");

  auto mud = constants::Rgas / options.Rd();
  inv_mu = register_buffer(
      "inv_mu",
      1. / (mud * torch::tensor(options.mu_ratio(), torch::kFloat64)));

  auto cv_R = torch::tensor(options.cref_R(), torch::kFloat64);
  auto uref_R = torch::tensor(options.uref_R(), torch::kFloat64);

  // J/mol/K -> J/kg/K
  cv0 = register_buffer("cv0", cv_R * constants::Rgas * inv_mu);
  u0 = register_buffer(
      "u0", uref_R * constants::Rgas * inv_mu - cv0 * options.Tref());

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

torch::Tensor ThermoYImpl::compute(
    std::string ab, std::initializer_list<torch::Tensor> args) const {
  if (ab == "V->Y") {
    _V.resize_as_(*args.begin());
    _V.copy_(*args.begin());
    _ivol_to_yfrac(_V, _Y);
    return _Y;
  } else if (ab == "Y->X") {
    _Y.resize_as_(*args.begin());
    _Y.copy_(*args.begin());
    _yfrac_to_xfrac(_Y, _X);
    return _X;
  } else if (ab == "DY->V") {
    _D.resize_as_(*args.begin());
    _D.copy_(*args.begin());
    _Y.resize_as_(*(args.begin() + 1));
    _Y.copy_(*(args.begin() + 1));
    _yfrac_to_ivol(_D, _Y, _V);
    return _V;
  } else if (ab == "PV->T") {
    _P.resize_as_(*args.begin());
    _P.copy_(*args.begin());
    _V.resize_as_(*(args.begin() + 1));
    _V.copy_(*(args.begin() + 1));
    _pres_to_temp(_P, _V, _T);
    return _T;
  } else if (ab == "VT->cv") {
    _V.resize_as_(*args.begin());
    _V.copy_(*args.begin());
    _T.resize_as_(*(args.begin() + 1));
    _T.copy_(*(args.begin() + 1));
    _cv_vol(_V, _T, _cv);
    return _cv;
  } else if (ab == "VT->U") {
    _T.resize_as_(*args.begin());
    _T.copy_(*args.begin());
    _V.resize_as_(*(args.begin() + 1));
    _V.copy_(*(args.begin() + 1));
    _temp_to_intEng(_T, _V, _U);
  } else if (ab == "VU->T") {
    _V.resize_as_(*args.begin());
    _V.copy_(*args.begin());
    _U.resize_as_(*(args.begin() + 1));
    _U.copy_(*(args.begin() + 1));
    _intEng_to_temp(_V, _U, _T);
    return _U;
  } else if (ab == "VT->P") {
    _V.resize_as_(*args.begin());
    _V.copy_(*args.begin());
    _T.resize_as_(*(args.begin() + 1));
    _T.copy_(*(args.begin() + 1));
    _temp_to_pres(_V, _T, _P);
  } else if (ab == "PVT->S") {
    // TODO(cli)
  } else if (ab == "TUS->F") {
    _T.resize_as_(*args.begin());
    _T.copy_(*args.begin());
    _U.resize_as_(*(args.begin() + 1));
    _U.copy_(*(args.begin() + 1));
    _S.resize_as_(*(args.begin() + 2));
    _S.copy_(*(args.begin() + 2));
    _F = _U - _T * _S;
    return _F;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

torch::Tensor ThermoYImpl::forward(torch::Tensor rho, torch::Tensor intEng,
                                   torch::Tensor yfrac) {
  auto yfrac0 = yfrac.clone();
  auto ivol = compute("DY->V", {rho, yfrac});

  // initial guess
  auto &temp = buffer("T");
  auto &pres = buffer("P");

  temp = compute("VU->T", {ivol, conc});
  pres = compute("VT->P", {temp, ivol});
  auto conc = ivol * inv_mu;

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
          .add_input(cv0)
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
  pres = compute("TV->P", {temp, ivol});
  return yfrac - yfrac0;
}

void ThermoYImpl::_ivol_to_yfrac(torch::Tensor ivol, torch::Tensor &out) const {
  int ny = ivol.size(-1) - 1;
  TORCH_CHECK(ny == options.vapor_ids().size() + options.cloud_ids().size(),
              "mass fraction size mismatch");

  auto vec = ivol.sizes().vec();
  for (int i = 0; i < vec.size() - 1; ++i) {
    vec[i + 1] = ivol.size(i);
  }
  vec[0] = ny;

  out = check_resize(out, vec, ivol.options());

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
  TORCH_CHECK(ny == options.vapor_ids().size() + options.cloud_ids().size(),
              "mass fraction size mismatch");

  auto vec = yfrac.sizes().vec();
  for (int i = 0; i < vec.size() - 1; ++i) {
    vec[i] = yfrac.size(i + 1);
  }
  vec.back() = ny + 1;

  out = check_resize(out, vec, yfrac.options());

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
  TORCH_CHECK(ny == options.vapor_ids().size() + options.cloud_ids().size(),
              "mass fraction size mismatch");

  auto vec = yfrac.sizes().vec();
  vec.erase(vec.begin());
  vec.push_back(1 + ny);

  out = check_resize(out, vec, yfrac.options());

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
  int ngas = 1 + options.vapor_ids().size();
  // kg/m^3 -> mol/m^3
  auto conc = ivol * inv_mu;
  auto cz = eval_czh(temp, conc, options).narrow(-1, 0, ngas);
  out = pres / ((cz * conc.narrow(-1, 0, ngas)).sum(-1) * constants::Rgas);
}

void ThermoYImpl::_cv_vol(torch::Tensor ivol, torch::Tensor temp,
                          torch::Tensor &out) const {
  // kg/m^3 -> mol/m^3
  auto conc = ivol * inv_mu;
  auto cv = eval_cv_R(temp, conc, options) * constants::Rgas;
  out = (cv * conc).sum(-1);
}

void ThermoYImpl::_temp_to_intEng(torch::Tensor ivol, torch::Tensor temp,
                                  torch::Tensor &out) const {
  // kg/m^3 -> mol/m^3
  auto conc = ivol * inv_mu;
  auto u = eval_intEng_R(temp, conc, options) * constants::Rgas;
  out = (u * conc).sum(-1);
}

void ThermoYImpl::_intEng_to_temp(torch::Tensor ivol, torch::Tensor intEng,
                                  torch::Tensor &out) const {
  // kg/m^3 -> mol/m^3
  auto u0_sum = (ivol * u0).sum(-1);
  auto cv0_sum = (ivol * cv0).sum(-1);
  auto conc = ivol * inv_mu;

  out = (intEng - u0_sum) / cv0_sum;
  int iter = 0;
  while (iter++ < options.max_iter()) {
    auto u = eval_intEng_R(out, conc, options) * constants::Rgas;
    auto cv = eval_cv_R(out, conc, options) * constants::Rgas;
    auto temp_pre = out.clone();
    out += (intEng - (u * conc).sum(-1)) / (cv * conc).sum(-1);
    if ((out - temp_pre).abs().max().item<double>() < options.ftol()) {
      break;
    }
  }

  if (iter >= options.max_iter()) {
    TORCH_WARN("ThermoYImpl::_intEng_to_temp: max iterations reached");
  }
}

void _temp_to_pres(torch::Tensor ivol, torch::Tensor temp,
                   torch::Tensor &out) const {
  int ngas = 1 + options.vapor_ids().size();

  // kg/m^3 -> mol/m^3
  auto conc = ivol * inv_mu;
  auto cz = eval_czh(temp, conc, options).narrow(-1, 0, ngas);
  out = constants::Rgas * temp * (cz * conc.narrow(-1, 0, ngas)).sum(-1);
}

}  // namespace kintera
