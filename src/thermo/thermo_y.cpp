// kintera
#include <kintera/constants.h>

#include <kintera/utils/check_resize.hpp>
#include <kintera/utils/serialize.hpp>

#include "eval_uhs.hpp"
#include "thermo.hpp"
#include "thermo_dispatch.hpp"
#include "thermo_formatter.hpp"

namespace kintera {

extern std::vector<double> species_weights;

ThermoYImpl::ThermoYImpl(const ThermoOptions &options_) : options(options_) {
  populate_thermo(options);
  reset();
}

ThermoYImpl::ThermoYImpl(const ThermoOptions &options1_,
                         const SpeciesThermo &options2_)
    : options(options1_) {
  auto options2 = options2_;
  populate_thermo(options);
  populate_thermo(options2);
  static_cast<SpeciesThermo &>(options) = merge_thermo(options, options2);
  reset();
}

void ThermoYImpl::reset() {
  auto species = options.species();
  auto nspecies = species.size();

  check_dimensions(options);

  std::vector<double> mu_vec(nspecies);
  for (int i = 0; i < options.vapor_ids().size(); ++i) {
    mu_vec[i] = species_weights[options.vapor_ids()[i]];
  }
  for (int i = 0; i < options.cloud_ids().size(); ++i) {
    mu_vec[i + options.vapor_ids().size()] =
        species_weights[options.cloud_ids()[i]];
  }
  inv_mu =
      register_buffer("inv_mu", 1. / torch::tensor(mu_vec, torch::kFloat64));

  // change internal energy offset to T = 0
  for (int i = 0; i < options.uref_R().size(); ++i) {
    options.uref_R()[i] -= options.cref_R()[i] * options.Tref();
  }

  // change entropy offset to T = 1, P = 1
  for (int i = 0; i < options.vapor_ids().size(); ++i) {
    auto Tref = std::max(options.Tref(), 1.);
    auto Pref = std::max(options.Pref(), 1.);
    options.sref_R()[i] -= (options.cref_R()[i] + 1) * log(Tref) - log(Pref);
  }

  // set cloud entropy offset to 0 (not used)
  for (int i = options.vapor_ids().size(); i < options.sref_R().size(); ++i) {
    options.sref_R()[i] = 0.;
  }

  auto cv_R = torch::tensor(options.cref_R(), torch::kFloat64);
  auto uref_R = torch::tensor(options.uref_R(), torch::kFloat64);

  // J/kg/K
  cv0 = register_buffer("cv0", cv_R * constants::Rgas * inv_mu);

  // J/kg
  u0 = register_buffer("u0", uref_R * constants::Rgas * inv_mu);

  // populate stoichiometry matrix
  auto reactions = options.reactions();
  stoich = register_buffer(
      "stoich",
      torch::zeros({(int)nspecies, (int)reactions.size()}, torch::kFloat64));

  for (int j = 0; j < reactions.size(); ++j) {
    auto const &r = reactions[j];
    for (int i = 0; i < nspecies; ++i) {
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

  // populate buffers
  _D = register_buffer("D", torch::empty({0}, torch::kFloat64));
  _P = register_buffer("P", torch::empty({0}, torch::kFloat64));
  _Y = register_buffer("Y", torch::empty({0}, torch::kFloat64));
  _X = register_buffer("X", torch::empty({0}, torch::kFloat64));
  _V = register_buffer("V", torch::empty({0}, torch::kFloat64));
  _T = register_buffer("T", torch::empty({0}, torch::kFloat64));
  _U = register_buffer("U", torch::empty({0}, torch::kFloat64));
  _S = register_buffer("S", torch::empty({0}, torch::kFloat64));
  _F = register_buffer("F", torch::empty({0}, torch::kFloat64));
  _cv = register_buffer("cv", torch::empty({0}, torch::kFloat64));
}

void ThermoYImpl::pretty_print(std::ostream &os) const {
  os << fmt::format("ThermoY({})", options) << std::endl;
}

torch::Tensor const &ThermoYImpl::compute(
    std::string ab, std::vector<torch::Tensor> const &args) {
  if (ab == "V->Y") {
    _V.set_(args[0]);
    _ivol_to_yfrac(_V, _Y);
    return _Y;
  } else if (ab == "Y->X") {
    _Y.set_(args[0]);
    _yfrac_to_xfrac(_Y, _X);
    return _X;
  } else if (ab == "DY->V") {
    _D.set_(args[0]);
    _Y.set_(args[1]);
    _yfrac_to_ivol(_D, _Y, _V);
    return _V;
  } else if (ab == "PV->T") {
    _P.set_(args[0]);
    _V.set_(args[1]);
    _pres_to_temp(_P, _V, _T);
    return _T;
  } else if (ab == "VT->cv") {
    _V.set_(args[0]);
    _T.set_(args[1]);
    _cv_vol(_V, _T, _cv);
    return _cv;
  } else if (ab == "VT->U") {
    _V.set_(args[0]);
    _T.set_(args[1]);
    _intEng_vol(_V, _T, _U);
    return _U;
  } else if (ab == "VU->T") {
    _V.set_(args[0]);
    _U.set_(args[1]);
    _intEng_to_temp(_V, _U, _T);
    return _T;
  } else if (ab == "VT->P") {
    _V.set_(args[0]);
    _T.set_(args[1]);
    _temp_to_pres(_V, _T, _P);
    return _P;
  } else if (ab == "PVT->S") {
    _P.set_(args[0]);
    _V.set_(args[1]);
    _T.set_(args[2]);
    _entropy_vol(_P, _V, _T, _S);
    return _S;
  } else if (ab == "TUS->F") {
    _T.set_(args[0]);
    _U.set_(args[1]);
    _S.set_(args[2]);
    _F.set_(_U - _T * _S);
    return _F;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

torch::Tensor ThermoYImpl::forward(torch::Tensor rho, torch::Tensor intEng,
                                   torch::Tensor &yfrac,
                                   torch::optional<torch::Tensor> diag) {
  if (options.reactions().size() == 0) {  // no-op
    return torch::Tensor();
  }

  auto yfrac0 = yfrac.clone();
  auto ivol = compute("DY->V", {rho, yfrac});
  auto vec = ivol.sizes().vec();
  auto reactions = options.reactions();

  // |reactions| x |reactions| weight matrix
  vec[ivol.dim() - 1] = reactions.size() * reactions.size();
  auto gain = torch::empty(vec, ivol.options());

  // diagnostic array
  vec[ivol.dim() - 1] = 1;
  if (!diag.has_value()) {
    diag = torch::zeros(vec, ivol.options());
  }

  // initial guess
  auto temp = compute("VU->T", {ivol, intEng});
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

  // call the equilibrium solver
  at::native::call_equilibrate_uv(
      conc.device().type(), iter, options.nucleation().logsvp().data(),
      options.nucleation().logsvp_ddT().data(), options.intEng_R_extra().data(),
      options.cv_R_extra().data(), options.ftol(), options.max_iter());

  ivol = conc / inv_mu;
  yfrac = compute("V->Y", {ivol});

  vec[ivol.dim() - 1] = reactions.size();
  vec.push_back(reactions.size());
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

  auto mud = species_weights[options.vapor_ids()[0]];
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

    // get a time stamp (string) to dump diagnostic data
    auto time_stamp = std::to_string(std::time(nullptr));

    // save torch tensor data to file with time stamp
    auto filename = "thermo_y_pres_to_temp_" + time_stamp + ".pt";

    std::map<std::string, torch::Tensor> data;
    data["pres"] = pres;
    data["ivol"] = ivol;
    save_tensors(data, filename);
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

    // get a time stamp (string) to dump diagnostic data
    auto time_stamp = std::to_string(std::time(nullptr));

    // save torch tensor data to file with time stamp
    auto filename = "thermo_y_intEng_to_temp_" + time_stamp + ".pt";

    std::map<std::string, torch::Tensor> data;
    data["ivol"] = ivol;
    data["intEng"] = intEng;
    save_tensors(data, filename);
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
