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

std::shared_ptr<ThermoYImpl> ThermoYImpl::create(ThermoOptions const &opts,
                                                 torch::nn::Module *p,
                                                 std::string const &name) {
  TORCH_CHECK(p, "[ThermoY] Parent module is null");
  TORCH_CHECK(opts, "[ThermoY] Options pointer is null");
  return p->register_module(name, ThermoY(opts));
}

ThermoYImpl::ThermoYImpl(const ThermoOptions &options_) : options(options_) {
  populate_thermo(options);
  reset();
}

ThermoYImpl::ThermoYImpl(const ThermoOptions &options1,
                         const SpeciesThermo &options2)
    : options(options1) {
  populate_thermo(options1);
  populate_thermo(options2);
  auto merged = merge_thermo(options1, options2);
  static_cast<SpeciesThermoImpl &>(*options) = (*merged);
  reset();
}

void ThermoYImpl::reset() {
  auto species = options->species();
  auto nspecies = species.size();

  if (options->verbose()) {
    std::cout << "[ThermoY] initializing with species: "
              << fmt::format("{}", species) << std::endl;
  }

  check_dimensions(options);

  std::vector<double> mu_vec(nspecies);
  for (int i = 0; i < options->vapor_ids().size(); ++i) {
    mu_vec[i] = species_weights[options->vapor_ids()[i]];
  }
  for (int i = 0; i < options->cloud_ids().size(); ++i) {
    mu_vec[i + options->vapor_ids().size()] =
        species_weights[options->cloud_ids()[i]];
  }
  inv_mu =
      register_buffer("inv_mu", 1. / torch::tensor(mu_vec, torch::kFloat64));

  if (options->verbose()) {
    std::cout << "[ThermoY] species molecular weights (kg/mol): "
              << fmt::format("{}", mu_vec) << std::endl;
  }

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
    std::cout << "[ThermoY] species cref_R (dimensionless): "
              << fmt::format("{}", options->cref_R()) << std::endl;
    std::cout << "[ThermoY] species uref_R (K) at T = 0: "
              << fmt::format("{}", options->uref_R()) << std::endl;
    std::cout << "[ThermoY] species sref_R (dimensionless) at T = 0: "
              << fmt::format("{}", options->sref_R()) << std::endl;
  }

  auto cv_R = torch::tensor(options->cref_R(), torch::kFloat64);
  auto uref_R = torch::tensor(options->uref_R(), torch::kFloat64);

  // J/kg/K
  cv0 = register_buffer("cv0", cv_R * constants::Rgas * inv_mu);

  // J/kg
  u0 = register_buffer("u0", uref_R * constants::Rgas * inv_mu);

  // populate stoichiometry matrix
  auto reactions = options->reactions();
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

  if (options->verbose()) {
    std::cout << "[ThermoY] stoichiometry matrix: " << std::endl;
    std::cout << stoich << std::endl;
  }
}

void ThermoYImpl::pretty_print(std::ostream &os) const {
  os << fmt::format("ThermoY({})", options) << std::endl;
}

torch::Tensor ThermoYImpl::compute(std::string ab,
                                   std::vector<torch::Tensor> const &args) {
  if (ab == "V->Y") {
    auto V = args[0];
    int ny = V.size(-1) - 1;
    TORCH_CHECK(
        ny + 1 == options->vapor_ids().size() + options->cloud_ids().size(),
        "mass fraction size mismatch");

    auto vec = V.sizes().vec();
    for (int i = 0; i < vec.size() - 1; ++i) {
      vec[i + 1] = V.size(i);
    }
    vec[0] = ny;

    auto Y = torch::empty(vec, V.options());
    _ivol_to_yfrac(V, Y);
    return Y;
  } else if (ab == "Y->X") {
    auto Y = args[0];
    int ny = Y.size(0);
    TORCH_CHECK(
        ny + 1 == options->vapor_ids().size() + options->cloud_ids().size(),
        "mass fraction size mismatch");

    auto vec = Y.sizes().vec();
    for (int i = 0; i < vec.size() - 1; ++i) {
      vec[i] = Y.size(i + 1);
    }
    vec.back() = ny + 1;

    auto X = torch::empty(vec, Y.options());
    _yfrac_to_xfrac(Y, X);
    return X;
  } else if (ab == "DY->V") {
    auto D = args[0];
    auto Y = args[1];

    int ny = Y.size(0);
    TORCH_CHECK(
        ny + 1 == options->vapor_ids().size() + options->cloud_ids().size(),
        "mass fraction size mismatch");

    auto vec = Y.sizes().vec();
    vec.erase(vec.begin());
    vec.push_back(1 + ny);

    auto V = torch::empty(vec, Y.options());
    _yfrac_to_ivol(D, Y, V);
    return V;
  } else if (ab == "PV->T") {
    auto P = args[0];
    auto V = args[1];
    auto T = torch::empty_like(P);
    _pres_to_temp(P, V, T);
    return T;
  } else if (ab == "VT->cv") {
    auto V = args[0];
    auto T = args[1];
    auto cv = torch::empty_like(T);
    _cv_vol(V, T, cv);
    return cv;
  } else if (ab == "VT->U") {
    auto V = args[0];
    auto T = args[1];
    auto U = torch::empty_like(T);
    _intEng_vol(V, T, U);
    return U;
  } else if (ab == "VU->T") {
    auto V = args[0];
    auto U = args[1];
    auto T = torch::empty_like(U);
    _intEng_to_temp(V, U, T);
    return T;
  } else if (ab == "VT->P") {
    auto V = args[0];
    auto T = args[1];
    auto P = torch::empty_like(T);
    _temp_to_pres(V, T, P);
    return P;
  } else if (ab == "PVT->S") {
    auto P = args[0];
    auto V = args[1];
    auto T = args[2];
    auto S = torch::empty_like(T);
    _entropy_vol(P, V, T, S);
    return S;
  } else if (ab == "TUS->F") {
    auto T = args[0];
    auto U = args[1];
    auto S = args[2];
    return U - T * S;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

torch::Tensor ThermoYImpl::forward(torch::Tensor rho, torch::Tensor intEng,
                                   torch::Tensor &yfrac, bool warm_start,
                                   torch::optional<torch::Tensor> diag) {
  if (options->reactions().size() == 0) {  // no-op
    return torch::Tensor();
  }

  auto yfrac0 = yfrac.clone();
  auto ivol = compute("DY->V", {rho, yfrac});
  auto vec = ivol.sizes().vec();
  auto reactions = options->reactions();

  // |reactions| x |reactions| weight matrix
  vec[ivol.dim() - 1] = reactions.size() * reactions.size();
  auto gain = torch::empty(vec, ivol.options());

  if (!warm_start || !reaction_set.defined()) {
    auto vec2 = rho.sizes().vec();
    vec2.push_back(reactions.size());
    reaction_set = torch::arange(0, (int)reactions.size(),
                                 rho.options().dtype(torch::kInt));
    for (int i = 0; i < rho.dim(); ++i)
      reaction_set = reaction_set.unsqueeze(0);
    reaction_set = reaction_set.expand(vec2).contiguous();
    nactive = torch::zeros_like(rho, rho.options().dtype(torch::kInt));
  }

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
          .add_input(reaction_set)
          .add_owned_input(nactive.unsqueeze(-1))
          .build();

  // call the equilibrium solver
  at::native::call_equilibrate_uv(
      conc.device().type(), iter, options->vapor_ids().size(), stoich,
      u0 / inv_mu,   // J/kg -> J/mol*/
      cv0 / inv_mu,  // J/(kg K) -> J/(mol K)*/
      options->nucleation()->logsvp(), options->intEng_R_extra(),
      options->ftol(), options->max_iter());

  ivol = conc / inv_mu;
  yfrac = compute("V->Y", {ivol});

  vec[ivol.dim() - 1] = reactions.size();
  vec.push_back(reactions.size());
  return gain.view(vec);
}

void ThermoYImpl::_ivol_to_yfrac(torch::Tensor ivol, torch::Tensor &out) const {
  int ny = ivol.size(-1) - 1;
  auto vec = ivol.sizes().vec();
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
  auto vec = yfrac.sizes().vec();
  // (ny, ...) -> (..., ny + 1)
  int ndim = yfrac.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  auto mud = species_weights[options->vapor_ids()[0]];
  out.narrow(-1, 1, ny) = yfrac.permute(vec) * inv_mu.narrow(0, 1, ny) * mud;

  auto sum = 1. + yfrac.permute(vec).matmul(mud * inv_mu.narrow(0, 1, ny) - 1.);
  out.narrow(-1, 1, ny) /= sum.unsqueeze(-1);
  out.select(-1, 0) = 1. - out.narrow(-1, 1, ny).sum(-1);
}

void ThermoYImpl::_yfrac_to_ivol(torch::Tensor rho, torch::Tensor yfrac,
                                 torch::Tensor &out) const {
  int ny = yfrac.size(0);
  auto vec = yfrac.sizes().vec();
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
  int ngas = options->vapor_ids().size();

  // kg/m^3 -> mol/m^3
  auto conc_gas =
      (ivol * inv_mu).narrow(-1, 0, ngas).clamp_min(options->gas_floor());

  out.set_(pres / (conc_gas.sum(-1) * constants::Rgas));
  int iter = 0;
  while (iter++ < options->max_iter()) {
    auto cz = eval_czh(out, conc_gas, options);
    auto func = out * (cz * conc_gas).sum(-1) - pres / constants::Rgas;
    auto cv_R = eval_cv_R(out, conc_gas, options);
    auto cp_R = eval_cp_R(out, conc_gas, options);
    auto temp_pre = out.clone();
    out += func / ((cp_R - cv_R) * conc_gas).sum(-1);
    if ((1. - temp_pre / out).abs().max().item<double>() < options->ftol()) {
      break;
    }
  }

  if (iter >= options->max_iter()) {
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
  while (iter++ < options->max_iter()) {
    auto u = eval_intEng_R(out, conc, options) * constants::Rgas;
    auto cv = eval_cv_R(out, conc, options) * constants::Rgas;
    auto temp_pre = out.clone();
    out += (intEng - (u * conc).sum(-1)) / (cv * conc).sum(-1);
    if ((1. - temp_pre / out).abs().max().item<double>() < options->ftol()) {
      break;
    }
  }

  if (iter >= options->max_iter()) {
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
  int ngas = options->vapor_ids().size();

  // kg/m^3 -> mol/m^3
  auto conc_gas = (ivol * inv_mu).narrow(-1, 0, ngas);
  auto cz = eval_czh(temp, conc_gas, options);
  out.set_(constants::Rgas * temp * (cz * conc_gas).sum(-1));
}

}  // namespace kintera
