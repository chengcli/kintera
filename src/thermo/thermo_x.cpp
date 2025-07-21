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

ThermoXImpl::ThermoXImpl(const ThermoOptions &options_) : options(options_) {
  populate_thermo(options);
  reset();
}

ThermoXImpl::ThermoXImpl(const ThermoOptions &options1_,
                         const SpeciesThermo &options2_)
    : options(options1_) {
  auto options2 = options2_;
  populate_thermo(options);
  populate_thermo(options2);
  static_cast<SpeciesThermo &>(options) = merge_thermo(options, options2);
  reset();
}

void ThermoXImpl::reset() {
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
  mu = register_buffer("mu", torch::tensor(mu_vec, torch::kFloat64));

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
  _T = register_buffer("T", torch::empty({0}, torch::kFloat64));
  _P = register_buffer("P", torch::empty({0}, torch::kFloat64));
  _X = register_buffer("X", torch::empty({0}, torch::kFloat64));
  _Y = register_buffer("Y", torch::empty({0}, torch::kFloat64));
  _V = register_buffer("V", torch::empty({0}, torch::kFloat64));
  _D = register_buffer("D", torch::empty({0}, torch::kFloat64));
  _H = register_buffer("H", torch::empty({0}, torch::kFloat64));
  _S = register_buffer("S", torch::empty({0}, torch::kFloat64));
  _G = register_buffer("G", torch::empty({0}, torch::kFloat64));
  _cp = register_buffer("cp", torch::empty({0}, torch::kFloat64));
}

void ThermoXImpl::pretty_print(std::ostream &os) const {
  os << fmt::format("ThermoX({})", options) << std::endl;
}

torch::Tensor const &ThermoXImpl::compute(
    std::string ab, std::vector<torch::Tensor> const &args) {
  if (ab == "X->Y") {
    _X.set_(args[0]);
    _xfrac_to_yfrac(_X, _Y);
    return _Y;
  } else if (ab == "V->D") {
    _V.set_(args[0]);
    _conc_to_dens(_V, _D);
    return _D;
  } else if (ab == "TV->cp") {
    _T.set_(args[0]);
    _V.set_(args[1]);
    _cp_vol(_T, _V, _cp);
    return _cp;
  } else if (ab == "TV->H") {
    _T.set_(args[0]);
    _V.set_(args[1]);
    _enthalpy_vol(_T, _V, _H);
    return _H;
  } else if (ab == "TPX->V") {
    _T.set_(args[0]);
    _P.set_(args[1]);
    _X.set_(args[2]);
    _xfrac_to_conc(_T, _P, _X, _V);
    return _V;
  } else if (ab == "TPV->S") {
    _T.set_(args[0]);
    _P.set_(args[1]);
    _V.set_(args[2]);
    _entropy_vol(_T, _P, _V, _S);
    return _S;
  } else if (ab == "PXS->T") {
    _P.set_(args[0]);
    _X.set_(args[1]);
    _S.set_(args[2]);
    _entropy_to_temp(_P, _X, _S, _T);
    return _T;
  } else if (ab == "THS->G") {
    _T.set_(args[0]);
    _H.set_(args[1]);
    _S.set_(args[2]);
    _G.set_(_H - _T * _S);
    return _G;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

torch::Tensor ThermoXImpl::forward(torch::Tensor temp, torch::Tensor pres,
                                   torch::Tensor &xfrac,
                                   torch::optional<torch::Tensor> diag) {
  if (options.reactions().size() == 0) {  // no-op
    return torch::Tensor();
  }

  auto xfrac0 = xfrac.clone();
  auto vec = xfrac.sizes().vec();
  auto reactions = options.reactions();

  // |reactions| x |reactions| weight matrix
  vec[xfrac.dim() - 1] = reactions.size() * reactions.size();
  auto gain = torch::zeros(vec, xfrac.options());

  // diagnostic array
  vec[xfrac.dim() - 1] = 1;
  if (!diag.has_value()) {
    diag = torch::empty(vec, xfrac.options());
  }

  // prepare data
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(false)
                  .declare_static_shape(xfrac.sizes(),
                                        /*squash_dims=*/{xfrac.dim() - 1})
                  .add_output(gain)
                  .add_output(diag.value())
                  .add_output(xfrac)
                  .add_owned_input(temp.unsqueeze(-1))
                  .add_owned_input(pres.unsqueeze(-1))
                  .add_owned_input(stoich)
                  .build();

  // call the equilibrium solver
  at::native::call_equilibrate_tp(
      xfrac.device().type(), iter, options.vapor_ids().size(),
      options.nucleation().logsvp().data(), options.ftol(), options.max_iter());

  vec[xfrac.dim() - 1] = reactions.size();
  vec.push_back(reactions.size());
  return gain.view(vec);
}

void ThermoXImpl::_xfrac_to_yfrac(torch::Tensor xfrac,
                                  torch::Tensor &out) const {
  int ny = xfrac.size(-1) - 1;

  auto vec = xfrac.sizes().vec();
  for (int i = 0; i < vec.size() - 1; ++i) {
    vec[i + 1] = xfrac.size(i);
  }
  vec[0] = ny;

  out.set_(check_resize(out, vec, xfrac.options()));

  // (..., ny + 1) -> (ny, ...)
  int ndim = xfrac.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  out.permute(vec) = xfrac.narrow(-1, 1, ny) * mu.narrow(0, 1, ny);
  out /= (xfrac * mu).sum(-1).unsqueeze(0);
}

void ThermoXImpl::_entropy_to_temp(torch::Tensor pres, torch::Tensor xfrac,
                                   torch::Tensor entropy, torch::Tensor &out) {
  // check if cached temperature is available
  if (out.sizes() != pres.sizes() || out.device() != pres.device() ||
      out.dtype() != pres.dtype()) {
    out.set_(options.Tref() * torch::ones_like(pres));
  }

  int iter = 0;
  while (iter++ < options.max_iter()) {
    auto conc = compute("TPX->V", {out, pres, xfrac});
    auto cp_vol = compute("TV->cp", {out, conc});
    auto entropy_vol = compute("TPV->S", {out, pres, conc});
    auto temp_pre = out.clone();
    out *= 1. + (entropy - entropy_vol) / cp_vol;
    forward(out, pres, xfrac);
    if ((1. - temp_pre / out).abs().max().item<double>() < options.ftol()) {
      break;
    }
  }

  if (iter >= options.max_iter()) {
    TORCH_WARN("ThermoX:_entropy_to_temp: max iteration reached");
    // get a time stamp (string) to dump diagnostic data
    auto time_stamp = std::to_string(std::time(nullptr));

    // save torch tensor data to file with time stamp
    auto filename = "thermo_x_entropy_to_temp_" + time_stamp + ".pt";

    std::map<std::string, torch::Tensor> data;
    data["pres"] = pres;
    data["xfrac"] = xfrac;
    data["entropy"] = entropy;
    save_tensors(data, filename);
  }
}

void ThermoXImpl::_xfrac_to_conc(torch::Tensor temp, torch::Tensor pres,
                                 torch::Tensor xfrac,
                                 torch::Tensor &out) const {
  int ngas = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  auto xgas = xfrac.narrow(-1, 0, ngas).sum(-1, /*keepdim=*/true);
  auto ideal_gas_conc = xfrac.narrow(-1, 0, ngas) * pres.unsqueeze(-1) /
                        (temp.unsqueeze(-1) * constants::Rgas * xgas);

  auto conc_gas = ideal_gas_conc.clone();
  int iter = 0;
  while (iter++ < options.max_iter()) {
    auto cz = eval_czh(temp, conc_gas, options);
    auto cz_ddC = eval_czh_ddC(temp, conc_gas, options);
    auto conc_gas_pre = conc_gas.clone();
    conc_gas += (ideal_gas_conc - cz * conc_gas) / (cz_ddC * conc_gas + cz);
    if ((1. - conc_gas_pre / conc_gas).abs().max().item<double>() <
        options.ftol()) {
      break;
    }
  }

  if (iter >= options.max_iter()) {
    TORCH_WARN("ThermoX:_xfrac_to_conc: max iteration reached");

    // get a time stamp (string) to dump diagnostic data
    auto time_stamp = std::to_string(std::time(nullptr));

    // save torch tensor data to file with time stamp
    auto filename = "thermo_x_xfrac_to_conc_" + time_stamp + ".pt";

    std::map<std::string, torch::Tensor> data;
    data["temp"] = temp;
    data["pres"] = pres;
    data["xfrac"] = xfrac;

    save_tensors(data, filename);
  }

  out.set_(check_resize(out, xfrac.sizes(), xfrac.options()));

  out.narrow(-1, 0, ngas) = conc_gas;
  out.narrow(-1, ngas, ncloud) = conc_gas.select(-1, 0).unsqueeze(-1) *
                                 xfrac.narrow(-1, ngas, ncloud) /
                                 xfrac.select(-1, 0).unsqueeze(-1);
}

}  // namespace kintera
