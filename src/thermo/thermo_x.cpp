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

std::shared_ptr<ThermoXImpl> ThermoXImpl::create(ThermoOptions const &opts,
                                                 torch::nn::Module *p,
                                                 std::string const &name) {
  TORCH_CHECK(p, "[ThermoX] Parent module is null");
  TORCH_CHECK(opts, "[ThermoX] Options pointer is null");
  return p->register_module(name, ThermoX(opts));
}

ThermoXImpl::ThermoXImpl(const ThermoOptions &options_) : options(options_) {
  populate_thermo(options);
  reset();
}

ThermoXImpl::ThermoXImpl(const ThermoOptions &options1,
                         const SpeciesThermo &options2)
    : options(options1) {
  populate_thermo(options1);
  populate_thermo(options2);
  auto merged = merge_thermo(options1, options2);
  static_cast<SpeciesThermoImpl &>(*options) = (*merged);
  reset();
}

void ThermoXImpl::reset() {
  auto species = options->species();
  auto nspecies = species.size();

  if (options->verbose()) {
    std::cout << "[ThermoX] initializing with species: "
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
  mu = register_buffer("mu", torch::tensor(mu_vec, torch::kFloat64));

  if (options->verbose()) {
    std::cout << "[ThermoX] species molecular weights (kg/mol): "
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
    std::cout << "[ThermoX] species cref_R (dimensionless): "
              << fmt::format("{}", options->cref_R()) << std::endl;
    std::cout << "[ThermoX] species uref_R (K) at T = 0: "
              << fmt::format("{}", options->uref_R()) << std::endl;
    std::cout << "[ThermoX] species sref_R (dimensionless): "
              << fmt::format("{}", options->sref_R()) << std::endl;
  }

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
    std::cout << "[ThermoX] stoichiometry matrix:\n" << stoich << std::endl;
  }
}

void ThermoXImpl::pretty_print(std::ostream &os) const {
  os << fmt::format("ThermoX({})", options) << std::endl;
}

torch::Tensor ThermoXImpl::compute(std::string ab,
                                   std::vector<torch::Tensor> const &args) {
  if (ab == "X->Y") {
    auto X = args[0];
    int ny = X.size(-1) - 1;
    TORCH_CHECK(
        ny + 1 == options->vapor_ids().size() + options->cloud_ids().size(),
        "mass fraction size mismatch");

    auto vec = X.sizes().vec();
    for (int i = 0; i < vec.size() - 1; ++i) {
      vec[i + 1] = X.size(i);
    }
    vec[0] = ny;

    auto Y = torch::empty(vec, X.options());
    _xfrac_to_yfrac(X, Y);
    return Y;
  } else if (ab == "V->D") {
    auto V = args[0];
    auto D = torch::empty_like(V.select(-1, 0));
    _conc_to_dens(V, D);
    return D;
  } else if (ab == "TV->cp") {
    auto T = args[0];
    auto V = args[1];
    auto cp = torch::empty_like(T);
    _cp_vol(T, V, cp);
    return cp;
  } else if (ab == "TV->H") {
    auto T = args[0];
    auto V = args[1];
    auto H = torch::empty_like(T);
    _enthalpy_vol(T, V, H);
    return H;
  } else if (ab == "TPX->V") {
    auto T = args[0];
    auto P = args[1];
    auto X = args[2];
    auto V = torch::empty_like(X);
    _xfrac_to_conc(T, P, X, V);
    return V;
  } else if (ab == "TPV->S") {
    auto T = args[0];
    auto P = args[1];
    auto V = args[2];
    auto S = torch::empty_like(T);
    _entropy_vol(T, P, V, S);
    return S;
  } else if (ab == "PXS->T") {
    auto P = args[0];
    auto X = args[1];
    auto S = args[2];
    auto T = options->Tref() * torch::ones_like(P);
    _entropy_to_temp(P, X, S, T);
    return T;
  } else if (ab == "THS->G") {
    auto T = args[0];
    auto H = args[1];
    auto S = args[2];
    return H - T * S;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

torch::Tensor ThermoXImpl::forward(torch::Tensor temp, torch::Tensor pres,
                                   torch::Tensor &xfrac, bool warm_start,
                                   torch::optional<torch::Tensor> diag) {
  if (options->reactions().size() == 0) {  // no-op
    return torch::Tensor();
  }

  auto xfrac0 = xfrac.clone();
  auto vec = xfrac.sizes().vec();
  auto reactions = options->reactions();

  // |reactions| x |reactions| weight matrix
  vec[xfrac.dim() - 1] = reactions.size() * reactions.size();
  auto gain = torch::zeros(vec, xfrac.options());

  if (!warm_start || !reaction_set.defined()) {
    auto vec2 = temp.sizes().vec();
    vec2.push_back(reactions.size());
    reaction_set = torch::arange(0, (int)reactions.size(),
                                 temp.options().dtype(torch::kInt));
    for (int i = 0; i < temp.dim(); ++i)
      reaction_set = reaction_set.unsqueeze(0);
    reaction_set = reaction_set.expand(vec2).contiguous();
    nactive = torch::zeros_like(temp, temp.options().dtype(torch::kInt));
  }

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
                  .add_input(reaction_set)
                  .add_owned_input(nactive.unsqueeze(-1))
                  .build();

  // call the equilibrium solver
  at::native::call_equilibrate_tp(
      xfrac.device().type(), iter, options->vapor_ids().size(), stoich,
      options->nucleation()->logsvp(), options->ftol(), options->max_iter());

  vec[xfrac.dim() - 1] = reactions.size();
  vec.push_back(reactions.size());
  return gain.view(vec);
}

void ThermoXImpl::_xfrac_to_yfrac(torch::Tensor xfrac,
                                  torch::Tensor &out) const {
  int ny = xfrac.size(-1) - 1;
  auto vec = xfrac.sizes().vec();
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
  int iter = 0;
  while (iter++ < options->max_iter()) {
    auto conc = compute("TPX->V", {out, pres, xfrac});
    auto cp_vol = compute("TV->cp", {out, conc});
    auto entropy_vol = compute("TPV->S", {out, pres, conc});
    auto temp_pre = out.clone();
    out *= 1. + (entropy - entropy_vol) / cp_vol;
    forward(out, pres, xfrac);
    if ((1. - temp_pre / out).abs().max().item<double>() < options->ftol()) {
      break;
    }
  }

  if (iter >= options->max_iter()) {
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
  int ngas = options->vapor_ids().size();
  int ncloud = options->cloud_ids().size();

  auto xgas = xfrac.narrow(-1, 0, ngas).sum(-1, /*keepdim=*/true);
  auto ideal_gas_conc = xfrac.narrow(-1, 0, ngas) * pres.unsqueeze(-1) /
                        (temp.unsqueeze(-1) * constants::Rgas * xgas);
  ideal_gas_conc.clamp_min_(options->gas_floor());

  auto conc_gas = ideal_gas_conc.clone();

  int iter = 0;
  while (iter++ < options->max_iter()) {
    auto cz = eval_czh(temp, conc_gas, options);
    auto cz_ddC = eval_czh_ddC(temp, conc_gas, options);
    auto conc_gas_pre = conc_gas.clone();
    conc_gas += (ideal_gas_conc - cz * conc_gas) / (cz_ddC * conc_gas + cz);
    if ((1. - conc_gas_pre / conc_gas).abs().max().item<double>() <
        options->ftol()) {
      break;
    }
  }

  if (iter >= options->max_iter()) {
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

  out.narrow(-1, 0, ngas) = conc_gas;
  out.narrow(-1, ngas, ncloud) = conc_gas.select(-1, 0).unsqueeze(-1) *
                                 xfrac.narrow(-1, ngas, ncloud) /
                                 xfrac.select(-1, 0).unsqueeze(-1);
}

}  // namespace kintera
