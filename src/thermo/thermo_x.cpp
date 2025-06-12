// kintera
#include <kintera/constants.h>

#include <kintera/utils/check_resize.hpp>

#include "eval_uh.hpp"
#include "thermo.hpp"
#include "thermo_dispatch.hpp"

namespace kintera {

ThermoXImpl::ThermoXImpl(const ThermoOptions &options_) : options(options_) {
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

void ThermoXImpl::reset() {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  TORCH_CHECK(options.mu_ratio().size() == 1 + nvapor + ncloud,
              "mu_ratio size  = ", options.mu_ratio().size(),
              ". Expected =  ", 1 + nvapor + ncloud);

  // restrict cref and uref
  options.cref_R().resize(1 + nvapor + ncloud);
  options.uref_R().resize(1 + nvapor + ncloud);

  auto mud = constants::Rgas / options.Rd();
  mu = register_buffer(
      "mu", mud * torch::tensor(options.mu_ratio(), torch::kFloat64));

  // change offset to T = 0
  for (int i = 0; i < options.uref_R().size(); ++i) {
    options.uref_R()[i] -= options.cref_R()[i] * options.Tref();
  }

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
  _T = register_buffer("T", torch::empty({0}));
  _P = register_buffer("P", torch::empty({0}));
  _X = register_buffer("X", torch::empty({0}));
  _Y = register_buffer("Y", torch::empty({0}));
  _V = register_buffer("V", torch::empty({0}));
  _D = register_buffer("D", torch::empty({0}));
  _H = register_buffer("H", torch::empty({0}));
  _S = register_buffer("S", torch::empty({0}));
  _G = register_buffer("G", torch::empty({0}));
  _cp = register_buffer("cp", torch::empty({0}));
}

torch::Tensor const &ThermoXImpl::compute(
    std::string ab, std::initializer_list<torch::Tensor> args) {
  if (ab == "X->Y") {
    _X.set_(*args.begin());
    _xfrac_to_yfrac(_X, _Y);
    return _Y;
  } else if (ab == "V->D") {
    _V.set_(*args.begin());
    _conc_to_dens(_V, _D);
    return _D;
  } else if (ab == "TV->cp") {
    _T.set_(*args.begin());
    _V.set_(*(args.begin() + 1));
    _cp_vol(_T, _V, _cp);
    return _cp;
  } else if (ab == "TV->H") {
    _T.set_(*args.begin());
    _V.set_(*(args.begin() + 1));
    _temp_to_enthalpy(_T, _V, _H);
    return _H;
  } else if (ab == "TPX->V") {
    _T.set_(*args.begin());
    _P.set_(*(args.begin() + 1));
    _X.set_(*(args.begin() + 2));
    _xfrac_to_conc(_T, _P, _X, _V);
    return _V;
  } else if (ab == "TPV->S") {
    // TODO(cli)
    return _S;
  } else if (ab == "THS->G") {
    _T.set_(*args.begin());
    _H.set_(*(args.begin() + 1));
    _S.set_(*(args.begin() + 2));
    _G.set_(_H - _T * _S);
    return _G;
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }
}

torch::Tensor ThermoXImpl::forward(torch::Tensor temp, torch::Tensor pres,
                                   torch::Tensor &xfrac) {
  auto xfrac0 = xfrac.clone();

  // prepare data
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(false)
                  .declare_static_shape(xfrac.sizes(),
                                        /*squash_dims=*/{xfrac.dim() - 1})
                  .add_output(xfrac)
                  .add_owned_input(temp.unsqueeze(-1))
                  .add_owned_input(pres.unsqueeze(-1))
                  .add_owned_input(stoich)
                  .build();

  // prepare svp function
  user_func1 *logsvp_func = new user_func1[options.react().size()];
  for (int i = 0; i < options.react().size(); ++i) {
    logsvp_func[i] = options.react()[i].func();
  }

  // call the equilibrium solver
  at::native::call_equilibrate_tp(xfrac.device().type(), iter,
                                  options.vapor_ids().size() + 1, logsvp_func,
                                  options.ftol(), options.max_iter());

  delete[] logsvp_func;

  return xfrac - xfrac0;
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

void ThermoXImpl::_xfrac_to_conc(torch::Tensor temp, torch::Tensor pres,
                                 torch::Tensor xfrac,
                                 torch::Tensor &out) const {
  int ngas = 1 + options.vapor_ids().size();
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
  }

  out.set_(check_resize(out, xfrac.sizes(), xfrac.options()));

  out.narrow(-1, 0, ngas) = conc_gas;
  out.narrow(-1, ngas, ncloud) = conc_gas.select(-1, 0).unsqueeze(-1) *
                                 xfrac.narrow(-1, ngas, ncloud) /
                                 xfrac.select(-1, 0).unsqueeze(-1);
}

}  // namespace kintera
