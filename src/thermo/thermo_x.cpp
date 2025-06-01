// kintera
#include "thermo.hpp"

namespace kintera {

void call_equilibrate_tp_cpu(at::TensorIterator &iter, int ngas,
                             user_func1 const *logsvp_func, double logsvp_eps,
                             int max_iter);

ThermoXImpl::ThermoXImpl(const ThermoOptions &options_) : options(options_) {
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

  reset();
}

void ThermoXImpl::reset() {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  TORCH_CHECK(options.mu_ratio().size() == nvapor + ncloud,
              "mu_ratio size mismatch");
  TORCH_CHECK(options.cref_R().size() == nvapor + ncloud,
              "cref_R size mismatch");
  TORCH_CHECK(options.uref_R().size() == nvapor + ncloud,
              "uref_R size mismatch");

  mu_ratio_m1 = register_buffer(
      "mu_ratio_m1", torch::tensor(options.mu_ratio(), torch::kFloat64));
  mu_ratio_m1 -= 1.;

  auto cp_R = torch::tensor(options.cref_R(), torch::kFloat64);
  cp_R.narrow(0, 0, nvapor) += 1.;

  auto href_R = torch::tensor(options.uref_R(), torch::kFloat64);
  href_R.narrow(0, 0, nvapor) += options.Tref();

  // J/mol/K
  cp_ratio_m1 = register_buffer(
      "cp_ratio_m1", cp_R * (options.gammad() - 1.) / options.gammad());
  cp_ratio_m1 -= 1.;

  h0_R = register_buffer("h0_R", href_R - cp_R * options.Tref());

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

torch::Tensor ThermoXImpl::f_psi(torch::Tensor xfrac) const {
  int nmass = xfrac.size(-1) - 1;
  return 1. + xfrac.narrow(-1, 1, nmass).matmul(cp_ratio_m1);
}

torch::Tensor ThermoXImpl::compute(std::string ab,
                                   std::initializer_list<torch::Tensor> args,
                                   torch::optional<torch::Tensor> out) const {
  if (ab == "X->Y") {
    out = _xfrac_to_yfrac(*args.begin());
  } else if (ab == "TPX->D") {
    out =
        _temp_to_dens(*args.begin(), *(args.begin() + 1), *(args.begin() + 2));
  } else {
    TORCH_CHECK(false, "Unknown abbreviation: ", ab);
  }

  return out.value_or(torch::Tensor());
}

torch::Tensor ThermoXImpl::forward(torch::Tensor temp, torch::Tensor pres,
                                   torch::Tensor xfrac) {
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
  if (xfrac.is_cpu()) {
    call_equilibrate_tp_cpu(iter, options.vapor_ids().size() + 1, logsvp_func,
                            options.ftol(), options.max_iter());
  } else if (xfrac.is_cuda()) {
    TORCH_CHECK(false, "CUDA support not implemented yet");
  } else {
    TORCH_CHECK(false, "Unsupported tensor type");
  }

  delete[] logsvp_func;

  return xfrac - xfrac0;
}

torch::Tensor ThermoXImpl::_temp_to_dens(torch::Tensor temp, torch::Tensor pres,
                                         torch::Tensor xfrac) const {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();
  int nmass = nvapor + ncloud;

  auto xgas = 1. - xfrac.narrow(-1, 1 + nvapor, ncloud).sum(-1);
  auto ftv = xgas / (1. + xfrac.narrow(-1, 1, nmass).matmul(mu_ratio_m1));
  return pres / (temp * ftv * options.Rd());
}

torch::Tensor ThermoXImpl::_xfrac_to_yfrac(torch::Tensor xfrac) const {
  int nmass = xfrac.size(-1) - 1;

  auto vec = xfrac.sizes().vec();
  for (int i = 0; i < vec.size() - 1; ++i) {
    vec[i + 1] = xfrac.size(i);
  }
  vec[0] = nmass;

  auto yfrac = torch::empty(vec, xfrac.options());

  // (..., nmass + 1) -> (nmass, ...)
  int ndim = xfrac.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  yfrac.permute(vec) = xfrac.narrow(-1, 1, nmass) * (mu_ratio_m1 + 1.);
  auto sum = 1. + xfrac.narrow(-1, 1, nmass).matmul(mu_ratio_m1);
  return yfrac / sum.unsqueeze(0);
}

}  // namespace kintera
