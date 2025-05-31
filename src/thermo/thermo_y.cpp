// kintera
#include <kintera/constants.h>

#include "thermo.hpp"

namespace kintera {

ThermoYImpl::ThermoYImpl(const ThermoOptions& options_) : options(options_) {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  if (options.mu_ratio().empty()) {
    options.mu_ratio() = std::vector<double>(nvapor + ncloud, 1.);
  }

  if (options.cv_R().empty()) {
    options.cv_R() = std::vector<double>(nvapor + ncloud, 5./2.);
  }

  if (options.u0_R().empty()) {
    options.u0_R() = std::vector<double>(nvapor + ncloud, 0.);
  }

  reset();
}

void ThermoYImpl::reset() {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  TORCH_CHECK(options.mu_ratio().size() == nvapor + ncloud,
              "mu_ratio size mismatch");
  TORCH_CHECK(options.cv_R().size() == nvapor + ncloud, "cv_R size mismatch");
  TORCH_CHECK(options.u0_R().size() == nvapor + ncloud, "u0_R size mismatch");

  mu_ratio_m1 = register_buffer(
      "mu_ratio_m1", 1. / torch::tensor(options.mu_ratio(), torch::kFloat64));
  mu_ratio_m1 -= 1.;

  cv_ratio_m1 =
      register_buffer("cv_ratio_m1", torch::tensor(options.cv_R(), torch::kFloat64));

  // J/mol/K -> J/kg/K
  cv_ratio_m1 = cv_ratio_m1 * (options.gammad() - 1.) * (mu_ratio_m1 + 1.) - 1;

  // J/kg
  u0 = register_buffer("u0", (mu_ratio_m1 + 1.) * options.Rd() *
                        torch::tensor(options.u0_R(), torch::kFloat64));

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

torch::Tensor ThermoYImpl::get_mole_fraction(torch::Tensor yfrac) const {
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

torch::Tensor ThermoYImpl::forward(torch::Tensor rho, torch::Tensor intEng,
                                   torch::Tensor yfrac) {
}

}  // namespace kintera
