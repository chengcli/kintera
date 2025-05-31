// kintera
#include <kintera/constants.h>

#include "thermo.hpp"

namespace kintera {

ThermoXImpl::ThermoXImpl(const ThermoOptions& options_) : options(options_) {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  if (options.mu_ratio().empty()) {
    options.mu_ratio() = std::vector<double>(nvapor + ncloud, 0.);
  }

  if (options.cp_R().empty()) {
    options.cp_R() = std::vector<double>(nvapor + ncloud, 0.);
  }

  if (options.u0_R().empty()) {
    options.u0_R() = std::vector<double>(nvapor + ncloud, 0.);
  }

  reset();
}

void ThermoXImpl::reset() {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  TORCH_CHECK(options.mu_ratio().size() == nvapor + ncloud,
              "mu_ratio size mismatch");
  TORCH_CHECK(options.cp_R().size() == nvapor + ncloud, "cp_R size mismatch");
  TORCH_CHECK(options.u0_R().size() == nvapor + ncloud, "u0 size mismatch");

  mu_ratio_m1 = register_buffer(
      "mu_ratio_m1", torch::tensor(options.mu_ratio(), torch::kFloat64));
  mu_ratio_m1 -= 1.;

  // J/mol/K
  cp_ratio_m1 = register_buffer("cp_ratio_m1",
                                torch::tensor(options.cp_R(), torch::kFloat64));

  // J/mol/K
  cp_ratio_m1 = cp_ratio_m1 / cp_ratio_m1[0] - 1;

  // J/mol
  h0 = register_buffer(
      "h0", constants::Rgas * torch::tensor(options.u0_R(), torch::kFloat64));
  h0.narrow(0, 0, vapor_ids.size()) += constants::Rgas * options.Tref();

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

torch::Tensor ThermoXImpl::f_psi(torch::Tensor xfrac) const {
  return 1. + xfrac.narrow(-1, 1, nmass).matmul(cp_ratio_m1);
}

torch::Tensor ThermoXImpl::get_mass_fraction(torch::Tensor xfrac) const {
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

torch::Tensor ThermoXImpl::get_density(torch::Tensor temp, torch::Tensor pres,
                                       torch::Tensor xfrac) const {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();
  int nspecies = nvapor + ncloud;

  auto xgas = 1. - xfrac.narrow(-1, 1 + nvapor, ncloud).sum(-1);
  auto ftv = xgas / (1. + xfrac.narrow(-1, 1, nspecies).matmul(mu_ratio_m1));
  return pres / (temp * ftv * options.Rd());
}

torch::Tensor ThermoXImpl::forward(torch::Tensor temp, torch::Tensor pres,
                                   torch::Tensor xfrac) {
  auto xfrac1 = xfrac.clone();
  int nvapor = options.vapor_ids().size();

  int iter = 0;
  for (; iter < options.max_iter(); ++iter) {
    auto rates = pcond->forward(temp, pres, xfrac1);
    xfrac1 += rates;

    if ((rates / (xfrac1 + 1.e-10)).max().item<double>() < options.rtol())
      break;
    TORCH_CHECK(xfrac1.min().item<double>() >= 0., "negative mole fraction");
  }

  return xfrac1 - xfrac;
}

}  // namespace kintera
