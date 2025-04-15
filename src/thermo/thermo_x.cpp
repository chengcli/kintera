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

  if (options.cv_R().empty()) {
    options.cv_R() = std::vector<double>(nvapor + ncloud, 0.);
  }

  if (options.cp_R().empty()) {
    options.cp_R() = std::vector<double>(nvapor + ncloud, 0.);
  }

  if (options.h0_R().empty()) {
    options.h0_R() = std::vector<double>(1 + nvapor + ncloud, 0.);
  }

  reset();
}

void ThermoXImpl::reset() {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  TORCH_CHECK(options.mu_ratio().size() == nvapor + ncloud,
              "mu_ratio size mismatch");
  TORCH_CHECK(options.cv_R().size() == nvapor + ncloud, "cv_R size mismatch");
  TORCH_CHECK(options.cp_R().size() == nvapor + ncloud, "cp_R size mismatch");
  TORCH_CHECK(options.h0_R().size() == 1 + nvapor + ncloud, "h0 size mismatch");

  mu_ratio_m1 = register_buffer(
      "mu_ratio_m1", torch::tensor(options.mu_ratio(), torch::kFloat64));
  mu_ratio_m1 -= 1.;

  cv_ratio_m1 = register_buffer("cv_ratio_m1",
                                torch::tensor(options.cv_R(), torch::kFloat64));

  // J/mol/K
  cv_ratio_m1 = cv_ratio_m1 / cv_ratio_m1[0] - 1;

  cp_ratio_m1 = register_buffer("cp_ratio_m1",
                                torch::tensor(options.cp_R(), torch::kFloat64));

  // J/mol/K
  cp_ratio_m1 = cp_ratio_m1 / cp_ratio_m1[0] - 1;

  h0_R =
      register_buffer("h0_R", torch::tensor(options.h0_R(), torch::kFloat64));

  // options.cond().species(options.species());
  pcond = register_module("cond", Condensation(options.cond()));
  // options.cond() = pcond->options;
}

torch::Tensor ThermoXImpl::get_mu() const {
  int nmass = options.vapor_ids().size() + options.cloud_ids().size();

  auto result = torch::ones({1 + nmass}, mu_ratio_m1.options());
  result[0] = constants::Rgas / options.Rd();
  result.narrow(0, 1, nmass) = result[0] / (mu_ratio_m1 + 1.);

  return result;
}

torch::Tensor ThermoXImpl::get_cv() const {
  int nmass = options.vapor_ids().size() + options.cloud_ids().size();

  auto result = torch::empty({1 + nmass}, cv_ratio_m1.options());
  result[0] = constants::Rgas / (options.gammad() - 1.);
  result.narrow(0, 1, nmass) = result[0] * (cv_ratio_m1 + 1.);

  return result;
}

torch::Tensor ThermoXImpl::get_cp() const {
  int nmass = options.vapor_ids().size() + options.cloud_ids().size();

  auto result = torch::empty({1 + nmass}, cp_ratio_m1.options());
  result[0] = options.gammad() / (options.gammad() - 1.) * constants::Rgas;
  result.narrow(0, 1, nmass) = result[0] * (cp_ratio_m1 + 1.);

  return result;
}

torch::Tensor ThermoXImpl::get_mass_fraction(torch::Tensor xfrac) const {
  int nmass = xfrac.size(-1) - 1;
  TORCH_CHECK(nmass == options.vapor_ids().size() + options.cloud_ids().size(),
              "mole fraction size mismatch");

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

  yfrac.permute(vec) = xfrac.narrow(-1, 1, nmass) / (mu_ratio_m1 + 1.);
  auto sum =
      1. - xfrac.narrow(-1, 1, nmass).matmul(mu_ratio_m1 / (mu_ratio_m1 + 1.));
  return yfrac / sum.unsqueeze(0);
}

torch::Tensor ThermoXImpl::forward(torch::Tensor temp, torch::Tensor pres,
                                   torch::Tensor xfrac) {
  auto xfrac1 = xfrac.clone();
  int nvapor = options.vapor_ids().size();

  int iter = 0;
  for (; iter < options.max_iter(); ++iter) {
    auto rates = pcond->equilibrate_tp(temp, pres, xfrac1, 1 + nvapor);
    xfrac1 += rates;

    if ((rates / (xfrac1 + 1.e-10)).max().item<double>() < options.rtol())
      break;
    TORCH_CHECK(xfrac1.min().item<double>() >= 0., "negative mole fraction");
  }

  return xfrac1 - xfrac;
}

}  // namespace kintera
