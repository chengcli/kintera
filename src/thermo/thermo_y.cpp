// kintera
#include <kintera/constants.h>

#include "thermo.hpp"

namespace kintera {

ThermoYImpl::ThermoYImpl(const ThermoOptions& options_) : options(options_) {
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  if (options.mu_ratio().empty()) {
    options.mu_ratio() = std::vector<double>(nvapor + ncloud, 0.);
  }

  if (options.cv_R().empty()) {
    options.cv_R() = std::vector<double>(nvapor + ncloud, 0.);
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

  // J/mol/K -> J/kg/K
  auto cv_R = torch::tensor(options.cv_R(), torch::kFloat64);
  cv_ratio_m1 =
      register_buffer("cv_ratio_m1", cv_R / cv_R[0] * (mu_ratio_m1 + 1.));
  cv_ratio_m1 -= 1.;

  // J/kg
  u0 =
      register_buffer("u0", (mu_ratio_m1 + 1.) * Rd *
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
  int nvapor = options.vapor_ids().size();
  int ncloud = options.cloud_ids().size();

  // pressure
  auto pres = get_pres(rho, intEng, yfrac);
  std::cout << "pres = " << pres << std::endl;

  // temperature
  auto temp = get_temp(rho, pres, yfrac);
  auto tempa = temp - options.Tref();

  // concentration
  auto conc = get_concentration(rho, yfrac);
  auto krate = torch::ones_like(temp);

  auto cvd_R = 1. / (options.gammad() - 1.);
  auto cv_R =
      torch::tensor(insert_first(cvd_R, options.cv_R()), conc.options());
  auto intEng_RT = torch::zeros_like(conc);

  int iter = 0;
  for (; iter < options.max_iter(); ++iter) {
    std::cout << "iter = " << iter << std::endl;
    // std::cout << "conc = " << conc << std::endl;

    // dry internal energy
    intEng_RT.select(-1, 0) = cvd_R * tempa;

    // vapors
    intEng_RT.narrow(-1, 1, nvapor) =
        (u0_R.narrow(0, 0, nvapor) +
         cv_R.narrow(0, 1, nvapor) * tempa.unsqueeze(-1));

    // clouds
    intEng_RT.narrow(-1, 1 + nvapor, ncloud) =
        (u0_R.narrow(0, nvapor, ncloud) +
         cv_R.narrow(0, 1 + nvapor, ncloud) * tempa.unsqueeze(-1));

    intEng_RT /= temp.unsqueeze(-1);

    // std::cout << "total intEng = "
    //           << (conc * intEng_RT).sum(-1) * constants::Rgas * temp
    //           << std::endl;

    auto rates = pcond->forward(temp, conc, intEng_RT, cv_R, krate);
    // std::cout << "rates = " << rates << std::endl;
    // std::cout << "krate = " << krate << std::endl;

    conc += rates;
    TORCH_CHECK(conc.min().item<double>() >= 0., "negative concentration");

    auto dT = -temp * (rates * intEng_RT).sum(-1) / conc.matmul(cv_R);
    krate = where(dT * krate > 0., 2 * krate, torch::sign(dT));
    krate.clamp_(-options.boost(), options.boost());

    if (dT.abs().max().item<double>() < options.ftol()) break;
    temp += dT;
    // pres = conc.narrow(-1, 0, 1 + nvapor).sum(-1) * constants::Rgas * temp;
    std::cout << "temp = " << temp << std::endl;
  }

  // (..., nmass + 1) -> (nmass, ...)
  auto vec = conc.sizes().vec();
  int ndim = conc.dim();
  vec[0] = ndim - 1;
  for (int i = 0; i < ndim - 1; ++i) vec[i + 1] = i;

  // molecular weights
  auto mu = torch::ones({1 + options.nspecies()}, mu_ratio_m1.options());
  mu[0] = constants::Rgas / options.Rd();
  mu.narrow(0, 1, options.nspecies()) = mu[0] / (mu_ratio_m1 + 1.);

  auto yfrac1 = (conc * mu).narrow(-1, 1, nvapor + ncloud).permute(vec);
  return yfrac1 - yfrac;
}

torch::Tensor ThermoYImpl::get_concentration(torch::Tensor rho,
                                             torch::Tensor yfrac) const {
  auto nvapor = options.vapor_ids().size();
  auto ncloud = options.cloud_ids().size();

  auto vec = yfrac.sizes().vec();
  vec.erase(vec.begin());
  vec.push_back(1 + nvapor + ncloud);

  auto result = torch::empty(vec, yfrac.options());

  // (nmass, ...) -> (..., nmass + 1)
  int ndim = yfrac.dim();
  for (int i = 0; i < ndim - 1; ++i) {
    vec[i] = i + 1;
  }
  vec[ndim - 1] = 0;

  auto rhod = rho * (1. - yfrac.sum(0));
  result.select(-1, 0) = rhod;
  result.narrow(-1, 1, nvapor + ncloud) =
      rho.unsqueeze(-1) * yfrac.permute(vec) * (mu_ratio_m1 + 1.);
  return result / (constants::Rgas / options.Rd());
}

torch::Tensor ThermoYImpl::get_intEng(torch::Tensor rho, torch::Tensor pres,
                                      torch::Tensor yfrac) const {
  auto vec = yfrac.sizes().vec();
  for (int n = 1; n < vec.size(); ++n) vec[n] = 1;

  auto cv_R = torch::tensor(options.cv_R(), rho.options());
  auto u0 = (u0_R - cv_R * options.Tref()) * (mu_ratio_m1 + 1.);
  auto yu0 = options.Rd() * rho * (yfrac * u0.view(vec)).sum(0);
  return yu0 + pres * f_sig(yfrac) / f_eps(yfrac) / (options.gammad() - 1.);
}

torch::Tensor ThermoYImpl::get_pres(torch::Tensor rho, torch::Tensor intEng,
                                    torch::Tensor yfrac) const {
  auto vec = yfrac.sizes().vec();
  for (int n = 1; n < vec.size(); ++n) vec[n] = 1;

  auto cv_R = torch::tensor(options.cv_R(), rho.options());
  auto u0 = (u0_R - cv_R * options.Tref()) * (mu_ratio_m1 + 1.);
  auto yu0 = options.Rd() * rho * (yfrac * u0.view(vec)).sum(0);
  return (options.gammad() - 1.) * (intEng - yu0) * f_eps(yfrac) / f_sig(yfrac);
}

}  // namespace kintera
