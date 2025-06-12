#pragma once

// base
#include <configure.h>

namespace kintera {

//! Eq.1 in Li2018
torch::Tensor cal_dlnT_dlnP(torch::Tensor xfrac, torch::Tensor gammad,
                            torch::Tensor cp_ratio, torch::Tensor latent,
                            int nvapor, int ncloud) {
  T x_gas = 1.;
  for (int n = nvapor; n < nvapor + ncloud; ++n) {
    x_gas -= xfrac[n];
  }

  T f_sig = 1.;
  // vapor
  for (int n = 0; n < nvapor; ++n) {
    f_sig += xfrac[n] * (cp_ratio_mole[n] - 1.);
  }
  auto cphat_ov_r = gammad / (gammad - 1.) * f_sig / x_gas;

  // vapor
  auto xd = x_gas;
  for (int n = 0; n < nvapor; ++n) xd -= xfrac[n];

  T c1 = 0., c2 = 0., c3 = 0.;
  for (int n = 0; n < nvapor; ++n) {
    c1 += xfrac[n] / xd * latent[n];
    c2 += xfrac[n] / xd * latent[n] * latent[n];
    c3 += xfrac[n] / xd;
  }

  return (1. + c1) / (cphat_ov_r + (c2 + c1 * c1) / (1. + c3));
}

}  // namespace kintera
