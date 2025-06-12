#pragma once

// C/C++
#include <cmath>
#include <string>

// kintera
#include <kintera/constants.h>

#include <kintera/utils/func1.hpp>

#include "cal_dlnT_dlnP.h"
#include "equilibrate_tp.h"

namespace kintera {

template <typename T>
void integrate_z(T* xfrac, T* temp, T* pres, T* mu, T dz, char const* method,
                 T grav, T adTdz, T const* stoich, int nspecies, int ngas,
                 T const* enthalpy, T const* cp, user_func1 const* logsvp_func,
                 float logsvp_eps, int* max_iter) {
  T* latent = (T*)malloc(ngas * sizeof(T));
  T* cp_ratio = (T*)malloc(nspecies * sizeof(T));
  memset(latent, 0, ngas * sizeof(T));
}

void extrapolate_z_(torch::Tensor temp, torch::Tensor pres, torch::Tensor xfrac,
                    double dz, std::string method, double grav, double adTdz,
                    ThermoX& thermo) {
  auto conc = thermo->compute("TPX->V", {temp, pres, xfrac});
  auto cp_ratio = eval_cp_R(temp, xfrac, thermo->options);
  cp_ratio /= cp_ratio[0];  // normalize to the first species

  // This only considers the first (ngas-1) reactions
  for (int i = 1; i < ngas; ++i) {
    for (int j = 0; j < nspecies; ++j) {  // find out condensates
      if (stoich[j * (ngas - 1) + (i - 1)] > 0 || xfrac[j] > 0.) {
        latent[i] -= enthalpy[j];
      }
    }
    if (latent[i] < 0.) latent[i] += enthalpy[i];
  }

  T xgas = 1., xeps = 1.;
  for (int i = 1; i < ngas; ++i) {
    xeps += xfrac[i] * (mu[i] / mu[0] - 1.);
  }

  for (int i = ngas; i < nspecies; ++i) {
    xeps += xfrac[i] * (mu[i] / mu[0] - 1.);
    xgas += -xfrac[i];
  }

  T Rd = constants::Rgas / mu[0];  // molar gas constant
  T g_ov_Rd = grav / Rd;
  T R_ov_Rd = xgas / xeps;
  T gammad = cp[0] / (cp[0] - Rd);  // adiabatic index
  T chi = 0.;

  if (strcmp(method, "reversible") == 0 || strcmp(method, "pseudo") == 0) {
    chi = cal_dlnT_dlnP(xfrac, &gammad, cp_ratio, latent, ngas - 1,
                        nspecies - ngas);
  } else if (strcmp(method, "dry") == 0) {
    for (int i = 1; i < ngas; ++i) latent[i] = 0;
    chi = cal_dlnT_dlnP(xfrac, &gammad, cp_ratio, latent, ngas - 1,
                        nspecies - ngas);
  } else {  // isothermal
    chi = 0.;
  }

  T dTdz = -chi * g_ov_Rd / R_ov_Rd + adTdz;
  chi = -R_ov_Rd / g_ov_Rd * dTdz;

  // integrate over dz
  T temp0 = *temp;
  (*temp) += dTdz * dz;

  if ((*temp) <= 0.) (*temp) = temp0;

  if (fabs((*temp) - temp0) > 0.01) {
    (*pres) *= pow(*temp / temp0, 1. / chi);
  } else {  // isothermal limit
    (*pres) *= exp(-2. * g_ov_Rd * dz / (R_ov_Rd * (*temp + temp0)));
  }

  equilibrate_tp(xfrac, *temp, *pres, stoich, nspecies, ngas - 1, ngas,
                 logsvp_func, logsvp_eps, max_iter);

  free(latent);
  free(cp_ratio);
}

}  // namespace kintera
