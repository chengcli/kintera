#pragma once

#include <functional>  // function

// base
#include <configure.h>

// kintera
#include <kintera/index.h>

#define W(n) w[(n) * stride]
#define Q(n) q[(n) * stride]

namespace kintera {

template <typename T>
inline DISPATCH_MACRO void mole_to_mass_fraction_impl(T *w, T const *q,
                                                      T const *mu_ratio_m1,
                                                      int nmass, int stride) {
  // set mass mixing ratio
  T sum = 1.;
  for (int n = 0; n < nmass; ++n) {
    W(n) = Q(n) / (mu_ratio_m1[n] + 1);
    sum -= Q(n) * mu_ratio_m1[n] / (mu_ratio_m1[n] + 1.);
  }
  for (int n = 0; n < nmass; ++n) W(n) /= sum;
}

template <typename T>
inline DIAPTCH_MACRO void mass_to_mole_fraction_impl(T *q, T const *w,
                                                     T const *mu_ratio_m1,
                                                     int nmass, int stride) {
  // set molar mixing ratio
  T sum = 1.;
  for (int n = 0; n < nmass; ++n) {
    Q(n) = W(n) * (mu_ratio_m1[n] + 1.);
    sum += W(n) * mu_ratio_m1[n];
  }
  for (int n = 0; n < nmass; ++n) Q(n) /= sum;
}

/*template <typename T>
inline DISPATCH_MACRO void update_tp_conserving_u(T *q, T rho, T uhat, T gammad)
{ T cv = 1., qtol = 1., qeps = 1.;

  for (int n = 1 + NVAPOR; n < NMASS; ++n) {
    uhat += beta_[n]*q[n];
    qtol -= q[n];
  }
  for (int n = 0; n < NMASS; ++n) {
    cv += (rcv_[n] * eps_[n] - 1.)*q[n];
    qeps += q[n]*(eps_[n] - 1.);
  }

  q[IDN] = (gammad - 1.)*uhat/cv;
  q[IPR] = rho*Rd_*q[IDN]*qtol/qeps;
}

inline DISPATCH_MACRO void print_error_msg() {
  std::stringstream msg;
  msg << "### FATAL ERROR in function Thermodynamics::SaturationAdjustment"
      << std::endl << "Iteration reaches maximum."
      << std::endl;
  msg << "Location: x3 = " << pmb->pcoord->x3v(k) << std::endl
      << "          x2 = " << pmb->pcoord->x2v(j) << std::endl
      << "          x1 = " << pmb->pcoord->x1v(i) << std::endl;
  msg << "Variables before iteration u = (";
  for (int n = 0; n < NHYDRO-1; ++n)
    msg << u(n,k,j,i) << ", ";
  msg << u(NHYDRO-1,k,j,i) << ")" << std::endl;
  msg << "Variables before iteration q0 = (";
  for (int n = 0; n < NHYDRO-1; ++n)
    msg << q0[n] << ", ";
  msg << q0[NHYDRO-1] << ")" << std::endl;
  msg << "Variables after iteration q = (";
  for (int n = 0; n < NHYDRO-1; ++n)
    msg << q[n] << ", ";
  msg << q[NHYDRO-1] << ")" << std::endl;
  ATHENA_ERROR(msg);
}

template <typename T>
inline DISPATCH_MACRO void saturation_adjustment(T* q, T* gammad)
{
  // mass to molar mixing ratio
  mass_to_mole_fraction(q, w, mu_ratio_m1, nmass);

  Real rho = 0., rho_hat = 0.;
  for (int n = 0; n < NMASS; ++n) {
    rho += u(n,k,j,i);
    rho_hat += u(n,k,j,i)/eps_[n];
    q[n] = u(n,k,j,i)/eps_[n];
  }
  for (int n = 0; n < NMASS; ++n)
    q[n] /= rho_hat;

  // save q for debug purpose
  memcpy(q0, q, NHYDRO*sizeof(Real));

  // calculate internal energy
  Real KE = 0.5*(u(IM1,k,j,i)*u(IM1,k,j,i)
               + u(IM2,k,j,i)*u(IM2,k,j,i)
               + u(IM3,k,j,i)*u(IM3,k,j,i))/rho;
  Real uhat = (u(IEN,k,j,i) - KE)/(Rd_*rho_hat);
  UpdateTPConservingU(q, rho, uhat);

  // check boiling condition
  for (int n = 1; n <= NVAPOR; ++n) {
    int ic = n + NVAPOR, ip = n + 2*NVAPOR;
    Real svp = SatVaporPressure(q[IDN], ic);
    if (svp > q[IPR]) { // boiling
      q[ic] += q[ip];
      q[ip] = 0.;
    }
  }

  Real qsig = 1.;
  for (int n = 1; n < NMASS; ++n)
    qsig += q[n]*(rcv_[n]*eps_[n] - 1.);

  int iter = 0;
  Real t, alpha, rate;
  while (iter++ < max_iter_) {
    // condensation
    for (int n = 1; n <= NVAPOR; ++n) {
      int nc = n + NVAPOR;
      t = q[IDN]/t3_[nc];
      alpha = (gamma - 1.)*(beta_[nc]/t - delta_[nc] - 1.)/qsig;
      rate = GasCloudIdeal(q, n, nc, t3_[nc], p3_[nc], alpha, beta_[nc],
delta_[nc]); q[n] -= rate; q[nc] += rate;
    }
    Real Told = q[IDN];
    update_tp_conserving_u(q, rho, uhat);
    if (fabs(q[IDN] - Told) < ftol_) break;
  }

  if (iter >= max_iter_) print_error_msg();
}*/

}  // namespace kintera

#undef W
#undef Q
