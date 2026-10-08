#pragma once

// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

// base
#include <configure.h>

// kintera
#include <kintera/math/core.h>
#include <kintera/math/leastsq_kkt.h>

#include <kintera/utils/user_funcs.hpp>

#include "svp_eval.h"

namespace kintera {

/*!
 * \brief Calculate thermodynamic equilibrium at gven temperature and pressure
 *
 * This function finds the thermodynamic equilibrium for an array
 * of species.
 *
 * \param[out] gain             WS gain matrix
 * \param[in,out]               xfrac array of species mole fractions, modified
 * in place.
 * \param[in] temp              equilibrium temperature in Kelvin.
 * \param[in] pres              equilibrium pressure in Pascals.
 * \param[in] nspecies          number of species in the system.
 * \param[in] nreaction         number of reactions in the system.
 * \param[in] ngas              number of gas species in the system.
 * \param[in] logsvp_func       user-defined function for logarithm of
 * saturation vapor pressure with respect to temperature.
 * \param[in] logsvp_eps        tolerance for convergence in logarithm
 *                              of saturation vapor pressure.
 * \param[in,out] max_iter      maximum number of iterations allowed for
 *                              convergence.
 * \param[in,out] reaction_set  active set of reactions, modified in place.
 * \param[in,out] nactive       number of active reactions, modified in place.
 */
template <typename T>
DISPATCH_MACRO int equilibrate_tp(T *gain, T *diag, T *xfrac, T temp, T pres,
                                  T const *stoich, int nspecies, int nreaction,
                                  int ngas, user_func1 const *logsvp_func,
                                  int const *svp_kind, double const *svp_params,
                                  float logsvp_eps, int *max_iter,
                                  int *reaction_set, int *nactive,
                                  char *work = nullptr) {
  diag[0] = -100.;  // invalid input unless overwritten below
  // check positive temperature and pressure
  if (temp <= 0 || pres <= 0) {
    printf("Error: Non-positive temperature or pressure.\n");
    return 1;  // error: non-positive temperature or pressure
  }

  // check positive gas fractions
  for (int i = 0; i < ngas; i++) {
    if (xfrac[i] < 0) {
      printf("Error: Negative gas fraction for species %d.\n", i);
      return 1;  // error: negative gas fraction
    }
  }

  // check non-negative solid concentration
  for (int i = ngas; i < nspecies; i++) {
    if (xfrac[i] < 0) {
      printf(
          "Warning: Negative solid concentration (%f) for species %d. Setting "
          "to "
          "zero\n",
          xfrac[i], i);
      xfrac[i] = 0.;
      // return 1;  // error: negative solid concentration
    }
  }

  // check dimensions
  if (nspecies <= 0 || nreaction <= 0 || ngas < 1) {
    printf(
        "Error: nspecies, nreaction must be positive integers and ngas >= "
        "1.\n");
    return 1;  // error: invalid dimensions
  }

  T initial_gas = 0.;
  for (int i = 0; i < ngas; ++i) initial_gas += xfrac[i];
  if (!(initial_gas > 0.)) return 1;

  T *logsvp, *weight, *rhs;
  T *stoich_active, *stoich_sum, *xfrac0;
  T *gain_cpy, *theta;
  size_t mark = pool_mark(work);
  logsvp = (T *)pmalloc(work, nreaction * sizeof(T));
  weight = (T *)pmalloc(work, nreaction * nspecies * sizeof(T));
  rhs = (T *)pmalloc(work, nreaction * sizeof(T));
  stoich_active = (T *)pmalloc(work, nspecies * nreaction * sizeof(T));
  stoich_sum = (T *)pmalloc(work, nreaction * sizeof(T));
  xfrac0 = (T *)pmalloc(work, nspecies * sizeof(T));
  gain_cpy = (T *)pmalloc(work, nreaction * nreaction * sizeof(T));
  theta = (T *)pmalloc(work, nspecies * sizeof(T));

  memset(weight, 0, nreaction * nspecies * sizeof(T));
  memset(rhs, 0, nreaction * sizeof(T));

  // evaluate log vapor saturation pressure and its derivative
  for (int j = 0; j < nreaction; j++) {
    stoich_sum[j] = 0.0;
    for (int i = 0; i < ngas; i++) stoich_sum[j] -= stoich[i * nreaction + j];
    logsvp[j] = eval_logsvp(svp_kind[j], svp_params + j * KSVP_NPARAM,
                            logsvp_func[j], temp) -
                stoich_sum[j] * log(pres);
  }

  int iter = 0;
  int kkt_err = 0;
  // Set when the active set is empty, including on the last allowed iteration.
  // The cap test below would otherwise report that solve as a failure.
  bool at_equilibrium = false;
  T lambda = 0.;  // rate scale factor
  while (iter++ < *max_iter) {
    /*printf("iter = %d\n ", iter);
    // print xfrac
    printf("- xfrac = ");
    for (int i = 0; i < nspecies; i++) {
      printf("%g ", xfrac[i]);
    }
    printf("\n");*/

    // fraction of gases
    T xg = 0.0;
    for (int i = 0; i < ngas; i++) xg += xfrac[i];

    // populate weight matrix, rhs vector and active set
    int first = 0;
    int last = nreaction;
    while (first < last) {
      int j = reaction_set[first];
      T log_frac_sum = 0.0;
      T prod = 1.0;

      // Signed gas quotient; condensed phases have unit activity.
      bool absent_reactant = false, absent_product = false;
      T nu_absent = 0., log_nu = 0.;
      for (int i = 0; i < nspecies; i++) {
        T nu = -stoich[i * nreaction + j];
        if (i < ngas && nu != 0.) {
          if (xfrac[i] == 0.) {
            absent_reactant |= nu > 0.;
            absent_product |= nu < 0.;
            nu_absent += nu;
            log_nu += nu * log(fabs(nu) / xg);
          } else {
            log_frac_sum += nu * log(xfrac[i] / xg);
          }
        } else if (i >= ngas && nu < 0.) {
          prod *= xfrac[i];
        }
      }
      // Boundary derivative along a feasible reaction extent, without seeding
      // gas amounts. This also handles an initially absent gaseous product.
      T log_r = 0.;
      if (nu_absent != 0.) {
        log_r = (logsvp[j] - log_frac_sum - log_nu) / nu_absent;
        T bound = 0.5 * log(std::numeric_limits<T>::max());
        log_r = fmax(-bound, fmin(bound, log_r));
        log_frac_sum = logsvp[j] - nu_absent;
      }

      // Neither direction is feasible when both sides lack a required gas.
      // Other reactions may supply it on a later outer iteration.
      if (absent_reactant && absent_product) {
        int tmp = reaction_set[first];
        reaction_set[first] = reaction_set[last - 1];
        reaction_set[last - 1] = tmp;
        --last;
        continue;
      }

      // active set, weight matrix and rhs vector
      if ((log_frac_sum < (logsvp[j] - logsvp_eps) && prod > 0.) ||
          (log_frac_sum > (logsvp[j] + logsvp_eps))) {
        for (int i = 0; i < ngas; i++) {
          weight[first * nspecies + i] = -stoich_sum[j] / xg;
          T nu = -stoich[i * nreaction + j];
          if (nu != 0.) {
            weight[first * nspecies + i] +=
                xfrac[i] > 0. ? nu / xfrac[i]
                              : (nu > 0. ? 1. : -1.) * exp(-log_r);
          }
        }
        for (int i = ngas; i < nspecies; i++) {
          weight[first * nspecies + i] = 0.0;
        }
        rhs[first] = logsvp[j] - log_frac_sum;
        first++;
      } else {
        int tmp = reaction_set[first];
        reaction_set[first] = reaction_set[last - 1];
        reaction_set[last - 1] = tmp;
        last--;
      }
    }

    *nactive = first;
    if (first == 0) {
      // all reactions are in equilibrium, no need to adjust saturation
      at_equilibrium = true;
      break;
    }

    // populate active stoichiometric and constraint matrix
    (*nactive) = first;
    for (int i = 0; i < nspecies; i++)
      for (int k = 0; k < (*nactive); k++) {
        int j = reaction_set[k];
        stoich_active[i * (*nactive) + k] = stoich[i * nreaction + j];
      }

    mmdot(gain, weight, stoich_active, *nactive, nspecies, *nactive);

    /* print gain
    printf("gain = \n");
    for (int i = 0; i < (*nactive); i++) {
      for (int j = 0; j < (*nactive); j++) {
        printf("%f ", gain[i * (*nactive) + j]);
      }
      printf("\n");
    }

    // print rhs
    printf("rhs = ");
    for (int k = 0; k < (*nactive); k++) {
      printf("%f ", rhs[k]);
    }

    // print xfrac
    printf("\nxfrac = ");
    for (int i = 0; i < nspecies; i++) {
      printf("%f ", xfrac[i]);
    }
    printf("\n");*/

    for (int i = 0; i < nspecies; i++)
      for (int k = 0; k < (*nactive); k++) {
        stoich_active[i * (*nactive) + k] *= -1;
      }
    // note that stoich_active is negated

    // solve constrained optimization problem (KKT)
    // Bound the inner active-set solve by the constraint count, not by the
    // outer Newton budget: sharing one knob makes a small max_iter abort the
    // KKT solve, after which equilibrate returns the state unchanged and
    // silently.
    int max_kkt_iter = nspecies + 1 > *max_iter ? nspecies + 1 : *max_iter;
    // x = 0 (no extent) is feasible: the bounds are the current mole fractions
    kkt_err = leastsq_kkt_feasible_origin(rhs, gain, stoich_active, xfrac,
                                          *nactive, *nactive, nspecies, 0,
                                          &max_kkt_iter, 0., work);
    if (kkt_err != 0) break;

    /* print rate
    printf("rate = ");
    for (int k = 0; k < (*nactive); k++) {
      printf("%f ", rhs[k]);
    }
    printf("\n");*/

    // rate -> xfrac
    // copy xfrac to xfrac0
    memcpy(xfrac0, xfrac, nspecies * sizeof(T));
    T lambda = 1.;  // scale
    // Per-reaction extent limit, as in equilibrate_uv: the KKT solve does not
    // enforce a row dependent on its active block, so clip each reaction to
    // the stock of what it consumes (a cloud must not go negative).
    for (int i = 0; i < nspecies; i++) {
      T demand = 0.;
      for (int k = 0; k < (*nactive); k++) {
        T ck = -stoich_active[i * (*nactive) + k] * rhs[k];
        if (ck < 0.) demand -= ck;
      }
      theta[i] =
          demand > xfrac0[i] ? (xfrac0[i] > 0. ? xfrac0[i] / demand : 0.) : 1.;
    }
    for (int k = 0; k < (*nactive); k++) {
      T lam = 1.;
      for (int i = 0; i < nspecies; i++) {
        T ck = -stoich_active[i * (*nactive) + k] * rhs[k];
        if (ck < 0. && theta[i] < lam) lam = theta[i];
      }
      rhs[k] *= lam;
    }
    T xsum;
    while (true) {
      bool positive_vapor = true;
      xsum = 0.;
      for (int i = 0; i < nspecies; i++) {
        for (int k = 0; k < (*nactive); k++) {
          xfrac[i] -= stoich_active[i * (*nactive) + k] * rhs[k] * lambda;
        }
        if (i < ngas && (xfrac[i] < 0. || (xfrac0[i] > 0. && xfrac[i] == 0.)))
          positive_vapor = false;
        xsum += xfrac[i];
      }
      if (positive_vapor) break;
      lambda *= 0.5;
      memcpy(xfrac, xfrac0, nspecies * sizeof(T));
      if (lambda < std::numeric_limits<T>::min()) {
        kkt_err = 4;
        break;
      }
    }

    if (kkt_err) break;

    // re-normalize mole fractions
    for (int i = 0; i < nspecies; i++) xfrac[i] /= xsum;
  }

  /////////// Construct a gain matrix of active reactions ///////////
  int first = 0;
  int last = nreaction;
  T xg = 0.0;
  for (int i = 0; i < ngas; i++) xg += xfrac[i];

  while (first < last) {
    int j = reaction_set[first];
    T log_frac_sum = 0.0;
    T prod = 1.0;

    // active set condition variables
    for (int i = 0; i < nspecies; i++) {
      if (i < ngas && stoich[i * nreaction + j] != 0.) {
        log_frac_sum += (-stoich[i * nreaction + j]) *
                        log(fmax(xfrac[i] / xg, std::numeric_limits<T>::min()));
      } else if (i >= ngas && stoich[i * nreaction + j] > 0) {  // cloud
        prod *= xfrac[i];
      }
    }

    // active set and weight matrix
    if ((log_frac_sum >= (logsvp[j] - logsvp_eps) &&
         (log_frac_sum <= (logsvp[j] + logsvp_eps)))) {
      for (int i = 0; i < ngas; i++) {
        weight[first * nspecies + i] = -stoich_sum[j] / xg;
        if (stoich[i * nreaction + j] != 0. && xfrac[i] > 0.) {
          weight[first * nspecies + i] -= stoich[i * nreaction + j] / xfrac[i];
        }
      }
      for (int i = ngas; i < nspecies; i++) {
        weight[first * nspecies + i] = 0.0;
      }
      first++;
    } else {
      int tmp = reaction_set[first];
      reaction_set[first] = reaction_set[last - 1];
      reaction_set[last - 1] = tmp;
      last--;
    }
  }

  // populate active stoichiometric and constraint matrix
  (*nactive) = first;
  for (int i = 0; i < nspecies; i++)
    for (int k = 0; k < (*nactive); k++) {
      int j = reaction_set[k];
      stoich_active[i * (*nactive) + k] = stoich[i * nreaction + j];
    }

  mmdot(gain_cpy, weight, stoich_active, *nactive, nspecies, *nactive);
  memset(gain, 0, nreaction * nreaction * sizeof(T));

  for (int k = 0; k < (*nactive); k++) {
    for (int l = 0; l < (*nactive); l++) {
      int i = reaction_set[k];
      int j = reaction_set[l];
      gain[i * nreaction + j] = gain_cpy[k * (*nactive) + l];
    }
  }

  int n_iter = iter > *max_iter ? *max_iter : iter;
  int status = kkt_err ? kkt_err : (at_equilibrium ? 0 : 20);
  diag[0] = status ? -(100. * status + n_iter) : n_iter;

  pfree(logsvp);
  pfree(rhs);
  pfree(weight);
  pfree(stoich_active);
  pfree(stoich_sum);
  pfree(xfrac0);
  pfree(gain_cpy);
  pfree(theta);
  pool_rewind(work, mark);

  if (status == 20) {
    printf("equilibrate_tp did not converge after %d iterations.\n", *max_iter);
  } else {
    *max_iter = n_iter;
  }
  return status;
}

}  // namespace kintera
