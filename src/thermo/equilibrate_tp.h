#pragma once

// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// base
#include <configure.h>

// kintera
#include <kintera/math/leastsq_kkt.h>
#include <kintera/math/mmdot.h>

#include <kintera/utils/func1.hpp>

namespace kintera {

/*!
 * \brief Calculate thermodynamic equilibrium at gven temperature and pressure
 *
 * This function finds the thermodynamic equilibrium for an array
 * of species.
 *
 * \param[out] gain           WS gain matrix
 * \param[in,out]             xfrac array of species mole fractions, modified in
 * place.
 * \param[in] temp            equilibrium temperature in Kelvin.
 * \param[in] pres            equilibrium pressure in Pascals.
 * \param[in] nspecies        number of species in the system.
 * \param[in] ngas            number of gas species in the system.
 * \param[in] logsvp_func     user-defined function for logarithm of saturation
 *                            vapor pressure with respect to temperature.
 * \param[in]                 logsvp_eps tolerance for convergence in logarithm
 * of saturation vapor pressure.
 * \param[in,out] max_iter    maximum number of iterations allowed for
 * convergence.
 */
template <typename T>
DISPATCH_MACRO int equilibrate_tp(T *gain, T *diag, T *xfrac, T temp, T pres,
                                  T const *stoich, int nspecies, int nreaction,
                                  int ngas, user_func1 const *logsvp_func,
                                  float logsvp_eps, int *max_iter) {
  // check positive temperature and pressure
  if (temp <= 0 || pres <= 0) {
    printf("Error: Non-positive temperature or pressure.\n");
    return 1;  // error: non-positive temperature or pressure
  }

  // check positive gas fractions
  for (int i = 0; i < ngas; i++) {
    if (xfrac[i] <= 0) {
      printf("Error: Non-positive gas fraction for species %d.\n", i);
      return 1;  // error: negative gas fraction
    }
  }

  // check non-negative solid concentration
  for (int i = ngas; i < nspecies; i++) {
    if (xfrac[i] < 0) {
      printf("Error: Negative solid concentration for species %d.\n", i);
      return 1;  // error: negative solid concentration
    }
  }

  // check dimensions
  if (nspecies <= 0 || nreaction <= 0 || ngas < 1) {
    printf(
        "Error: nspecies, nreaction must be positive integers and ngas >= "
        "1.\n");
    return 1;  // error: invalid dimensions
  }

  T *logsvp = (T *)malloc(nreaction * sizeof(T));

  // weight matrix
  T *weight = (T *)malloc(nreaction * nspecies * sizeof(T));
  memset(weight, 0, nreaction * nspecies * sizeof(T));

  // right-hand-side vector
  T *rhs = (T *)malloc(nreaction * sizeof(T));
  memset(rhs, 0, nreaction * sizeof(T));

  // active set
  int *reaction_set = (int *)malloc(nreaction * sizeof(int));
  for (int i = 0; i < nreaction; i++) {
    reaction_set[i] = i;
  }

  // active stoichiometric matrix
  T *stoich_active = (T *)malloc(nspecies * nreaction * sizeof(T));

  // sum of reactant stoichiometric coefficients
  T *stoich_sum = (T *)malloc(nreaction * sizeof(T));

  // copy of xfrac
  T *xfrac0 = (T *)malloc(nspecies * sizeof(T));

  // evaluate log vapor saturation pressure and its derivative
  for (int j = 0; j < nreaction; j++) {
    stoich_sum[j] = 0.0;
    for (int i = 0; i < nspecies; i++)
      if (stoich[i * nreaction + j] < 0) {  // reactant
        stoich_sum[j] += (-stoich[i * nreaction + j]);
      }
    logsvp[j] = logsvp_func[j](temp) - stoich_sum[j] * log(pres);
  }

  int iter = 0;
  int kkt_err = 0;
  int nactive = 0;
  T lambda = 0.;  // rate scale factor
  while (iter++ < *max_iter) {
    /*printf("iter = %d\n ", iter);
    // print xfrac
    printf("xfrac = ");
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

      // active set condition variables
      for (int i = 0; i < nspecies; i++) {
        if ((stoich[i * nreaction + j] < 0) && (xfrac[i] > 0.)) {  // reactant
          log_frac_sum += (-stoich[i * nreaction + j]) * log(xfrac[i] / xg);
        } else if (stoich[i * nreaction + j] > 0) {  // product
          prod *= xfrac[i];
        }
      }

      // active set, weight matrix and rhs vector
      if ((log_frac_sum < (logsvp[j] - logsvp_eps) && prod > 0.) ||
          (log_frac_sum > (logsvp[j] + logsvp_eps))) {
        for (int i = 0; i < ngas; i++) {
          weight[first * nspecies + i] = -stoich_sum[j] / xg;
          if ((stoich[i * nreaction + j] < 0) && (xfrac[i] > 0.)) {
            weight[first * nspecies + i] -=
                stoich[i * nreaction + j] / xfrac[i];
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

    if (first == 0) {
      // all reactions are in equilibrium, no need to adjust saturation
      break;
    }

    // populate active stoichiometric and constraint matrix
    nactive = first;
    for (int i = 0; i < nspecies; i++)
      for (int k = 0; k < nactive; k++) {
        int j = reaction_set[k];
        stoich_active[i * nactive + k] = stoich[i * nreaction + j];
      }

    mmdot(gain, weight, stoich_active, nactive, nspecies, nactive);

    for (int i = 0; i < nspecies; i++)
      for (int k = 0; k < nactive; k++) {
        stoich_active[i * nactive + k] *= -1;
      }
    // note that stoich_active is negated

    // solve constrained optimization problem (KKT)
    int max_kkt_iter = *max_iter;
    kkt_err = leastsq_kkt(rhs, gain, stoich_active, xfrac, nactive, nactive,
                          nspecies, 0, &max_kkt_iter);
    if (kkt_err != 0) break;

    /* print rate
    printf("rate = ");
    for (int k = 0; k < nactive; k++) {
      printf("%f ", rhs[k]);
    }
    printf("\n");*/

    // rate -> xfrac
    // copy xfrac to xfrac0
    memcpy(xfrac0, xfrac, nspecies * sizeof(T));
    T lambda = 1.;  // scale
    T xsum;
    while (true) {
      bool positive_vapor = true;
      xsum = 0.;
      for (int i = 0; i < nspecies; i++) {
        for (int k = 0; k < nactive; k++) {
          xfrac[i] -= stoich_active[i * nactive + k] * rhs[k] * lambda;
        }
        if (i < ngas && xfrac[i] <= 0.) positive_vapor = false;
        xsum += xfrac[i];
      }
      if (positive_vapor) break;
      lambda *= 0.99;
      memcpy(xfrac, xfrac0, nspecies * sizeof(T));
    }

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
      if (stoich[i * nreaction + j] < 0) {  // reactant
        log_frac_sum += (-stoich[i * nreaction + j]) * log(xfrac[i] / xg);
      } else if (stoich[i * nreaction + j] > 0) {  // product
        prod *= xfrac[i];
      }
    }

    // active set and weight matrix
    if ((log_frac_sum >= (logsvp[j] - logsvp_eps) &&
         (log_frac_sum <= (logsvp[j] + logsvp_eps)))) {
      for (int i = 0; i < ngas; i++) {
        weight[first * nspecies + i] = -stoich_sum[j] / xg;
        if (stoich[i * nreaction + j] < 0) {
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
  nactive = first;
  for (int i = 0; i < nspecies; i++)
    for (int k = 0; k < nactive; k++) {
      int j = reaction_set[k];
      stoich_active[i * nactive + k] = stoich[i * nreaction + j];
    }

  T *gain_cpy = (T *)malloc(nreaction * nreaction * sizeof(T));
  mmdot(gain_cpy, weight, stoich_active, nactive, nspecies, nactive);
  memset(gain, 0, nreaction * nreaction * sizeof(T));

  for (int k = 0; k < nactive; k++) {
    for (int l = 0; l < nactive; l++) {
      int i = reaction_set[k];
      int j = reaction_set[l];
      gain[i * nreaction + j] = gain_cpy[k * nactive + l];
    }
  }

  // save number of iterations to diag
  diag[0] = iter;

  free(logsvp);
  free(rhs);
  free(weight);
  free(reaction_set);
  free(stoich_active);
  free(stoich_sum);
  free(xfrac0);
  free(gain_cpy);

  if (iter >= *max_iter) {
    printf("equilibrate_tp did not converge after %d iterations.\n", *max_iter);
    return 2 * 10 + kkt_err;  // failure to converge
  } else {
    *max_iter = iter;
    return kkt_err;  // success or KKT error
  }
}

}  // namespace kintera
