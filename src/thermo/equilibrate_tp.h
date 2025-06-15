#pragma once

// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
 * \param[out] gain WS gain matrix
 * \param[in,out] xfrac array of species mole fractions, modified in place.
 * \param[in] temp equilibrium temperature in Kelvin.
 * \param[in] pres equilibrium pressure in Pascals.
 * \param[in] nspecies number of species in the system.
 * \param[in] ngas number of gas species in the system.
 * \param[in] logsvp_func user-defined function for logarithm of saturation
 * vapor pressure.
 * with respect to temperature.
 * \param[in] logsvp_eps tolerance for convergence in logarithm of saturation
 * vapor pressure.
 * \param[in,out] max_iter maximum number of iterations allowed for convergence.
 */
template <typename T>
int equilibrate_tp(T *gain, T *diag, T *xfrac, T temp, T pres, T const *stoich,
                   int nspecies, int nreaction, int ngas,
                   user_func1 const *logsvp_func, float logsvp_eps,
                   int *max_iter) {
  // check positive temperature and pressure
  if (temp <= 0 || pres <= 0) {
    fprintf(stderr, "Error: Non-positive temperature or pressure.\n");
    return 1;  // error: non-positive temperature or pressure
  }

  // check positive gas fractions
  for (int i = 0; i < ngas; i++) {
    if (xfrac[i] <= 0) {
      fprintf(stderr, "Error: Non-positive gas fraction for species %d.\n", i);
      return 1;  // error: negative gas fraction
    }
  }

  // check non-negative solid concentration
  for (int i = ngas; i < nspecies; i++) {
    if (xfrac[i] < 0) {
      fprintf(stderr, "Error: Negative solid concentration for species %d.\n",
              i);
      return 1;  // error: negative solid concentration
    }
  }

  // check dimensions
  if (nspecies <= 0 || nreaction <= 0 || ngas < 1) {
    fprintf(stderr,
            "Error: nspecies, nreaction must be positive integers and ngas >= "
            "1.\n");
    return 1;  // error: invalid dimensions
  }

  T *logsvp = (T *)malloc(nreaction * sizeof(T));
  T *log_frac_sum = (T *)malloc(nreaction * sizeof(T));

  // weight matrix
  T *weight = (T *)malloc(nreaction * nspecies * sizeof(T));

  // right-hand-side vector
  T *rhs = (T *)malloc(nreaction * sizeof(T));

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
  // oversaturated reactions remains in the active set
  int oversaturated = 0;
  T lambda = 0.;  // rate scale factor
  while (iter++ < *max_iter) {
    /*printf("iter = %d, oversaturated = %d, lambda = %f\n", iter,
    oversaturated, lambda);
    // print xfrac
    printf("xfrac = ");
    for (int i = 0; i < nspecies; i++) {
      printf("%g ", xfrac[i]);
    }
    printf("\n");*/

    // fraction of gases
    T xg = 0.0;
    for (int i = 0; i < ngas; i++) {
      xg += xfrac[i];
    }

    // reorder reaction set
    // reset oversaturated if rates have been scaled
    if (lambda != 1.) oversaturated = 0;
    int first = oversaturated;
    int last = nreaction;

    // inactive                  |<--------------->|
    // undersaturated            |->               :
    // oversaturated     |<----->|                 :
    //               :...o       f                 :...l
    //               :   |       |                 :   |
    // | * * * * * * * | * * * * * * * * | * * * * * | x
    // |---------------|-----------------|-----------|
    // | OVERSATURATED | UNDERSATURATED  | INACTIVE  |
    while (first < last) {
      int j = reaction_set[first];
      log_frac_sum[j] = 0.0;
      T prod = 1.0;

      // active set condition variables
      for (int i = 0; i < nspecies; i++) {
        if (stoich[i * nreaction + j] < 0) {  // reactant
          log_frac_sum[j] += (-stoich[i * nreaction + j]) * log(xfrac[i] / xg);
        } else if (stoich[i * nreaction + j] > 0) {  // product
          prod *= xfrac[i];
        }
      }

      if (log_frac_sum[j] > (logsvp[j] + logsvp_eps)) {  // oversaturated
        int tmp = reaction_set[first];
        reaction_set[first] = reaction_set[oversaturated];
        reaction_set[oversaturated] = tmp;
        oversaturated++;
        if (oversaturated >= first) first++;
      } else if (log_frac_sum[j] < (logsvp[j] - logsvp_eps) && prod > 0.) {
        first++;
      } else {  // inactive
        int tmp = reaction_set[first];
        reaction_set[first] = reaction_set[last - 1];
        reaction_set[last - 1] = tmp;
        last--;
      }
    }

    /* print reaction set
    printf("reaction_set = ");
    for (int i = 0; i < oversaturated; i++) {
      printf("%d ", reaction_set[i]);
    }
    printf("| ");
    for (int i = oversaturated; i < first; i++) {
      printf("%d ", reaction_set[i]);
    }
    printf("| ");
    for (int i = first; i < nreaction; i++) {
      printf("%d ", reaction_set[i]);
    }
    printf("\n");*/

    // populate weight matrix and rhs vector
    nactive = first;
    for (int k = 0; k < nactive; k++) {
      int j = reaction_set[k];
      for (int i = 0; i < ngas; i++) {
        weight[k * nspecies + i] = -stoich_sum[j] / xg;
        if (stoich[i * nreaction + j] < 0) {
          weight[k * nspecies + i] += (-stoich[i * nreaction + j]) / xfrac[i];
        }
      }

      for (int i = ngas; i < nspecies; i++) {
        weight[k * nspecies + i] = 0.0;
      }

      rhs[k] = logsvp[j] - log_frac_sum[j];
    }

    if ((first == nactive) && (lambda == 1.)) {
      // all reactions are in equilibrium, no need to adjust saturation
      bool check_equilibrium = true;
      for (int k = 0; k < nactive; ++k) {
        if (fabs(rhs[k]) > logsvp_eps) check_equilibrium = false;
      }
      if (check_equilibrium) break;
    }

    // populate active stoichiometric and constraint matrix
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
    lambda = 1.;  // scale
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

  // restore the reaction order of gain
  T *gain_cpy = (T *)malloc(nreaction * nreaction * sizeof(T));
  memcpy(gain_cpy, gain, nreaction * nreaction * sizeof(T));
  memset(gain, 0, nreaction * nreaction * sizeof(T));

  for (int k = 0; k < nactive; k++) {
    for (int l = 0; l < nactive; l++) {
      int i = reaction_set[k];
      int j = reaction_set[l];
      gain[i * nreaction + j] = gain_cpy[k * nreaction + l];
    }
  }

  // save number of iterations to diag
  diag[0] = iter;

  free(logsvp);
  free(log_frac_sum);
  free(rhs);
  free(weight);
  free(reaction_set);
  free(stoich_active);
  free(stoich_sum);
  free(xfrac0);
  free(gain_cpy);

  if (iter >= *max_iter) {
    fprintf(stderr, "equilibrate_tp did not converge after %d iterations.\n",
            *max_iter);
    return 2 * 10 + kkt_err;  // failure to converge
  } else {
    *max_iter = iter;
    return kkt_err;  // success or KKT error
  }
}

}  // namespace kintera
