// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// kintera
#include <kintera/utils/func1.hpp>
#include <kintera/math/leastsq_kkt.h>
#include <kintera/math/mmdot.h>

namespace kintera {

/*!
 * \brief Calculate thermodynamic equilibrium at gven temperature and pressure
 *
 * This function finds the thermodynamic equilibrium for an array
 * of species.
 *
 * \param[in,out] xfrac array of species mole fractions, modified in place.
 * \param[in] temp equilibrium temperature in Kelvin.
 * \param[in] pres equilibrium pressure in Pascals.
 * \param[in] nspecies number of species in the system.
 * \param[in] ngas number of gas species in the system.
 * \param[in] logsvp_func user-defined function for logarithm of saturation
 * vapor pressure.
 * \param[in] logsvp_func_ddT user-defined function for derivative of logsvp
 * with respect to temperature.
 * \param[in] logsvp_eps tolerance for convergence in logarithm of saturation
 * vapor pressure.
 * \param[in,out] max_iter maximum number of iterations allowed for convergence.
 */
template <typename T>
int equilibrate_tp(T *xfrac, T temp, T pres,
                   T const *stoich, int nspecies, int nreaction, int ngas,
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

  // weight matrix
  T *weight = (T *)malloc(nreaction * nspecies * sizeof(T));

  // U matrix
  T *umat = (T *)malloc(nreaction * nreaction * sizeof(T));

  // right-hand-side vector
  T *rhs = (T *)malloc(nreaction * sizeof(T));

  // active set
  int *reaction_set = (int *)malloc(nreaction * sizeof(int));
  for (int i = 0; i < nreaction; i++) {
    reaction_set[i] = i;
  }

  // active stoichiometric matrix
  T *stoich_active =
      (T *)malloc(nspecies * nreaction * sizeof(T));

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
  while (iter++ < *max_iter) {
    // fraction of gases
    T xg = 0.0;
    for (int i = 0; i < ngas; i++) {
      xg += xfrac[i];
    }

    // populate weight matrix, rhs vector and active set
    int first = 0;
    int last = nreaction;
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

      // active set, weight matrix and rhs vector
      if ((log_frac_sum < (logsvp[j] - logsvp_eps) && prod > 0.) ||
          (log_frac_sum > (logsvp[j] + logsvp_eps))) {
        for (int i = 0; i < nspecies; i++) {
          weight[first * nspecies + i] = 0.0;
          if (stoich[i * nreaction + j] < 0) {
            weight[first * nspecies + i] +=
                (-stoich[i * nreaction + j]) / xfrac[i];
          }

          if (i < ngas) {
            weight[first * nspecies + i] -= stoich_sum[j] / xg;
          }
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

    // form active stoichiometric and constraint matrix
    int nactive = first;
    for (int i = 0; i < nspecies; i++)
      for (int k = 0; k < nactive; k++) {
        int j = reaction_set[k];
        stoich_active[i * nactive + k] = stoich[i * nreaction + j];
      }

    mmdot(umat, weight, stoich_active, nactive, nspecies, nactive);

    for (int i = 0; i < nspecies; i++)
      for (int k = 0; k < nactive; k++) {
        stoich_active[i * nactive + k] *= -1;
      }

    // solve constrained optimization problem (KKT)
    int max_kkt_iter = *max_iter;
    kkt_err = leastsq_kkt(rhs, umat, stoich_active, xfrac, nactive, nactive,
                          nspecies, 0, &max_kkt_iter);
    if (kkt_err != 0) break;

    // rate -> xfrac
    // copy xfrac to xfrac0
    memcpy(xfrac0, xfrac, nspecies * sizeof(T));
    T lambda = 1.;  // scale

    while (true) {
      bool positive_vapor = true;
      for (int i = 0; i < nspecies; i++) {
        for (int k = 0; k < nactive; k++) {
          int j = reaction_set[k];
          xfrac[i] = xfrac0[i] + stoich[i * nactive + j] * rhs[k] * lambda;
        }
        if (i < ngas && xfrac[i] <= 0.) positive_vapor = false;
      }
      if (positive_vapor) break;
      lambda *= 0.99;
    }
  }

  free(logsvp);
  free(weight);
  free(rhs);
  free(umat);
  free(reaction_set);
  free(stoich_active);
  free(stoich_sum);
  free(xfrac0);

  if (iter >= *max_iter) {
    fprintf(stderr,
            "Saturation adjustment did not converge after %d iterations.\n",
            *max_iter);
    return 2;  // failure to converge
  } else {
    *max_iter = iter;
    return kkt_err;  // success or KKT error
  }
}

} // namespace kintera
