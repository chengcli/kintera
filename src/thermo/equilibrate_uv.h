#pragma once

// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>

// base
#include <configure.h>

// kintera
#include <kintera/constants.h>
#include <kintera/math/core.h>
#include <kintera/math/leastsq_kkt.h>

#include <kintera/utils/user_funcs.hpp>

namespace kintera {

/*!
 * \brief Calculate thermodynamic equilibrium at fixed volume and internal
 * energy
 *
 * Given an initial guess of temperature and concentrations, this function
 * adjusts the temperature and concentrations to satisfy the saturation
 * condition.
 *
 * \param[out] gain             WS gain matrix
 * \param[out] diag             diagnostic output
 * \param[in,out] temp          in:initial temperature
 *                              out: adjusted temperature.
 * \param[in,out] conc          in:initial concentrations for each species
 *                              out: adjusted concentrations.
 * \param[in] h0                initial internal energy.
 * \param[in] stoich            reaction stoichiometric matrix, nspecies x
 *                              nreaction.
 * \param[in] nspecies          number of species in the system.
 * \param[in] nreaction         number of reactions in the system.
 * \param[in] ngas              number of gas species in the system.
 * \param[in] intEng_offset     offset for internal energy calculations.
 * \param[in] cv_const          const component of heat capacity.
 * \param[in] logsvp_func       user-defined functions for logarithm of
 *                              saturation vapor pressure.
 * \param[in] logsvp_func_ddT   user-defined functions for derivative of logsvp
 *                              with respect to temperature.
 * \param[in] intEng_R_extra    user-defined functions for internal energy
 *                              calculation in addition to the linear term.
 * \param[in] cv_R_extra        user-defined functions for heat capacity
 *                              calculation in addition to the constant term.
 * \param[in] lnsvp_eps         tolerance for convergence in logarithm of
 *                              saturation vapor pressure.
 * \param[in] reaction_set      active set of reactions, modified in place.
 * \param[in] nactive           number of active reactions, modified in place.
 * \param[in,out] max_iter      maximum number of iterations allowed for
 *                              convergence.
 */
template <typename T>
DISPATCH_MACRO int equilibrate_uv(
    T *gain, T *diag, T *temp, T *conc, T h0, T const *stoich, int nspecies,
    int nreaction, int ngas, T const *intEng_offset, T const *cv_const,
    user_func1 const *logsvp_func, user_func1 const *logsvp_func_ddT,
    user_func2 const *intEng_R_extra, user_func2 const *cv_R_extra,
    float logsvp_eps, int *max_iter, int *reaction_set, int *nactive,
    char *work = nullptr) {
  // check positive temperature
  if (*temp <= 0) {
    printf("Error: Non-positive temperature = %g.\n", *temp);
    return 1;  // error: non-positive temperature
  }

  // check non-negative concentration
  for (int i = 0; i < nspecies; i++) {
    if (conc[i] < 0) {
      printf("Warning: Negative concentration for species %d = %g.\n", i,
             conc[i]);
      printf("Setting it to zero.\n");
      conc[i] = 0.;
      // return 1;  // error: negative concentration
    }
  }

  // check dimensions
  if (nspecies <= 0 || nreaction <= 0) {
    printf("Error: nspecies and nreaction must be positive integers.\n");
    return 1;  // error: invalid dimensions
  }

  // check non-negative cp
  for (int i = 0; i < nspecies; i++) {
    if (cv_const[i] < 0) {
      printf("Error: Negative heat capacity for species %d.\n", i);
      return 1;  // error: negative heat capacity
    }
  }

  T *intEng, *intEng_ddT, *logsvp, *logsvp_ddT, *weight, *rhs;
  T *stoich_active, *conc0;
  T *gain_cpy;

  if (work == nullptr) {
    intEng = (T *)malloc(nspecies * sizeof(T));
    intEng_ddT = (T *)malloc(nspecies * sizeof(T));
    logsvp = (T *)malloc(nreaction * sizeof(T));
    logsvp_ddT = (T *)malloc(nreaction * sizeof(T));

    // weight matrix
    weight = (T *)malloc(nreaction * nspecies * sizeof(T));

    // right-hand-side vector
    rhs = (T *)malloc(nreaction * sizeof(T));

    // active stoichiometric matrix
    stoich_active = (T *)malloc(nspecies * nreaction * sizeof(T));

    // concentration copy
    conc0 = (T *)malloc(nspecies * sizeof(T));

    // gain matrix copy
    gain_cpy = (T *)malloc(nreaction * nreaction * sizeof(T));
  } else {
    intEng = alloc_from<T>(work, nspecies);
    intEng_ddT = alloc_from<T>(work, nspecies);
    logsvp = alloc_from<T>(work, nreaction);
    logsvp_ddT = alloc_from<T>(work, nreaction);
    weight = alloc_from<T>(work, nreaction * nspecies);
    rhs = alloc_from<T>(work, nreaction);
    stoich_active = alloc_from<T>(work, nspecies * nreaction);
    conc0 = alloc_from<T>(work, nspecies);
    gain_cpy = alloc_from<T>(work, nreaction * nreaction);
  }

  memset(weight, 0, nreaction * nspecies * sizeof(T));
  memset(rhs, 0, nreaction * sizeof(T));

  // evaluate internal energy and its derivative (cv)
  for (int i = 0; i < nspecies; i++) {
    intEng[i] = intEng_offset[i] + cv_const[i] * (*temp);
    if (intEng_R_extra[i]) {
      intEng[i] += intEng_R_extra[i](*temp, conc[i]) * constants::Rgas;
    }
    intEng_ddT[i] = cv_const[i];
    if (cv_R_extra[i]) {
      intEng_ddT[i] += cv_R_extra[i](*temp, conc[i]) * constants::Rgas;
    }
  }

  int iter = 0;
  int err_code = 0;
  while (iter++ < *max_iter) {
    /*printf("iteration %d: T = %g\n", iter, *temp);
    // print conc
    printf("concentrations: ");
    for (int i = 0; i < nspecies; i++) {
      printf("%g ", conc[i]);
    }
    printf("\n");*/

    // evaluate log vapor saturation pressure and its derivative
    for (int j = 0; j < nreaction; j++) {
      T stoich_sum = 0.0;
      for (int i = 0; i < nspecies; i++)
        if (stoich[i * nreaction + j] < 0) {  // reactant
          stoich_sum += (-stoich[i * nreaction + j]);
        }
      logsvp[j] =
          logsvp_func[j](*temp) - stoich_sum * log(constants::Rgas * (*temp));
      logsvp_ddT[j] = logsvp_func_ddT[j](*temp) - stoich_sum / (*temp);
    }

    // calculate heat capacity
    T heat_capacity = 0.0;
    for (int i = 0; i < nspecies; i++) {
      heat_capacity += intEng_ddT[i] * conc[i];
    }

    // populate weight matrix, rhs vector and active set
    int first = 0;
    int last = nreaction;
    while (first < last) {
      int j = reaction_set[first];
      T log_conc_sum = 0.0;
      T prod = 1.0;

      // active set condition variables
      for (int i = 0; i < nspecies; i++) {
        if (stoich[i * nreaction + j] < 0) {  // reactant
          if (conc[i] == 0.) {
            log_conc_sum = -99;  // force to be in active set
          } else {
            log_conc_sum += (-stoich[i * nreaction + j]) * log(conc[i]);
          }
        } else if (stoich[i * nreaction + j] > 0) {  // product
          prod *= conc[i];
        }
      }

      // active set, weight matrix and rhs vector
      if ((log_conc_sum < (logsvp[j] - logsvp_eps) && prod > 0.) ||
          (log_conc_sum > (logsvp[j] + logsvp_eps))) {
        for (int i = 0; i < nspecies; i++) {
          weight[first * nspecies + i] =
              logsvp_ddT[j] * intEng[i] / heat_capacity;
          if (stoich[i * nreaction + j] < 0) {
            if (conc[i] == 0.) {
              weight[first * nspecies + i] += 1.e5;
            } else {
              weight[first * nspecies + i] +=
                  (-stoich[i * nreaction + j]) / conc[i];
            }
          }
        }
        rhs[first] = logsvp[j] - log_conc_sum;
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
    (*nactive) = first;
    for (int i = 0; i < nspecies; i++)
      for (int k = 0; k < (*nactive); k++) {
        int j = reaction_set[k];
        stoich_active[i * (*nactive) + k] = stoich[i * nreaction + j];
      }

    mmdot(gain, weight, stoich_active, *nactive, nspecies, *nactive);

    for (int i = 0; i < nspecies; i++)
      for (int k = 0; k < (*nactive); k++) {
        stoich_active[i * (*nactive) + k] *= -1;
      }
    // note that stoich_active is negated

    // solve constrained optimization problem (KKT)
    int max_kkt_iter = *max_iter;
    err_code = leastsq_kkt(rhs, gain, stoich_active, conc, *nactive, *nactive,
                           nspecies, 0, &max_kkt_iter, -1.e-10, work);
    if (err_code != 0) break;

    // rate -> conc
    memcpy(conc0, conc, nspecies * sizeof(T));
    T lambda = 1.;  // scale
    while (true) {
      bool good = true;
      for (int i = 0; i < nspecies; i++) {
        for (int k = 0; k < (*nactive); k++) {
          conc[i] -= stoich_active[i * (*nactive) + k] * rhs[k] * lambda;
        }
        if ((i < ngas) && (conc0[i] > 0.) &&
            ((conc[i] / conc0[i] > 100.) || (conc[i] / conc0[i] < 0.01)))
          good = false;
      }
      if (good) break;
      lambda *= 0.99;
      memcpy(conc, conc0, nspecies * sizeof(T));
    }

    // temperature iteration
    T temp0 = 0.;
    while (fabs(*temp - temp0) > 1e-4) {
      T zh = 0.;
      T zc = 0.;

      // re-evaluate internal energy and its derivative
      for (int i = 0; i < nspecies; i++) {
        intEng[i] = intEng_offset[i] + cv_const[i] * (*temp);
        if (intEng_R_extra[i]) {
          intEng[i] += intEng_R_extra[i](*temp, conc[i]) * constants::Rgas;
        }
        intEng_ddT[i] = cv_const[i];
        if (cv_R_extra[i]) {
          intEng_ddT[i] += cv_R_extra[i](*temp, conc[i]) * constants::Rgas;
        }
        zh += intEng[i] * conc[i];
        zc += intEng_ddT[i] * conc[i];
      }

      temp0 = *temp;
      (*temp) += (h0 - zh) / zc;
    }

    if (*temp <= 0.) {
      printf("Error: Non-positive temperature after adjustment.\n");
      err_code = 3;  // error: non-positive temperature after adjustment
      break;
    }
  }

  // restore the reaction order of gain
  memcpy(gain_cpy, gain, nreaction * nreaction * sizeof(T));
  memset(gain, 0, nreaction * nreaction * sizeof(T));

  for (int i = 0; i < (*nactive); i++) {
    for (int j = 0; j < nreaction; j++) {
      int k = reaction_set[i];
      int l = reaction_set[j];
      gain[k * nreaction + l] = gain_cpy[i * nreaction + j];
    }
  }

  // save number of iterations to diag
  diag[0] = iter;

  if (work == nullptr) {
    free(intEng);
    free(intEng_ddT);
    free(logsvp);
    free(logsvp_ddT);
    free(weight);
    free(rhs);
    free(stoich_active);
    free(gain_cpy);
  }

  if (iter >= *max_iter) {
    printf("equilibrate_uv did not converge after %d iterations.\n", *max_iter);
    return 2 * 10 + err_code;  // failure to converge
  } else {
    *max_iter = iter;
    return err_code;  // success or KKT error
  }
}

}  // namespace kintera
