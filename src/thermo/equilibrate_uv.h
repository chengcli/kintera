// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>

// kintera
#include <kintera/constants.h>
#include <kintera/math/leastsq_kkt.h>
#include <kintera/math/mmdot.h>

#include <kintera/utils/func1.hpp>

namespace kintera {

/*!
 * \brief Calculate thermodynamic equilibrium at fixed volume and internal
 * energy
 *
 * Given an initial guess of temperature and concentrations, this function
 * adjusts the temperature and concentrations to satisfy the saturation
 * condition.
 *
 * \param[in,out] temp in: initial temperature, out: adjusted temperature.
 * \param[in,out] conc in: initial concentrations for each species, out:
 * adjusted concentrations.
 * \param[in] h0 initial enthalpy.
 * \param[in] stoich reaction stoichiometric matrix, nspecies x nreaction.
 * \param[in] nspecies number of species in the system.
 * \param[in] nreaction number of reactions in the system.
 * \param[in] enthalpy_offset offset for enthalpy calculations.
 * \param[in] cp_const const component of heat capacity.
 * \param[in] logsvp_func user-defined functions for logarithm of saturation
 * vapor pressure.
 * \param[in] logsvp_func_ddT user-defined functions for derivative of logsvp
 *            with respect to temperature.
 * \param[in] enthalpy_extra user-defined functions for enthalpy calculation
 *            in addition to the linear term.
 * \param[in] enthalpy_extra_ddT user-defined functions for enthalpy derivative
 *            with respect to temperature in addition to the constant term.
 * \param[in] lnsvp_eps tolerance for convergence in logarithm of saturation
 * vapor pressure.
 * \param[in,out] max_iter maximum number of iterations allowed for convergence.
 */
template <typename T>
int equilibrate_uv(T *temp, T *conc, T h0, T const *stoich, int nspecies,
                   int nreaction, T const *enthalpy_offset, T const *cp_const,
                   user_func1 const *logsvp_func,
                   user_func1 const *logsvp_func_ddT,
                   user_func1 const *enthalpy_extra,
                   user_func1 const *enthalpy_extra_ddT, float logsvp_eps,
                   int *max_iter) {
  // check positive temperature
  if (*temp <= 0) {
    fprintf(stderr, "Error: Non-positive temperature.\n");
    return 1;  // error: non-positive temperature
  }

  // check non-negative concentration
  for (int i = 0; i < nspecies; i++) {
    if (conc[i] < 0) {
      fprintf(stderr, "Error: Negative concentration for species %d.\n", i);
      return 1;  // error: negative concentration
    }
  }

  // check dimensions
  if (nspecies <= 0 || nreaction <= 0) {
    fprintf(stderr,
            "Error: nspecies and nreaction must be positive integers.\n");
    return 1;  // error: invalid dimensions
  }

  // check non-negative cp
  for (int i = 0; i < nspecies; i++) {
    if (cp_const[i] < 0) {
      fprintf(stderr, "Error: Negative heat capacity for species %d.\n", i);
      return 1;  // error: negative heat capacity
    }
  }

  T *enthalpy = (T *)malloc(nspecies * sizeof(T));
  T *enthalpy_ddT = (T *)malloc(nspecies * sizeof(T));
  T *logsvp = (T *)malloc(nreaction * sizeof(T));
  T *logsvp_ddT = (T *)malloc(nreaction * sizeof(T));

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

  // evaluate enthalpy and its derivative
  for (int i = 0; i < nspecies; i++) {
    enthalpy[i] = enthalpy_offset[i] + cp_const[i] * (*temp);
    if (enthalpy_extra[i]) {
      enthalpy[i] += enthalpy_extra[i](*temp);
    }
    enthalpy_ddT[i] = cp_const[i];
    if (enthalpy_extra_ddT[i]) {
      enthalpy_ddT[i] += enthalpy_extra_ddT[i](*temp);
    }
  }

  // active stoichiometric matrix
  T *stoich_active = (T *)malloc(nspecies * nreaction * sizeof(T));

  int iter = 0;
  int err_code = 0;
  while (iter++ < *max_iter) {
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
      heat_capacity += enthalpy_ddT[i] * conc[i];
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
          log_conc_sum += (-stoich[i * nreaction + j]) * log(conc[i]);
        } else if (stoich[i * nreaction + j] > 0) {  // product
          prod *= conc[i];
        }
      }

      // active set, weight matrix and rhs vector
      if ((log_conc_sum < (logsvp[j] - logsvp_eps) && prod > 0.) ||
          (log_conc_sum > (logsvp[j] + logsvp_eps))) {
        for (int i = 0; i < nspecies; i++) {
          weight[first * nspecies + i] =
              logsvp_ddT[j] * enthalpy[i] / heat_capacity;
          if (stoich[i * nreaction + j] < 0) {
            weight[first * nspecies + i] +=
                (-stoich[i * nreaction + j]) / conc[i];
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
    err_code = leastsq_kkt(rhs, umat, stoich_active, conc, nactive, nactive,
                           nspecies, 0, &max_kkt_iter);
    if (err_code != 0) break;

    // rate -> conc
    for (int i = 0; i < nspecies; i++) {
      for (int k = 0; k < nactive; k++) {
        conc[i] -= stoich_active[i * nactive + k] * rhs[k];
      }
    }

    // temperature iteration
    T temp0 = 0.;
    while (fabs(*temp - temp0) > 1e-4) {
      T zh = 0.;
      T zc = 0.;

      // re-evaluate enthalpy and its derivative
      for (int i = 0; i < nspecies; i++) {
        enthalpy[i] = enthalpy_offset[i] + cp_const[i] * (*temp);
        if (enthalpy_extra[i]) {
          enthalpy[i] += enthalpy_extra[i](*temp);
        }
        enthalpy_ddT[i] = cp_const[i];
        if (enthalpy_extra_ddT[i]) {
          enthalpy_ddT[i] += enthalpy_extra_ddT[i](*temp);
        }
        zh += enthalpy[i] * conc[i];
        zc += enthalpy_ddT[i] * conc[i];
      }

      temp0 = *temp;
      (*temp) += (h0 - zh) / zc;
    }

    if (*temp <= 0.) {
      fprintf(stderr, "Error: Non-positive temperature after adjustment.\n");
      err_code = 3;  // error: non-positive temperature after adjustment
      break;
    }
  }

  free(enthalpy);
  free(enthalpy_ddT);
  free(logsvp);
  free(logsvp_ddT);
  free(weight);
  free(rhs);
  free(umat);
  free(reaction_set);
  free(stoich_active);

  if (iter >= *max_iter) {
    fprintf(stderr,
            "Saturation adjustment did not converge after %d iterations.\n",
            *max_iter);
    return 2;  // failure to converge
  } else {
    *max_iter = iter;
    return err_code;  // success or KKT error
  }
}

}  // namespace kintera
