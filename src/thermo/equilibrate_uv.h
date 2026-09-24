#pragma once

// C/C++
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

// base
#include <configure.h>

// kintera
#include <kintera/constants.h>
#include <kintera/math/core.h>
#include <kintera/math/leastsq_kkt.h>

#include <kintera/utils/user_funcs.hpp>

#include "svp_eval.h"

namespace kintera {

template <typename T>
DISPATCH_MACRO bool is_depleted_cloud(T amount, T reference) {
  return reference > 0. && amount >= 0. &&
         amount <= 8. * std::numeric_limits<T>::epsilon() * reference;
}

template <typename T>
DISPATCH_MACRO bool partition_uv_state(
    T temperature, T* conc, T const* baseline, T const* totals, T* molar_energy,
    T const* stoich, int nspecies, int nreaction, T const* intEng_offset,
    T const* cv_const, user_func1 const* logsvp_func,
    user_func1 const* logsvp_func_ddT, int const* svp_kind,
    double const* svp_params, user_func2 const* intEng_R_extra,
    user_func2 const* cv_R_extra, T* energy, T* derivative) {
  memcpy(conc, baseline, nspecies * sizeof(T));
  for (int reaction_index = 0; reaction_index < nreaction; ++reaction_index) {
    int vapor_index = -1;
    int cloud_index = -1;
    T reactant_coefficient = 0.;
    T product_coefficient = 0.;
    for (int species_index = 0; species_index < nspecies; ++species_index) {
      T coefficient = stoich[species_index * nreaction + reaction_index];
      if (coefficient < 0.) {
        vapor_index = species_index;
        reactant_coefficient = -coefficient;
      } else if (coefficient > 0.) {
        cloud_index = species_index;
        product_coefficient = coefficient;
      }
    }
    T total = totals[reaction_index];
    if (total == 0.) {
      conc[vapor_index] = 0.;
      conc[cloud_index] = 0.;
      continue;
    }
    T log_saturation = eval_logsvp(svp_kind[reaction_index],
                                   svp_params + reaction_index * KSVP_NPARAM,
                                   logsvp_func[reaction_index], temperature) /
                           reactant_coefficient -
                       log(constants::Rgas * temperature);
    if (!std::isfinite(log_saturation)) return false;
    T vapor = log_saturation >= log(total) ? total : exp(log_saturation);
    if (vapor > total) vapor = total;
    conc[vapor_index] = vapor;
    conc[cloud_index] =
        (total - vapor) * product_coefficient / reactant_coefficient;
  }

  *energy = 0.;
  *derivative = 0.;
  for (int species_index = 0; species_index < nspecies; ++species_index) {
    molar_energy[species_index] =
        intEng_offset[species_index] + cv_const[species_index] * temperature;
    T molar_cv = cv_const[species_index];
    if (intEng_R_extra[species_index]) {
      molar_energy[species_index] +=
          intEng_R_extra[species_index](temperature, conc[species_index]) *
          constants::Rgas;
    }
    if (cv_R_extra[species_index]) {
      molar_cv += cv_R_extra[species_index](temperature, conc[species_index]) *
                  constants::Rgas;
    }
    *energy += molar_energy[species_index] * conc[species_index];
    *derivative += molar_cv * conc[species_index];
  }

  for (int reaction_index = 0; reaction_index < nreaction; ++reaction_index) {
    int vapor_index = -1;
    int cloud_index = -1;
    T reactant_coefficient = 0.;
    T product_coefficient = 0.;
    for (int species_index = 0; species_index < nspecies; ++species_index) {
      T coefficient = stoich[species_index * nreaction + reaction_index];
      if (coefficient < 0.) {
        vapor_index = species_index;
        reactant_coefficient = -coefficient;
      } else if (coefficient > 0.) {
        cloud_index = species_index;
        product_coefficient = coefficient;
      }
    }
    if (conc[vapor_index] >= totals[reaction_index]) continue;
    T vapor_derivative =
        conc[vapor_index] *
        (eval_logsvp_ddT(svp_kind[reaction_index],
                         svp_params + reaction_index * KSVP_NPARAM,
                         logsvp_func_ddT[reaction_index], temperature) /
             reactant_coefficient -
         1. / temperature);
    *derivative += vapor_derivative *
                   (molar_energy[vapor_index] - product_coefficient /
                                                    reactant_coefficient *
                                                    molar_energy[cloud_index]);
  }
  return std::isfinite(*energy) && std::isfinite(*derivative);
}

template <typename T, PoolBackend Backend = PoolBackend::Shared>
DISPATCH_MACRO int equilibrate_uv_partition(
    T* gain, T* diag, T* temp, T* conc, T h0, T const* stoich, int nspecies,
    int nreaction, T const* intEng_offset, T const* cv_const,
    user_func1 const* logsvp_func, user_func1 const* logsvp_func_ddT,
    int const* svp_kind, double const* svp_params,
    user_func2 const* intEng_R_extra, user_func2 const* cv_R_extra,
    int* max_iter, int* nactive, char* work) {
  size_t mark = pool_mark<Backend>(work);
  T* baseline = (T*)pmalloc<Backend>(work, nspecies * sizeof(T));
  T* molar_energy = (T*)pmalloc<Backend>(work, nspecies * sizeof(T));
  T* totals = (T*)pmalloc<Backend>(work, nreaction * sizeof(T));
  memcpy(baseline, conc, nspecies * sizeof(T));
  T original_temperature = *temp;
  bool valid = true;

  for (int reaction_index = 0; reaction_index < nreaction; ++reaction_index) {
    int vapor_index = -1;
    int cloud_index = -1;
    T reactant_coefficient = 0.;
    T product_coefficient = 0.;
    for (int species_index = 0; species_index < nspecies; ++species_index) {
      T coefficient = stoich[species_index * nreaction + reaction_index];
      if (coefficient < 0.) {
        vapor_index = species_index;
        reactant_coefficient = -coefficient;
      } else if (coefficient > 0.) {
        cloud_index = species_index;
        product_coefficient = coefficient;
      }
    }
    totals[reaction_index] = baseline[vapor_index] + reactant_coefficient /
                                                         product_coefficient *
                                                         baseline[cloud_index];
    if (!std::isfinite(totals[reaction_index]) || totals[reaction_index] < 0.)
      valid = false;
  }

  T energy = 0.;
  T derivative = 0.;
  T residual = 0.;
  T energy_scale = fabs(h0) > 1. ? fabs(h0) : 1.;
  T tolerance = 64. * std::numeric_limits<T>::epsilon() * energy_scale;
  int iterations = 1;
  if (valid) {
    valid = partition_uv_state(
        *temp, conc, baseline, totals, molar_energy, stoich, nspecies,
        nreaction, intEng_offset, cv_const, logsvp_func, logsvp_func_ddT,
        svp_kind, svp_params, intEng_R_extra, cv_R_extra, &energy, &derivative);
    residual = energy - h0;
  }

  if (valid && fabs(residual) > tolerance) {
    T lower = *temp;
    T upper = *temp;
    T bracket_energy = energy;
    T bracket_derivative = derivative;
    bool bracketed = false;
    for (int bracket_step = 0; bracket_step < 64; ++bracket_step) {
      T candidate = residual > 0. ? lower * 0.5 : upper * 2.;
      if (!(candidate > 0.) || !std::isfinite(candidate)) break;
      if (!partition_uv_state(candidate, conc, baseline, totals, molar_energy,
                              stoich, nspecies, nreaction, intEng_offset,
                              cv_const, logsvp_func, logsvp_func_ddT, svp_kind,
                              svp_params, intEng_R_extra, cv_R_extra,
                              &bracket_energy, &bracket_derivative))
        break;
      if (residual > 0.) {
        lower = candidate;
        bracketed = bracket_energy <= h0;
      } else {
        upper = candidate;
        bracketed = bracket_energy >= h0;
      }
      if (bracketed) break;
    }
    valid = bracketed;

    T current = original_temperature;
    while (valid && fabs(residual) > tolerance && iterations < *max_iter) {
      T candidate = current - residual / derivative;
      if (!std::isfinite(candidate) || candidate <= lower ||
          candidate >= upper) {
        candidate = lower + 0.5 * (upper - lower);
      }
      if (candidate == current) candidate = lower + 0.5 * (upper - lower);
      if (candidate == current) break;
      valid = partition_uv_state(
          candidate, conc, baseline, totals, molar_energy, stoich, nspecies,
          nreaction, intEng_offset, cv_const, logsvp_func, logsvp_func_ddT,
          svp_kind, svp_params, intEng_R_extra, cv_R_extra, &energy,
          &derivative);
      if (!valid) break;
      current = candidate;
      residual = energy - h0;
      if (residual < 0.)
        lower = current;
      else
        upper = current;
      ++iterations;
    }
    *temp = current;
  }

  int status = valid && fabs(residual) <= tolerance ? 0 : -1;
  if (status == 0) {
    memset(gain, 0, nreaction * nreaction * sizeof(T));
    diag[0] = iterations;
    *nactive = 0;
    *max_iter = iterations;
  } else {
    memcpy(conc, baseline, nspecies * sizeof(T));
    *temp = original_temperature;
    memset(gain, 0, nreaction * nreaction * sizeof(T));
    diag[0] = -1.;
    *nactive = 0;
  }
  pfree<Backend>(baseline);
  pfree<Backend>(molar_energy);
  pfree<Backend>(totals);
  pool_rewind<Backend>(work, mark);
  return status;
}

/*!
 * \brief Calculate thermodynamic equilibrium at fixed volume and internal
 * energy
 *
 * Given an initial guess of temperature and concentrations, this function
 * adjusts the temperature and concentrations to satisfy the saturation
 * condition.
 *
 * \param[out] gain             WS gain matrix
 * \param[out] diag             iterations, or -(100 * status + iterations)
 *                              when the return status is nonzero
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
template <typename T, PoolBackend Backend = PoolBackend::Shared>
DISPATCH_MACRO int equilibrate_uv(
    T* gain, T* diag, T* temp, T* conc, T h0, T const* stoich, int nspecies,
    int nreaction, int ngas, T const* intEng_offset, T const* cv_const,
    user_func1 const* logsvp_func, user_func1 const* logsvp_func_ddT,
    int const* svp_kind, double const* svp_params,
    user_func2 const* intEng_R_extra, user_func2 const* cv_R_extra,
    float logsvp_eps, int* max_iter, int* reaction_set, int* nactive,
    int uv_solver = 0, char* work = nullptr) {
  // check positive temperature
  if (*temp <= 0) {
    printf("Error: Non-positive temperature = %g.\n", *temp);
    diag[0] = -100.;
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
    diag[0] = -100.;
    return 1;  // error: invalid dimensions
  }

  // check non-negative cp
  for (int i = 0; i < nspecies; i++) {
    if (cv_const[i] < 0) {
      printf("Error: Negative heat capacity for species %d.\n", i);
      diag[0] = -100.;
      return 1;  // error: negative heat capacity
    }
  }

  if (uv_solver != 0) {
    int partition_status = equilibrate_uv_partition<T, Backend>(
        gain, diag, temp, conc, h0, stoich, nspecies, nreaction, intEng_offset,
        cv_const, logsvp_func, logsvp_func_ddT, svp_kind, svp_params,
        intEng_R_extra, cv_R_extra, max_iter, nactive, work);
    if (partition_status == 0) return 0;
    if (uv_solver == 2) {
      printf("[Warning] equilibrate_uv partition did not converge.\n");
      return 2;
    }
  }

  T *intEng, *intEng_ddT, *logsvp, *logsvp_ddT, *weight, *rhs;
  T *stoich_active, *conc0;
  T* gain_cpy;
  T* theta;

  size_t mark = pool_mark<Backend>(work);
  intEng = (T*)pmalloc<Backend>(work, nspecies * sizeof(T));
  intEng_ddT = (T*)pmalloc<Backend>(work, nspecies * sizeof(T));
  logsvp = (T*)pmalloc<Backend>(work, nreaction * sizeof(T));
  logsvp_ddT = (T*)pmalloc<Backend>(work, nreaction * sizeof(T));
  weight = (T*)pmalloc<Backend>(work, nreaction * nspecies * sizeof(T));
  rhs = (T*)pmalloc<Backend>(work, nreaction * sizeof(T));
  stoich_active = (T*)pmalloc<Backend>(work, nspecies * nreaction * sizeof(T));
  conc0 = (T*)pmalloc<Backend>(work, nspecies * sizeof(T));
  gain_cpy = (T*)pmalloc<Backend>(work, nreaction * nreaction * sizeof(T));
  theta = (T*)pmalloc<Backend>(work, nspecies * sizeof(T));

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
  bool converged = false;
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
      double const* p = svp_params + j * KSVP_NPARAM;
      logsvp[j] = eval_logsvp(svp_kind[j], p, logsvp_func[j], *temp) -
                  stoich_sum * log(constants::Rgas * (*temp));
      logsvp_ddT[j] =
          eval_logsvp_ddT(svp_kind[j], p, logsvp_func_ddT[j], *temp) -
          stoich_sum / (*temp);
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
      T nu_absent = 0.;  // stoichiometry of the reactants that are absent
      T log_nu = 0.;

      // active set condition variables
      for (int i = 0; i < nspecies; i++) {
        T nu = -stoich[i * nreaction + j];
        if (nu > 0) {  // reactant
          if (conc[i] == 0.) {
            nu_absent += nu;
            log_nu += nu * log(nu);
          } else {
            log_conc_sum += nu * log(conc[i]);
          }
        } else if (nu < 0) {  // product
          prod *= conc[i];
        }
      }

      // Absent reactants are linearized at their saturation value nu*r*,
      // restored from the product; rhs = nu_absent lands the step there.
      T log_r = 0.;
      if (nu_absent > 0.) {
        log_r = (logsvp[j] - log_conc_sum - log_nu) / nu_absent;
        // keep the weight 1/r* finite: it overflows float for a cold enough svp
        T log_r_min = -0.5 * log(std::numeric_limits<T>::max());
        if (log_r < log_r_min) log_r = log_r_min;
        log_conc_sum = logsvp[j] - nu_absent;
      }

      // active set, weight matrix and rhs vector
      if ((log_conc_sum < (logsvp[j] - logsvp_eps) && prod > 0.) ||
          (log_conc_sum > (logsvp[j] + logsvp_eps))) {
        for (int i = 0; i < nspecies; i++) {
          weight[first * nspecies + i] =
              logsvp_ddT[j] * intEng[i] / heat_capacity;
          T nu = -stoich[i * nreaction + j];
          if (nu > 0) {
            weight[first * nspecies + i] +=
                conc[i] == 0. ? exp(-log_r) : nu / conc[i];
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

    // (*nactive) sizes the gain scatter at exit; a converged solve has none
    (*nactive) = first;

    if (first == 0) {
      // all reactions are in equilibrium, no need to adjust saturation
      converged = true;
      break;
    }

    // form active stoichiometric and constraint matrix
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
    // The inner active-set solve needs its own bound: sharing max_iter with
    // the outer Newton lets a small budget abort it and return state unchanged.
    int max_kkt_iter = nspecies + 1 > *max_iter ? nspecies + 1 : *max_iter;
    // x = 0 (no extent) is feasible: the bounds are the current concentrations
    err_code = leastsq_kkt_feasible_origin<T, Backend>(
        rhs, gain, stoich_active, conc, *nactive, *nactive, nspecies, 0,
        &max_kkt_iter, 0., work);
    if (err_code != 0) break;

    // rate -> conc
    memcpy(conc0, conc, nspecies * sizeof(T));
    T lambda = 1.;  // scale
    // Per-reaction extent limit: clip each reaction to the stock
    // of what it consumes. No production credit, no boundary factor.
    for (int i = 0; i < nspecies; i++) {
      T demand = 0.;
      for (int k = 0; k < (*nactive); k++) {
        // stoich_active is negated: species i changes by -sa[i][k]*rhs[k]
        T ck = -stoich_active[i * (*nactive) + k] * rhs[k];
        if (ck < 0.) demand -= ck;
      }
      // conc0 is re-copied each iteration, so it is not clamped non-negative;
      // a negative must give 0, not a scale that flips the step.
      if (demand > conc0[i]) {
        theta[i] = conc0[i] > 0. ? conc0[i] / demand : 0.;
      } else {
        theta[i] = 1.;
      }
    }
    for (int k = 0; k < (*nactive); k++) {
      T lam = 1.;
      for (int i = 0; i < nspecies; i++) {
        T ck = -stoich_active[i * (*nactive) + k] * rhs[k];
        if (ck < 0. && theta[i] < lam) lam = theta[i];
      }
      rhs[k] *= lam;
    }
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
      err_code = 4;  // error: non-positive temperature after adjustment
      break;
    }
  }

  // restore the reaction order of gain
  memcpy(gain_cpy, gain, nreaction * nreaction * sizeof(T));
  memset(gain, 0, nreaction * nreaction * sizeof(T));

  // mmdot wrote gain as (*nactive) x (*nactive), so the copy's leading
  // dimension is (*nactive), not nreaction (cf. equilibrate_tp, #105)
  for (int i = 0; i < (*nactive); i++) {
    for (int j = 0; j < (*nactive); j++) {
      int k = reaction_set[i];
      int l = reaction_set[j];
      gain[k * nreaction + l] = gain_cpy[i * (*nactive) + j];
    }
  }

  int n_iter = iter > *max_iter ? *max_iter : iter;
  int status = err_code ? err_code : (converged ? 0 : 2 * 10);
  // diag = iterations, or -(100 * status + iterations) on failure
  diag[0] = status ? -(100. * status + n_iter) : n_iter;

  pfree<Backend>(intEng);
  pfree<Backend>(intEng_ddT);
  pfree<Backend>(logsvp);
  pfree<Backend>(logsvp_ddT);
  pfree<Backend>(weight);
  pfree<Backend>(rhs);
  pfree<Backend>(stoich_active);
  pfree<Backend>(conc0);
  pfree<Backend>(gain_cpy);
  pfree<Backend>(theta);
  pool_rewind<Backend>(work, mark);

  if (status == 2 * 10) {
    printf("[Warning] equilibrate_uv did not converge after %d iterations.\n",
           *max_iter);
  } else {
    *max_iter = n_iter;
  }
  return status;
}

}  // namespace kintera
