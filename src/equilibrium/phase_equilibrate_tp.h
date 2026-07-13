#pragma once

#include <configure.h>
#include <kintera/math/core.h>
#include <kintera/math/leastsq_kkt.h>
#include <kintera/utils/alloc.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace kintera {

template <typename T>
DISPATCH_MACRO T equilibrium_max_error(T const *moles, T pres,
                                       T standard_pressure, T const *log_k,
                                       T const *stoich, int const *phase_ids,
                                       int nspecies, int nreaction, int nphase,
                                       int gas_phase, T *phase_totals,
                                       T *residual) {
  for (int p = 0; p < nphase; ++p) phase_totals[p] = 0.;
  for (int i = 0; i < nspecies; ++i) {
    phase_totals[phase_ids[i]] += moles[i];
  }

  for (int j = 0; j < nreaction; ++j) residual[j] = -log_k[j];
  for (int i = 0; i < nspecies; ++i) {
    T log_activity = log(moles[i] / phase_totals[phase_ids[i]]);
    if (phase_ids[i] == gas_phase) {
      log_activity += log(pres / standard_pressure);
    }
    for (int j = 0; j < nreaction; ++j) {
      residual[j] += stoich[i * nreaction + j] * log_activity;
    }
  }

  T max_error = 0.;
  for (int j = 0; j < nreaction; ++j) {
    T value = fabs(residual[j]);
    if (value > max_error) max_error = value;
  }
  return max_error;
}

template <typename T>
DISPATCH_MACRO int phase_equilibrate_tp(
    T *gain, T *diag, T *out_moles, T temp, T pres, T const *in_moles,
    T const *log_k, T const *stoich, int const *phase_ids, int nspecies,
    int nreaction, int nphase, int gas_phase, T standard_pressure, T ftol,
    T mole_floor, int max_iter, char *work = nullptr) {
  if (!(temp > 0.) || !(pres > 0.) || nspecies <= 0 || nreaction <= 0 ||
      nphase <= 0 || gas_phase < 0 || gas_phase >= nphase) {
    diag[0] = 1.;
    return 1;
  }
  for (int i = 0; i < nspecies; ++i) {
    if (!(in_moles[i] > mole_floor)) {
      diag[0] = 1.;
      return 1;
    }
  }

  T *phase_totals, *phase_stoich, *residual, *jac, *newton_jac, *constraints;
  T *bounds, *step, *trial;
  bool own_work = work == nullptr;
  if (own_work) {
    phase_totals = (T *)malloc(nphase * sizeof(T));
    phase_stoich = (T *)malloc(nphase * nreaction * sizeof(T));
    residual = (T *)malloc(nreaction * sizeof(T));
    jac = (T *)malloc(nreaction * nreaction * sizeof(T));
    newton_jac = (T *)malloc(nreaction * nreaction * sizeof(T));
    constraints = (T *)malloc(nspecies * nreaction * sizeof(T));
    bounds = (T *)malloc(nspecies * sizeof(T));
    step = (T *)malloc(nreaction * sizeof(T));
    trial = (T *)malloc(nspecies * sizeof(T));
  } else {
    phase_totals = alloc_from<T>(work, nphase);
    phase_stoich = alloc_from<T>(work, nphase * nreaction);
    residual = alloc_from<T>(work, nreaction);
    jac = alloc_from<T>(work, nreaction * nreaction);
    newton_jac = alloc_from<T>(work, nreaction * nreaction);
    constraints = alloc_from<T>(work, nspecies * nreaction);
    bounds = alloc_from<T>(work, nspecies);
    step = alloc_from<T>(work, nreaction);
    trial = alloc_from<T>(work, nspecies);
  }

  memcpy(out_moles, in_moles, nspecies * sizeof(T));
  memset(jac, 0, nreaction * nreaction * sizeof(T));
  memset(phase_stoich, 0, nphase * nreaction * sizeof(T));
  for (int i = 0; i < nspecies; ++i) {
    for (int j = 0; j < nreaction; ++j) {
      phase_stoich[phase_ids[i] * nreaction + j] += stoich[i * nreaction + j];
    }
  }
  T target = fmax(log1p(ftol), 4. * std::numeric_limits<T>::epsilon());
  T max_error = 0.;
  int status = 2;
  int iter = 0;
  for (; iter < max_iter; ++iter) {
    max_error = equilibrium_max_error(
        out_moles, pres, standard_pressure, log_k, stoich, phase_ids, nspecies,
        nreaction, nphase, gas_phase, phase_totals, residual);
    if (max_error <= target) {
      status = 0;
      break;
    }

    // Jacobian d(residual)/d(reaction extent).
    for (int j = 0; j < nreaction; ++j) {
      for (int k = 0; k < nreaction; ++k) {
        T value = 0.;
        for (int i = 0; i < nspecies; ++i) {
          int phase = phase_ids[i];
          T phase_delta = phase_stoich[phase * nreaction + k];
          value += stoich[i * nreaction + j] *
                   (stoich[i * nreaction + k] / out_moles[i] -
                    phase_delta / phase_totals[phase]);
        }
        jac[j * nreaction + k] = value;
      }
      step[j] = -residual[j];
    }

    // -S dx <= n - floor is equivalent to n + S dx >= floor.
    for (int i = 0; i < nspecies; ++i) {
      bounds[i] = out_moles[i] - mole_floor;
      for (int j = 0; j < nreaction; ++j) {
        constraints[i * nreaction + j] = -stoich[i * nreaction + j];
      }
    }

    // The reaction Jacobian is square. Solve the Newton system directly before
    // falling back to constrained least squares. Forming J^T J in the KKT
    // solver squares the condition number and loses useful precision for
    // multiphase systems spanning many orders of magnitude in abundance.
    memcpy(newton_jac, jac, nreaction * nreaction * sizeof(T));
    bool direct_ok = dsolve_lu(newton_jac, step, nreaction);
    for (int j = 0; j < nreaction && direct_ok; ++j) {
      if (!isfinite(step[j])) direct_ok = false;
    }

    bool accepted = false;
    T scale = 1.;
    while (direct_ok && scale >= 1.e-12) {
      bool positive = true;
      for (int i = 0; i < nspecies; ++i) {
        T delta = 0.;
        for (int j = 0; j < nreaction; ++j) {
          delta += stoich[i * nreaction + j] * step[j];
        }
        trial[i] = out_moles[i] + scale * delta;
        if (!(trial[i] > mole_floor)) positive = false;
      }
      if (positive) {
        T trial_error = equilibrium_max_error(
            trial, pres, standard_pressure, log_k, stoich, phase_ids, nspecies,
            nreaction, nphase, gas_phase, phase_totals, residual);
        if (trial_error < max_error) {
          memcpy(out_moles, trial, nspecies * sizeof(T));
          accepted = true;
          break;
        }
      }
      scale *= .5;
    }
    if (accepted) continue;

    // A direct Newton direction can fail close to an active mole bound. Retain
    // the active-set KKT solve as the bounded fallback for those cases.
    for (int j = 0; j < nreaction; ++j) step[j] = -residual[j];
    int kkt_iter = max_iter;
    int err = leastsq_kkt(step, jac, constraints, bounds, nreaction, nreaction,
                          nspecies, 0, &kkt_iter, 1.e-12, work);
    if (err != 0) {
      status = 3;
      break;
    }

    scale = 1.;
    while (scale >= 1.e-12) {
      bool positive = true;
      for (int i = 0; i < nspecies; ++i) {
        T delta = 0.;
        for (int j = 0; j < nreaction; ++j) {
          delta += stoich[i * nreaction + j] * step[j];
        }
        trial[i] = out_moles[i] + scale * delta;
        if (!(trial[i] > mole_floor)) positive = false;
      }
      if (positive) {
        T trial_error = equilibrium_max_error(
            trial, pres, standard_pressure, log_k, stoich, phase_ids, nspecies,
            nreaction, nphase, gas_phase, phase_totals, residual);
        if (trial_error < max_error) {
          memcpy(out_moles, trial, nspecies * sizeof(T));
          accepted = true;
          break;
        }
      }
      scale *= .5;
    }
    if (!accepted) {
      status = 4;
      break;
    }
  }

  max_error = equilibrium_max_error(out_moles, pres, standard_pressure, log_k,
                                    stoich, phase_ids, nspecies, nreaction,
                                    nphase, gas_phase, phase_totals, residual);
  memcpy(gain, jac, nreaction * nreaction * sizeof(T));

  diag[0] = static_cast<T>(status);
  diag[1] = static_cast<T>(iter);
  diag[2] = exp(max_error) - 1.;

  if (own_work) {
    free(phase_totals);
    free(phase_stoich);
    free(residual);
    free(jac);
    free(newton_jac);
    free(constraints);
    free(bounds);
    free(step);
    free(trial);
  }
  return status;
}

}  // namespace kintera
