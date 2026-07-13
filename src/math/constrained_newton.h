#pragma once

// C/C++
#include <cmath>
#include <cstdlib>
#include <cstring>

// base
#include <configure.h>

// kintera
#include <kintera/math/leastsq_kkt.h>
#include <kintera/utils/alloc.h>

namespace kintera {

/*!
 * \brief Solve a square Newton system, retaining KKT for active bounds.
 *
 * The direct solve avoids the loss of precision caused by forming A^T.A.  A
 * direction that violates a bound at unit length is still usable when a
 * positive scaling makes it feasible.  Only a direction pointing out of an
 * already-active bound is sent through constrained least squares.
 */
template <typename T>
DISPATCH_MACRO int constrained_newton_step(T* b, T const* a, T const* c,
                                           T const* d, int n, int nconstraint,
                                           int* max_iter, float reg = 0.,
                                           char* work = nullptr) {
  T *direct_a, *direct_b;
  if (work == nullptr) {
    direct_a = (T*)malloc(n * n * sizeof(T));
    direct_b = (T*)malloc(n * sizeof(T));
  } else {
    char* cursor = work;
    direct_a = alloc_from<T>(cursor, n * n);
    direct_b = alloc_from<T>(cursor, n);
  }

  memcpy(direct_a, a, n * n * sizeof(T));
  memcpy(direct_b, b, n * sizeof(T));
  bool usable = dsolve_lu(direct_a, direct_b, n);
  for (int j = 0; j < n && usable; ++j) {
    if (!std::isfinite(direct_b[j])) usable = false;
  }
  for (int i = 0; i < nconstraint && usable; ++i) {
    T value = 0.;
    for (int j = 0; j < n; ++j) value += c[i * n + j] * direct_b[j];
    if (!(d[i] > 0.) && value > 0.) usable = false;
  }

  if (usable) {
    memcpy(b, direct_b, n * sizeof(T));
    *max_iter = 1;
  }
  if (work == nullptr) {
    free(direct_a);
    free(direct_b);
  }
  if (usable) return 0;

  return leastsq_kkt(b, a, c, d, n, n, nconstraint, 0, max_iter, reg, work);
}

/*!
 * \brief Form a scaled Newton trial state and enforce simple bounds.
 *
 * `constraint` uses the same convention as leastsq_kkt, so the state update
 * is state - constraint.step.  The first `npositive` entries must remain
 * strictly above `floor`; all remaining entries must remain non-negative.
 * A positive `max_ratio` additionally limits relative changes of positive
 * entries in the first `nlimited` entries.
 */
template <typename T>
DISPATCH_MACRO bool constrained_newton_trial(T* trial, T const* state,
                                             T const* constraint, T const* step,
                                             int nstate, int nstep,
                                             int npositive, int nlimited,
                                             T scale, T floor = 0.,
                                             T max_ratio = 0.) {
  for (int i = 0; i < nstate; ++i) {
    T delta = 0.;
    for (int j = 0; j < nstep; ++j) {
      delta -= constraint[i * nstep + j] * step[j];
    }
    trial[i] = state[i] + scale * delta;
    if (!std::isfinite(trial[i]) || trial[i] < 0.) return false;
    if (i < npositive && !(trial[i] > floor)) return false;
    if (max_ratio > 0. && i < nlimited && state[i] > 0. &&
        (trial[i] > max_ratio * state[i] || trial[i] * max_ratio < state[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace kintera
