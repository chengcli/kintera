#pragma once

// C/C++
#include <cmath>

// base
#include <configure.h>

// kintera
#include <kintera/vapors/vapor_functions.h>

#include <kintera/utils/user_funcs.hpp>

namespace kintera {

//! Number of inline SVP parameters stored per reaction (padded).
//!   ideal:   {T3, P3, beta, gamma, betas, gammas}
//!   antoine: {A, B, C}
constexpr int KSVP_NPARAM = 6;

//! \brief Evaluate log saturation vapor pressure for one reaction.
//!
//! \param kind   0 = named func-table formula (use \p fptr); 1 = inline
//! 'ideal';
//!               2 = inline 'antoine'.
//! \param p      pointer to this reaction's parameter block (KSVP_NPARAM long).
//! \param fptr   func-table pointer for the named case (kind == 0).
//! \param T      temperature [K].
//!
//! The inline forms reuse the exact same helpers as the named func-table
//! functions (e.g. h2o_ideal), so a parametrized curve is bit-for-bit
//! consistent with its hardcoded counterpart.
DISPATCH_MACRO
inline double eval_logsvp(int kind, double const* p, user_func1 fptr,
                          double T) {
  if (kind == 1) {  // ideal, two-branch (liquid above T3, solid below)
    double T3 = p[0], P3 = p[1];
    double beta = (T > T3) ? p[2] : p[4];
    double gamma = (T > T3) ? p[3] : p[5];
    return logsvp_ideal(T / T3, beta, gamma) + log(P3);
  } else if (kind == 2) {  // antoine
    return logsvp_antoine(T, p[0], p[1], p[2]);
  }
  return fptr(T);
}

//! \brief Evaluate d(log svp)/dT for one reaction (see \ref eval_logsvp).
DISPATCH_MACRO
inline double eval_logsvp_ddT(int kind, double const* p, user_func1 fptr_ddT,
                              double T) {
  if (kind == 1) {  // ideal
    double T3 = p[0];
    double beta = (T > T3) ? p[2] : p[4];
    double gamma = (T > T3) ? p[3] : p[5];
    return logsvp_ideal_ddT(T / T3, beta, gamma) / T3;
  } else if (kind == 2) {  // antoine
    return logsvp_antoine_ddT(T, p[1], p[2]);
  }
  return fptr_ddT(T);
}

}  // namespace kintera
