#pragma once

// C/C++
#include <cmath>

// kintera
#include <kintera/utils/func1.hpp>

#define VAPOR_FUNCTION(name, var)                   \
  double name(double);                              \
  static Func1Registrar logsvp_##name(#name, name); \
  double name(double var)

inline double logsvp_ideal(double t, double beta, double gamma) {
  return (1. - 1. / t) * beta - gamma * log(t);
}

inline double logsvp_ideal_ddT(double t, double beta, double gamma) {
  return beta / (t * t) - gamma / t;
}
