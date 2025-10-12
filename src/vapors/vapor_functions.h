#pragma once

// C/C++
#include <cmath>

// base
#include <configure.h>

namespace kintera {

DISPATCH_MACRO
inline double logsvp_ideal(double t, double beta, double gamma) {
  return (1. - 1. / t) * beta - gamma * log(t);
}

DISPATCH_MACRO
inline double logsvp_ideal_ddT(double t, double beta, double gamma) {
  return beta / (t * t) - gamma / t;
}

DISPATCH_MACRO
inline double logsvp_antoine(double T, double A, double B, double C) {
  return log(1.E5) + (A - B / (T + C)) * log(10.);
}

DISPATCH_MACRO
inline double logsvp_antoine_ddT(double T, double B, double C) {
  return B * log(10.) / ((T + C) * (T + C));
}

DISPATCH_MACRO
inline double h2o_ideal(double T) {
  double betal = 24.845, gammal = 4.986009, betas = 22.98, gammas = 0.52,
         tr = 273.16, pr = 611.7;
  return (T > tr ? logsvp_ideal(T / tr, betal, gammal)
                 : logsvp_ideal(T / tr, betas, gammas)) +
         log(pr);
}

DISPATCH_MACRO
inline double h2o_ideal_ddT(double T) {
  double betal = 24.845, gammal = 4.986009, betas = 22.98, gammas = 0.52,
         tr = 273.16;
  return (T > tr ? logsvp_ideal_ddT(T / tr, betal, gammal)
                 : logsvp_ideal_ddT(T / tr, betas, gammas)) /
         tr;
}

DISPATCH_MACRO
inline double nh3_ideal(double T) {
  double betal = 20.08, gammal = 5.62, betas = 20.64, gammas = 1.43, tr = 195.4,
         pr = 6060.;

  return (T > tr ? logsvp_ideal(T / tr, betal, gammal)
                 : logsvp_ideal(T / tr, betas, gammas)) +
         log(pr);
}

DISPATCH_MACRO
inline double nh3_ideal_ddT(double T) {
  double betal = 20.08, gammal = 5.62, betas = 20.64, gammas = 1.43, tr = 195.4;

  return (T > tr ? logsvp_ideal_ddT(T / tr, betal, gammal)
                 : logsvp_ideal_ddT(T / tr, betas, gammas)) /
         tr;
}

DISPATCH_MACRO
inline double nh3_h2s_lewis(double T) {
  return (14.82 - 4705. / T) * log(10.) + 2. * log(101325.);
}

DISPATCH_MACRO
inline double nh3_h2s_lewis_ddT(double T) { return 4705. * log(10.) / (T * T); }

// H2S vapor function
// T3: 187.63, P3: 23300., beta: 11.89, delta: 5.04, minT: 100.
// double check for solid phase later
DISPATCH_MACRO
inline double h2s_ideal(double T) {
  double betal = 11.89, gammal = 5.04, betas = 11.89, gammas = 5.04,
         tr = 187.63, pr = 23300.0;
  return (T > tr ? logsvp_ideal(T / tr, betal, gammal)
                 : logsvp_ideal(T / tr, betas, gammas)) +
         log(pr);
}

DISPATCH_MACRO
inline double h2s_ideal_ddT(double T) {
  double betal = 11.89, gammal = 5.04, betas = 11.89, gammas = 5.04,
         tr = 187.63;
  return (T > tr ? logsvp_ideal_ddT(T / tr, betal, gammal)
                 : logsvp_ideal_ddT(T / tr, betas, gammas)) /
         tr;
}

DISPATCH_MACRO
inline double h2s_antoine(double T) {
  if (T < 212.8) {
    return logsvp_antoine(T, 4.43681, 829.439, 25.412);
  } else {
    return logsvp_antoine(T, 4.52887, 958.587, 0.539);
  }
}

DISPATCH_MACRO
inline double h2s_antoine_ddT(double T) {
  if (T < 212.8) {
    return logsvp_antoine_ddT(T, 829.439, 25.412);
  } else {
    return logsvp_antoine_ddT(T, 958.587, 0.539);
  }
}

DISPATCH_MACRO
inline double ch4_ideal(double T) {
  double betal = 10.15, gammal = 2.1, betas = 10.41, gammas = 0.9, tr = 90.67,
         pr = 11690.;

  return (T > tr ? logsvp_ideal(T / tr, betal, gammal)
                 : logsvp_ideal(T / tr, betas, gammas)) +
         log(pr);
}

DISPATCH_MACRO
inline double ch4_ideal_ddT(double T) {
  double betal = 10.15, gammal = 2.1, betas = 10.41, gammas = 0.9, tr = 90.67;

  return (T > tr ? logsvp_ideal_ddT(T / tr, betal, gammal)
                 : logsvp_ideal_ddT(T / tr, betas, gammas)) /
         tr;
}

DISPATCH_MACRO
inline double so2_antoine(double T) {
  double A = 3.48586;
  double B = 668.225;
  double C = -72.252;
  return logsvp_antoine(T, A, B, C);
}

DISPATCH_MACRO
inline double so2_antoine_ddT(double T) {
  double B = 668.225;
  double C = -72.252;
  return logsvp_antoine_ddT(T, B, C);
}

DISPATCH_MACRO
inline double co2_antoine(double T) {
  double A = 6.81228;
  double B = 1301.679;
  double C = -34.94;
  return logsvp_antoine(T, A, B, C);
}

DISPATCH_MACRO
inline double co2_antoine_ddT(double T) {
  double B = 1301.679;
  double C = -34.94;
  return logsvp_antoine_ddT(T, B, C);
}

}  // namespace kintera
