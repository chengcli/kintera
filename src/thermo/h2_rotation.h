#pragma once

#include <configure.h>

#include <cmath>

namespace kintera {

constexpr double kNasa9Tref = 300.0;  // matches ThermoOptions default Tref
constexpr double kThetaRot = 87.55;   // K, H2 rotational temperature
constexpr int kJmax = 40;

// Rigid-rotor H2 (theta_rot, J = 0..kJmax) for use-h2-cp, per cell on CPU/CUDA:
// <m>, cv_rot/R and s_rot/R = ln Z + theta <m>/T; eq 0 normal (1:3), 1 equil.
DISPATCH_MACRO inline void h2_rot(double T, int eq, double* m1, double* cv,
                                  double* s) {
  double z[2] = {}, s1[2] = {}, s2[2] = {};
  for (int J = 0; J <= kJmax; ++J) {
    int k = eq ? 0 : J % 2;  // normal: para (even J) and ortho apart
    double m = J * (J + 1.);
    double w =
        (2. * J + 1.) * (eq && J % 2 ? 3. : 1.) * exp(-kThetaRot * m / T);
    z[k] += w;
    s1[k] += w * m;
    s2[k] += w * m * m;
  }
  *m1 = *cv = *s = 0.;
  for (int k = 0; k < 2 - eq; ++k) {
    double f = eq ? 1. : (k ? 0.75 : 0.25), a = s1[k] / z[k];
    *m1 += f * a;
    *cv += f * ((kThetaRot / T) * (kThetaRot / T) * (s2[k] / z[k] - a * a));
    *s += f * (log(z[k]) + kThetaRot * a / T);
  }
}

// 0 e/R, 1 cv/R, 2 s/R of H2; e and s are relative to kNasa9Tref
DISPATCH_MACRO inline double h2_rot_eval(double T, int eq, int what) {
  double m, cv, s, m0, cv0, s0;
  h2_rot(T, eq, &m, &cv, &s);
  h2_rot(kNasa9Tref, eq, &m0, &cv0, &s0);
  if (what == 1) return 1.5 + cv;
  if (what == 2) return 2.5 * log(T / kNasa9Tref) + (s - s0);
  return 1.5 * (T - kNasa9Tref) + kThetaRot * (m - m0);
}
DISPATCH_MACRO inline double h2_rot_normal(double T, double) {
  return h2_rot_eval(T, 0, 0);
}
DISPATCH_MACRO inline double h2_rot_normal_ddT(double T, double) {
  return h2_rot_eval(T, 0, 1);
}
DISPATCH_MACRO inline double h2_rot_normal_s(double T, double) {
  return h2_rot_eval(T, 0, 2);
}
DISPATCH_MACRO inline double h2_rot_equilibrium(double T, double) {
  return h2_rot_eval(T, 1, 0);
}
DISPATCH_MACRO inline double h2_rot_equilibrium_ddT(double T, double) {
  return h2_rot_eval(T, 1, 1);
}
DISPATCH_MACRO inline double h2_rot_equilibrium_s(double T, double) {
  return h2_rot_eval(T, 1, 2);
}

}  // namespace kintera
