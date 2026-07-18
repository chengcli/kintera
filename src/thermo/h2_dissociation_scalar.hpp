#pragma once
// SCALAR (per-cell, plain-double) transcription of h2_dissociation.hpp.
//
// WHY: the torch h2diss::eval expresses the H2<->2H equilibrium thermo as ~1000
// tensor ops; on the BD field (512 cells/rank) that is ~1.12 s/cyc of pure ATen
// dispatch overhead (E0: 93% field-size-independent). The fused kernel (Design C
// / ISSUES P1(3)) turns the loop inside out -- one launch per solve, per-cell
// Newton -- and needs the physics as ONE scalar function body. This header is
// that body: cp_R_of/h_R_of/s_R_of/speciate/eval, line-for-line the same math as
// h2_dissociation.hpp, in double. NO torch, NO printf, header-only, host-callable
// now (GPU DISPATCH_MACRO templating is deferred to S5).
//
// SINGLE SOURCE OF TRUTH for the NASA-9 coefficients: the caller passes `ab`, the
// SAME (2,3,9) = [low|high][H2,H,He][coeff] block that nasa9_coeffs_by_name(
// {"H2","H","He"}) builds for the torch path. We never hardcode the 54 numbers.
// Layout (row-major, contiguous): ab[range*27 + species*9 + k], range 0=low
// 1=high, species 0=H2 1=H 2=He, k in 0..8 with k=7 the h integration constant
// (a8) and k=8 the s integration constant (a9). Matches a.select(-1,k) in the
// torch helpers.
//
// GATE: tests/test_h2diss_scalar.cpp compares this against torch h2diss::eval on
// a (T,c) grid to ~1e-14 rel. Do NOT change the semantics here without re-running
// it -- it is the transcription guard.

#include <algorithm>  // std::min, std::max
#include <cmath>      // std::log, std::sqrt, std::exp, std::fmax, std::fmin

#include <kintera/constants.h>  // constants::Rgas (torch-free)

namespace kintera {
namespace h2diss_scalar {

constexpr double kP0 = 1.0e5;   // NASA-9 standard state [Pa]  (== h2diss::kP0)
constexpr double kTref = 300.;  // energy reference [K]        (== h2diss::kTref)

//! One species' NASA-9 coefficient row (9 doubles). The lnT overloads take a
//! precomputed std::log(T) -- speciate() would otherwise evaluate the SAME
//! log(T) five times per call (profile: log/exp are ~40% of the whole fused
//! run). Bit-identical: same std::log value, just computed once.
inline double cp_R_of(double const* a, double T) {
  return a[0] / (T * T) + a[1] / T + a[2] + a[3] * T + a[4] * (T * T) +
         a[5] * (T * T * T) + a[6] * (T * T * T * T);
}
//! h/R [K] -- ABSOLUTE (a[7] carries the formation enthalpy), so 2 h_H - h_H2 =
//! D(T).
inline double h_R_of(double const* a, double T, double lnT) {
  double T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;
  return -a[0] / T + a[1] * lnT + a[2] * T + a[3] * T2 / 2 + a[4] * T3 / 3 +
         a[5] * T4 / 4 + a[6] * T5 / 5 + a[7];
}
inline double h_R_of(double const* a, double T) {
  return h_R_of(a, T, std::log(T));
}
inline double s_R_of(double const* a, double T, double lnT) {
  double T2 = T * T, T3 = T2 * T, T4 = T3 * T;
  return -a[0] / (2 * T2) - a[1] / T + a[2] * lnT + a[3] * T + a[4] * T2 / 2 +
         a[5] * T3 / 3 + a[6] * T4 / 4 + a[8];
}
inline double s_R_of(double const* a, double T) {
  return s_R_of(a, T, std::log(T));
}

//! Everything the model needs at one (T, c). Mirrors h2diss::State.
struct State {
  double H, H2, He, ntot, Kc;
  double hH, hH2, hHe, cpH, cpH2, cpHe;
  double U;  // internal energy /R per mole of the lumped species [K]
};

//! Faithful scalar copy of h2diss::speciate. `ab` is the (2,3,9) block; `cc` is
//! the (already-clamped, in eval) molar conc of the lumped species [mol/m^3].
inline State speciate(double temp, double cc, double nH, double nHe,
                      double const* ab) {
  int r = (temp >= 1000.) ? 1 : 0;  // torch: hot = (temp >= 1000.)
  double const* aH2 = ab + r * 27 + 0 * 9;
  double const* aH = ab + r * 27 + 1 * 9;
  double const* aHe = ab + r * 27 + 2 * 9;
  const double lnT = std::log(temp);  // shared by the 5 log-bearing polys

  State s;
  s.hH2 = h_R_of(aH2, temp, lnT);
  s.hH = h_R_of(aH, temp, lnT);
  s.hHe = h_R_of(aHe, temp, lnT);
  s.cpH2 = cp_R_of(aH2, temp);
  s.cpH = cp_R_of(aH, temp);
  s.cpHe = cp_R_of(aHe, temp);
  double sH2 = s_R_of(aH2, temp, lnT);
  double sH = s_R_of(aH, temp, lnT);

  double gH = s.hH / temp - sH;  // G/(RT)
  double gH2 = s.hH2 / temp - sH2;
  double lnKp = -(2.0 * gH - gH2);
  double Kp = std::exp(std::max(-700., std::min(700., lnKp)));
  s.Kc = Kp * kP0 / (constants::Rgas * temp);  // mol/m^3

  double nHc = nH * cc;
  // cancellation-free root of 2[H]^2 + Kc[H] - Kc*nHc = 0:
  //   H = 2*Kc*nHc / (Kc + sqrt(Kc^2 + 8*Kc*nHc))
  double disc = std::max(0., s.Kc * s.Kc + 8.0 * s.Kc * nHc);
  double H = 2.0 * s.Kc * nHc / std::max(1e-300, s.Kc + std::sqrt(disc));
  s.H = std::min(nHc, std::max(0., H));
  s.H2 = (nHc - s.H) / 2.0;
  s.He = nHe * cc;
  s.ntot = s.H + s.H2 + s.He;

  // u_i/R = h_i/R - T (ideal gas)
  s.U = (s.H * (s.hH - temp) + s.H2 * (s.hH2 - temp) + s.He * (s.hHe - temp)) /
        cc;
  return s;
}

//! Mirrors h2diss::Result.
struct Result {
  double cz;      // particles per mole of the lumped species
  double cz_ddC;  // d cz / d c
  double cp_R;
  double cv_R;
  double e_R;  // internal energy /R, referenced to kTref
};

//! The T0=300 K internal-energy reference, s0.U. It is a MODEL CONSTANT: at 300 K
//! dissociation is ~e^-70, so [H]~0 and s0.U depends only on (nH,nHe,ab), NOT c
//! (the c cancels: n_i proportional to cc, U divides by cc). eval() below still
//! computes it faithfully via speciate(kTref,cc) to match torch to ~1e-30; the
//! fused Newton loop (S2/S3) hoists THIS constant out of the iteration instead.
inline double e0_ref(double nH, double nHe, double const* ab) {
  return speciate(kTref, 1.0, nH, nHe, ab).U;  // cc=1: c-independent
}

//! Scalar h2diss::eval with a PRECOMPUTED T0 reference (`e0` = e0_ref(...)):
//! skips the second speciate() the faithful eval performs at kTref every call
//! -- at 300 K dissociation is ~e^-70, so s0.U/cc is c-independent to ~1e-13
//! rel (gated by tests/test_h2diss_scalar.cpp T0ReferenceIsCIndependent),
//! far below ftol. Halves the per-iteration cost of the fused Newton kernels.
inline Result eval(double temp, double c, double nH, double nHe,
                   double const* ab, double e0) {
  double cc = std::max(c, 1e-30);
  State s = speciate(temp, cc, nH, nHe, ab);

  double denom = std::max(1e-300, 4.0 * s.H + s.Kc);
  double dH_R = 2.0 * s.hH - s.hH2;  // (2h_H - h_H2)/R [K] == D(T)/R
  double dKc_dT = s.Kc * (dH_R / (temp * temp) - 1.0 / temp);  // van 't Hoff
  double dH_dT = dKc_dT * (nH * cc - s.H) / denom;
  double dH_dc = s.Kc * nH / denom;

  double cz = s.ntot / cc;
  double dcz_dT = dH_dT / (2.0 * cc);
  double dcz_dc = dH_dc / (2.0 * cc) - s.H / (2.0 * cc * cc);

  double uH = s.hH - temp, uH2 = s.hH2 - temp;
  double cv_R = (s.H * (s.cpH - 1.0) + s.H2 * (s.cpH2 - 1.0) +
                 s.He * (s.cpHe - 1.0) + uH * dH_dT + uH2 * (-dH_dT / 2.0)) /
                cc;

  double num = (cz + temp * dcz_dT) * (cz + temp * dcz_dT);
  double den = std::max(1e-8, cz + cc * dcz_dc);
  double cp_R = cv_R + num / den;

  Result out;
  out.cz = cz;
  out.cz_ddC = dcz_dc;
  out.cp_R = cp_R;
  out.cv_R = cv_R;
  out.e_R = s.U - e0;
  return out;
}

//! Faithful scalar copy of h2diss::eval (recomputes the T0 reference at the
//! SAME cc, matching torch bit-for-bit; the transcription guard compares THIS).
inline Result eval(double temp, double c, double nH, double nHe,
                   double const* ab) {
  double cc = std::max(c, 1e-30);
  // reference the internal energy to kTref at the SAME c (continuity)
  State s0 = speciate(kTref, cc, nH, nHe, ab);
  return eval(temp, c, nH, nHe, ab, s0.U);
}

}  // namespace h2diss_scalar
}  // namespace kintera
