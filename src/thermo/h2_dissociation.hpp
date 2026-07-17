#pragma once
// H2 <-> 2H equilibrium folded into the thermodynamics of ONE lumped H/He
// species.
//
// WHY: snapy's `moist-mixture` EOS delegates every thermodynamic question to
// kintera
// ("VT->P", "VU->T", "VT->cv", eval_czh, ...). So a dissociating gas can be
// represented WITHOUT advecting H: kintera resolves the equilibrium internally
// at (T, c) and reports the resulting particle count, internal energy and heat
// capacities. No extra tracers, no chemistry operator, no stiff solve -- and
// snapy needs NO change at all.
//
// The lumped species has composition {H: nH, He: nHe} per mole. At (T, c):
//
//   Kc(T) = Kp(T) * P0/(R T)        Kp from NASA-9 Gibbs:  ln Kp = -(2 g_H -
//   g_H2) [H]   = (-Kc + sqrt(Kc^2 + 8 Kc nH c)) / 4        <- closed form, ONE
//   quadratic (~10 flops) [H2]  = (nH c - [H]) / 2 ,   [He] = nHe c
//
// and then, ALL DERIVED FROM THAT SAME SPECIATION so they are mutually
// consistent:
//
//   cz   = n_tot / c                     <- particles per mole of species (>1
//   once dissociated) e/R  = sum_i n_i (h_i/R - T) / c     <- NASA-9 h is
//   ABSOLUTE, so the 436 kJ/mol lands here cv/R = [sum_i n_i cv_i/R + sum_i
//   (u_i/R) dn_i/dT] / c   <- the 2nd term IS the latent heat cp/R = cv/R + (cz
//   + T dcz/dT)^2 / (cz + c dcz/dc)       <- exact reacting-gas Mayer relation
//
// Consistency is the whole point: grad_ad is EMERGENT from (P, u, cv), not
// imposed. Correcting cp alone is NOT enough -- the particle-number term (delta
// = 1.13 in the BD deep) carries ~12% of grad_ad, so a cp-only fix lands on
// 0.162 instead of the true 0.197 (fixed 2/7 gives 0.286).

#include <kintera/constants.h>
#include <torch/torch.h>

#include "../species.hpp"

namespace kintera {
namespace h2diss {

constexpr double kP0 = 1.0e5;  // NASA-9 standard state [Pa]
constexpr double kTref =
    300.0;  // MUST match kNasa9Tref in eval_uhs.cpp (energy continuity)

inline torch::Tensor cp_R_of(torch::Tensor const& a, torch::Tensor const& T) {
  return a.select(-1, 0) * T.pow(-2) + a.select(-1, 1) / T + a.select(-1, 2) +
         a.select(-1, 3) * T + a.select(-1, 4) * T.pow(2) +
         a.select(-1, 5) * T.pow(3) + a.select(-1, 6) * T.pow(4);
}
//! h/R [K] -- ABSOLUTE (a7 carries the formation enthalpy), so 2 h_H - h_H2 =
//! D(T).
inline torch::Tensor h_R_of(torch::Tensor const& a, torch::Tensor const& T) {
  return -a.select(-1, 0) / T + a.select(-1, 1) * T.log() +
         a.select(-1, 2) * T + a.select(-1, 3) * T.pow(2) / 2 +
         a.select(-1, 4) * T.pow(3) / 3 + a.select(-1, 5) * T.pow(4) / 4 +
         a.select(-1, 6) * T.pow(5) / 5 + a.select(-1, 7);
}
inline torch::Tensor s_R_of(torch::Tensor const& a, torch::Tensor const& T) {
  return -a.select(-1, 0) / (2 * T.pow(2)) - a.select(-1, 1) / T +
         a.select(-1, 2) * T.log() + a.select(-1, 3) * T +
         a.select(-1, 4) * T.pow(2) / 2 + a.select(-1, 5) * T.pow(3) / 3 +
         a.select(-1, 6) * T.pow(4) / 4 + a.select(-1, 8);
}

//! Everything the model needs at one (T, c). `ab` is (2,3,9) =
//! [low|high][H2,H,He][coeff].
struct State {
  torch::Tensor H, H2, He, ntot, Kc;
  torch::Tensor hH, hH2, hHe, cpH, cpH2, cpHe;
  torch::Tensor U;  // internal energy /R per mole of the lumped species [K]
};

inline State speciate(torch::Tensor const& temp, torch::Tensor const& cc,
                      double nH, double nHe, torch::Tensor const& ab) {
  auto low = ab.select(0, 0), high = ab.select(0, 1);  // (3,9)
  auto hot = (temp >= 1000.).unsqueeze(-1);            // (...,1)
  auto A = [&](int sp) {
    return torch::where(hot, high.select(0, sp), low.select(0, sp));  // (...,9)
  };
  auto Tb = temp;  // per-cell (spatial...): h_R_of/cp_R_of/s_R_of consume the
                   // coeff axis via a.select(-1,k) -> (spatial...), so T must
                   // NOT carry a trailing singleton (else it OUTER-PRODUCTS).
                   // n=1 masked this for all prior validation.
  auto aH2 = A(0), aH = A(1), aHe = A(2);

  State s;
  s.hH2 = h_R_of(aH2, Tb);
  s.hH = h_R_of(aH, Tb);
  s.hHe = h_R_of(aHe, Tb);
  s.cpH2 = cp_R_of(aH2, Tb);
  s.cpH = cp_R_of(aH, Tb);
  s.cpHe = cp_R_of(aHe, Tb);
  auto sH2 = s_R_of(aH2, Tb);
  auto sH = s_R_of(aH, Tb);

  auto gH = s.hH / temp - sH;  // G/(RT)
  auto gH2 = s.hH2 / temp - sH2;
  auto lnKp = -(2.0 * gH - gH2);
  auto Kp = torch::exp(lnKp.clamp(-700., 700.));
  s.Kc = Kp * kP0 / (constants::Rgas * temp);  // mol/m^3

  auto nHc = nH * cc;
  // root of 2[H]^2 + Kc[H] - Kc*nHc = 0 in the cancellation-free form
  // H = 2*Kc*nHc / (Kc + sqrt(Kc^2 + 8*Kc*nHc)): exact -> nHc as Kc -> inf
  // (full dissociation), whereas (-Kc + sqrt(...))/4 loses all precision there
  // (large-Kc cancellation).
  auto disc = (s.Kc * s.Kc + 8.0 * s.Kc * nHc).clamp_min(0.);
  s.H = (2.0 * s.Kc * nHc / (s.Kc + disc.sqrt()).clamp_min(1e-300))
            .clamp_min(0.)
            .minimum(nHc);
  s.H2 = (nHc - s.H) / 2.0;
  s.He = nHe * cc;
  s.ntot = s.H + s.H2 + s.He;

  // u_i/R = h_i/R - T (ideal gas)
  s.U = (s.H * (s.hH - temp) + s.H2 * (s.hH2 - temp) + s.He * (s.hHe - temp)) /
        cc;
  return s;
}

struct Result {
  torch::Tensor cz;      // particles per mole of the lumped species
  torch::Tensor cz_ddC;  // d cz / d c
  torch::Tensor cp_R;
  torch::Tensor cv_R;
  torch::Tensor e_R;  // internal energy /R, referenced to kTref (== 0 there)
};

//! \param temp (...) [K]   \param c (...) molar conc. of the lumped species
//! [mol/m^3]
inline Result eval(torch::Tensor const& temp, torch::Tensor const& c, double nH,
                   double nHe, torch::Tensor const& ab) {
  auto cc = c.clamp_min(1e-30);
  auto s = speciate(temp, cc, nH, nHe, ab);

  auto denom = (4.0 * s.H + s.Kc).clamp_min(1e-300);
  auto dH_R = 2.0 * s.hH - s.hH2;  // (2h_H - h_H2)/R  [K]  == D(T)/R
  auto dKc_dT = s.Kc * (dH_R / temp.pow(2) - 1.0 / temp);  // van 't Hoff
  auto dH_dT = dKc_dT * (nH * cc - s.H) / denom;
  auto dH_dc = s.Kc * nH / denom;

  auto cz = s.ntot / cc;
  auto dcz_dT = dH_dT / (2.0 * cc);
  auto dcz_dc = dH_dc / (2.0 * cc) - s.H / (2.0 * cc * cc);

  auto uH = s.hH - temp, uH2 = s.hH2 - temp;
  auto cv_R = (s.H * (s.cpH - 1.0) + s.H2 * (s.cpH2 - 1.0) +
               s.He * (s.cpHe - 1.0) + uH * dH_dT + uH2 * (-dH_dT / 2.0)) /
              cc;

  auto num = (cz + temp * dcz_dT).pow(2);
  auto den = (cz + cc * dcz_dc).clamp_min(1e-8);
  auto cp_R = cv_R + num / den;

  // reference the internal energy to kTref at the SAME c (eval_nasa9's
  // convention -> continuity)
  auto T0 = torch::full_like(temp, kTref);
  auto s0 = speciate(T0, cc, nH, nHe, ab);
  auto e_R = s.U - s0.U;

  return {cz, dcz_dc, cp_R, cv_R, e_R};
}

}  // namespace h2diss
}  // namespace kintera
