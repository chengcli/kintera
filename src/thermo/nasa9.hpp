#pragma once

// torch
#include <torch/torch.h>

namespace kintera {

//! Compute dimensionless Gibbs free energy g/RT for all species
//! using NASA-9 polynomials with two temperature ranges.
/*!
 * NASA-9 formulas (9 coefficients a0..a6, a8, a9 per range):
 *
 *   h/RT = -a0/T^2 + a1*ln(T)/T + a2 + a3*T/2 + a4*T^2/3
 *          + a5*T^3/4 + a6*T^4/5 + a8/T
 *
 *   s/R  = -a0/(2*T^2) - a1/T + a2*ln(T) + a3*T + a4*T^2/2
 *          + a5*T^3/3 + a6*T^4/4 + a9
 *
 *   g/RT = h/RT - s/R
 *
 * \param temp          temperature [K], shape (...)
 * \param coeffs_low    low-T coefficients, shape (nspecies, 9)
 * \param coeffs_high   high-T coefficients, shape (nspecies, 9)
 * \param Tmid          mid-point temperature, shape (nspecies,)
 * \return              g/RT, shape (..., nspecies)
 */
inline torch::Tensor nasa9_gibbs_RT(torch::Tensor temp,
                                    torch::Tensor coeffs_low,
                                    torch::Tensor coeffs_high,
                                    torch::Tensor Tmid) {
  auto T = temp.unsqueeze(-1);  // (..., 1)
  auto T2 = T * T;
  auto T3 = T2 * T;
  auto T4 = T3 * T;
  auto lnT = T.log();
  auto invT = 1.0 / T;
  auto invT2 = invT * invT;

  auto mask = (T < Tmid);  // (..., nspecies)

  // select coefficients based on temperature range
  // coeffs shape: (nspecies, 9) -> broadcast with (..., nspecies)
  auto a = torch::where(mask.unsqueeze(-1), coeffs_low, coeffs_high);
  // a shape: (..., nspecies, 9)

  auto a0 = a.select(-1, 0);
  auto a1 = a.select(-1, 1);
  auto a2 = a.select(-1, 2);
  auto a3 = a.select(-1, 3);
  auto a4 = a.select(-1, 4);
  auto a5 = a.select(-1, 5);
  auto a6 = a.select(-1, 6);
  auto a8 = a.select(-1, 7);  // stored at index 7 (we skip the unused a7)
  auto a9 = a.select(-1, 8);  // stored at index 8

  auto h_RT = -a0 * invT2 + a1 * lnT * invT + a2 + a3 * T / 2.0 +
              a4 * T2 / 3.0 + a5 * T3 / 4.0 + a6 * T4 / 5.0 + a8 * invT;

  auto s_R = -a0 * invT2 / 2.0 - a1 * invT + a2 * lnT + a3 * T +
             a4 * T2 / 2.0 + a5 * T3 / 3.0 + a6 * T4 / 4.0 + a9;

  return h_RT - s_R;  // (..., nspecies)
}

}  // namespace kintera
