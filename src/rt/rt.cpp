//! @file rt.cpp
//! @brief Beer-Lambert radiative transfer for UV photolysis
//!
//! Computes the actinic flux at each atmospheric layer by integrating
//! optical depth from the top of the atmosphere downward:
//!
//!   tau(z) = integral_TOA^z  sum_i [ n_i * sigma_i ] dz'
//!   F(z)   = F_top * exp( -tau(z) / cos(zenith) )
//!
//! This is equivalent to DISORT with pure absorption (no scattering),
//! which is the correct physics for UV photodissociation.

#include "rt.hpp"

namespace kintera {

torch::Tensor compute_optical_depth(
    torch::Tensor y, torch::Tensor cross_sections,
    torch::Tensor dz, std::vector<int> absorber_ids) {
  int nz = y.size(0);
  int nwave = cross_sections.size(1);
  int n_abs = absorber_ids.size();
  auto opts = y.options();

  TORCH_CHECK(cross_sections.size(0) == n_abs,
              "cross_sections first dim must match absorber_ids length");
  TORCH_CHECK(dz.size(0) == nz, "dz must have length nz");

  // Compute total absorption coefficient at each layer:
  // alpha(z, lambda) = sum_i [ n_i(z) * sigma_i(lambda) ]
  // alpha shape: (nz, nwave)
  auto alpha = torch::zeros({nz, nwave}, opts);
  for (int i = 0; i < n_abs; i++) {
    // n_i: (nz,), sigma_i: (nwave,)
    auto n_i = y.select(1, absorber_ids[i]);  // (nz,)
    auto sigma_i = cross_sections[i];          // (nwave,)
    alpha += n_i.unsqueeze(1) * sigma_i.unsqueeze(0);
  }

  // dtau per layer: dtau = alpha * dz
  auto dtau = alpha * dz.unsqueeze(1);  // (nz, nwave)

  // Cumulative optical depth from TOA (top = index nz-1) downward.
  // Level 0 = TOA (tau=0), level nz = surface.
  // In our convention y[0] = bottom (highest P), y[nz-1] = top (lowest P).
  // So we accumulate from index nz-1 (top) down to index 0 (bottom).
  auto tau = torch::zeros({nz + 1, nwave}, opts);

  // tau[0] = TOA = 0 (at the top of layer nz-1)
  // tau[1] = after passing through layer nz-1
  // ...
  // tau[nz] = surface (after passing through all layers)
  for (int k = 0; k < nz; k++) {
    int layer = nz - 1 - k;  // top-down: layer nz-1, nz-2, ..., 0
    tau[k + 1] = tau[k] + dtau[layer];
  }

  return tau;
}

torch::Tensor compute_actinic_flux(
    torch::Tensor stellar_flux, torch::Tensor tau, double cos_zenith) {
  int nz = tau.size(0) - 1;
  TORCH_CHECK(cos_zenith > 0 && cos_zenith <= 1.0,
              "cos_zenith must be in (0, 1]");

  // Direct beam at each level: F_level = F_top * exp(-tau / mu)
  auto F_levels = stellar_flux.unsqueeze(0) *
                  torch::exp(-tau / cos_zenith);  // (nz+1, nwave)

  // Actinic flux at layer centers = average of top and bottom interfaces
  // Layer nz-1 (top) uses levels 0 and 1
  // Layer 0 (bottom) uses levels nz-1 and nz
  // In level indexing: layer j (from bottom) corresponds to
  //   level_top = nz - j - 1 + 0 = nz - 1 - j
  //   level_bot = nz - j
  // Wait, let me be more careful.
  //
  // Level ordering (top-down):
  //   level 0 = TOA (above layer nz-1)
  //   level 1 = interface between layer nz-1 and nz-2
  //   ...
  //   level k = interface above layer nz-1-k
  //   level nz = surface (below layer 0)
  //
  // So layer j (0-indexed from bottom) has:
  //   top interface = level (nz - 1 - j)
  //   bottom interface = level (nz - j)
  //
  // Actinic flux at layer j = (F[nz-1-j] + F[nz-j]) / 2

  auto aflux = torch::zeros({nz, F_levels.size(1)}, F_levels.options());
  for (int j = 0; j < nz; j++) {
    aflux[j] = (F_levels[nz - 1 - j] + F_levels[nz - j]) / 2.0;
  }

  return aflux;
}

torch::Tensor compute_actinic_flux(
    torch::Tensor y, torch::Tensor cross_sections,
    torch::Tensor stellar_flux, torch::Tensor dz,
    double cos_zenith, std::vector<int> absorber_ids) {
  auto tau = compute_optical_depth(y, cross_sections, dz, absorber_ids);
  return compute_actinic_flux(stellar_flux, tau, cos_zenith);
}

}  // namespace kintera
