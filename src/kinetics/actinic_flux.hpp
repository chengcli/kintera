#pragma once

// C/C++
#include <memory>
#include <vector>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/add_arg.h>
#include <kintera/math/interpolation.hpp>

namespace kintera {

//! Data structure for storing and interpolating actinic flux
/*!
 * Actinic flux F(λ) represents the rate at which photons of wavelength λ
 * are available to drive photochemical reactions. Units are typically
 * photons cm^-2 s^-1 nm^-1.
 *
 * This structure supports:
 * - Single-point (0D) flux data
 * - Column/layer-varying (1D/2D) flux data
 * - Time-varying flux (optional)
 */
struct ActinicFluxData {
  //! Create an ActinicFluxData with given wavelength grid and flux
  /*!
   * \param wavelength Wavelength grid [nm], shape (nwave,)
   * \param flux Actinic flux [photons cm^-2 s^-1 nm^-1], shape (nwave, ...)
   */
  ActinicFluxData(torch::Tensor wavelength_, torch::Tensor flux_)
      : wavelength(wavelength_), flux(flux_) {
    TORCH_CHECK(wavelength.dim() == 1, "Wavelength must be 1D tensor");
    TORCH_CHECK(flux.size(0) == wavelength.size(0),
                "Flux first dimension must match wavelength size");
  }

  //! Default constructor for empty flux
  ActinicFluxData() = default;

  //! Wavelength grid [nm], shape (nwave,)
  torch::Tensor wavelength;

  //! Actinic flux [photons cm^-2 s^-1 nm^-1], shape (nwave, ...)
  torch::Tensor flux;

  //! Check if flux data is valid
  bool is_valid() const {
    return wavelength.defined() && flux.defined() && wavelength.numel() > 0;
  }

  //! Get number of wavelength points
  int64_t nwave() const {
    return wavelength.defined() ? wavelength.size(0) : 0;
  }

  //! Interpolate flux to new wavelength grid
  /*!
   * \param new_wavelength Target wavelength grid [nm]
   * \return Interpolated flux at new wavelengths
   */
  torch::Tensor interpolate_to(torch::Tensor new_wavelength) const {
    if (!is_valid()) {
      return torch::zeros_like(new_wavelength);
    }
    return interpn({new_wavelength}, {wavelength}, flux);
  }

  //! Get flux as a map for passing to forward()
  std::map<std::string, torch::Tensor> to_map() const {
    return {{"wavelength", wavelength}, {"actinic_flux", flux}};
  }
};

//! Options for actinic flux configuration
struct ActinicFluxOptionsImpl {
  static std::shared_ptr<ActinicFluxOptionsImpl> create() {
    return std::make_shared<ActinicFluxOptionsImpl>();
  }

  //! Wavelength grid [nm]
  ADD_ARG(std::vector<double>, wavelength) = {};

  //! Default flux values [photons cm^-2 s^-1 nm^-1]
  ADD_ARG(std::vector<double>, default_flux) = {};

  //! Minimum wavelength for integration [nm]
  ADD_ARG(double, wave_min) = 0.0;

  //! Maximum wavelength for integration [nm]
  ADD_ARG(double, wave_max) = 1000.0;
};
using ActinicFluxOptions = std::shared_ptr<ActinicFluxOptionsImpl>;

//! Create ActinicFluxData from options
inline ActinicFluxData create_actinic_flux(ActinicFluxOptions const& opts,
                                           torch::Device device = torch::kCPU,
                                           torch::Dtype dtype = torch::kFloat64) {
  if (opts->wavelength().empty()) {
    return ActinicFluxData();
  }

  auto wavelength = torch::tensor(opts->wavelength(),
                                  torch::device(device).dtype(dtype));

  torch::Tensor flux;
  if (!opts->default_flux().empty()) {
    flux = torch::tensor(opts->default_flux(),
                         torch::device(device).dtype(dtype));
  } else {
    // Default to unit flux
    flux = torch::ones_like(wavelength);
  }

  return ActinicFluxData(wavelength, flux);
}

//! Create simple uniform actinic flux for testing
inline ActinicFluxData create_uniform_flux(double wave_min, double wave_max,
                                           int nwave, double flux_value,
                                           torch::Device device = torch::kCPU,
                                           torch::Dtype dtype = torch::kFloat64) {
  auto wavelength = torch::linspace(wave_min, wave_max, nwave,
                                    torch::device(device).dtype(dtype));
  auto flux = flux_value * torch::ones_like(wavelength);
  return ActinicFluxData(wavelength, flux);
}

//! Create solar-like actinic flux (simplified model)
/*!
 * Creates a simplified solar actinic flux profile that peaks in the
 * visible range and decreases towards UV.
 *
 * \param wave_min Minimum wavelength [nm]
 * \param wave_max Maximum wavelength [nm]
 * \param nwave Number of wavelength points
 * \param peak_flux Peak flux value [photons cm^-2 s^-1 nm^-1]
 */
inline ActinicFluxData create_solar_flux(double wave_min, double wave_max,
                                         int nwave, double peak_flux = 1.e14,
                                         torch::Device device = torch::kCPU,
                                         torch::Dtype dtype = torch::kFloat64) {
  auto wavelength = torch::linspace(wave_min, wave_max, nwave,
                                    torch::device(device).dtype(dtype));

  // Simple Planck-like profile (not physically accurate, for testing)
  // Peak around 500 nm for solar-like spectrum
  auto peak_wave = 500.0;
  auto width = 200.0;
  auto flux = peak_flux *
              torch::exp(-torch::pow((wavelength - peak_wave) / width, 2));

  return ActinicFluxData(wavelength, flux);
}

}  // namespace kintera

#undef ADD_ARG

