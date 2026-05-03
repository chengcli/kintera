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

//! Options for constructing actinic flux on a target wavelength grid.
struct ActinicFluxOptionsImpl {
  static std::shared_ptr<ActinicFluxOptionsImpl> create() {
    return std::make_shared<ActinicFluxOptionsImpl>();
  }

  //! Source wavelength grid [nm] for tabulated default flux values.
  ADD_ARG(std::vector<double>, wavelength) = {};

  //! Default flux values [photons cm^-2 s^-1 nm^-1] on `wavelength()`.
  ADD_ARG(std::vector<double>, default_flux) = {};

  //! Minimum wavelength for convenience-generated grids [nm]
  ADD_ARG(double, wave_min) = 0.0;

  //! Maximum wavelength for convenience-generated grids [nm]
  ADD_ARG(double, wave_max) = 1000.0;
};
using ActinicFluxOptions = std::shared_ptr<ActinicFluxOptionsImpl>;

//! Interpolate a tabulated actinic flux field to a new wavelength grid.
inline torch::Tensor interpolate_actinic_flux(torch::Tensor wavelength,
                                              torch::Tensor flux,
                                              torch::Tensor new_wavelength) {
  TORCH_CHECK(wavelength.dim() == 1, "Wavelength must be 1D tensor");
  TORCH_CHECK(flux.size(0) == wavelength.size(0),
              "Flux first dimension must match wavelength size");
  return interpn({new_wavelength}, {wavelength}, flux);
}

//! Create actinic flux on a target wavelength grid from options.
inline torch::Tensor create_actinic_flux(ActinicFluxOptions const& opts,
                                         torch::Tensor wavelength) {
  if (!wavelength.defined() || wavelength.numel() == 0) {
    return torch::Tensor();
  }

  if (opts->default_flux().empty()) {
    return torch::ones_like(wavelength);
  }

  auto flux = torch::tensor(opts->default_flux(), wavelength.options());

  if (opts->wavelength().empty()) {
    TORCH_CHECK(
        flux.numel() == wavelength.numel(),
        "ActinicFluxOptions default_flux must match target wavelength size "
        "when no source wavelength grid is provided");
    return flux;
  }

  auto src_wavelength = torch::tensor(opts->wavelength(), wavelength.options());
  return interpolate_actinic_flux(src_wavelength, flux, wavelength);
}

//! Create simple uniform actinic flux on a target wavelength grid.
inline torch::Tensor create_uniform_flux(torch::Tensor wavelength,
                                         double flux_value) {
  return flux_value * torch::ones_like(wavelength);
}

//! Create solar-like actinic flux on a target wavelength grid.
inline torch::Tensor create_solar_flux(torch::Tensor wavelength,
                                       double peak_flux = 1.e14) {
  auto peak_wave = 500.0;
  auto width = 200.0;
  return peak_flux *
         torch::exp(-torch::pow((wavelength - peak_wave) / width, 2));
}

}  // namespace kintera

#undef ADD_ARG
