#pragma once

// C/C++
#include <optional>

// torch
#include <torch/torch.h>

// kintera
#include <kintera/thermo/thermo.hpp>

namespace kintera {

torch::Tensor jacobian_mass_action(
    torch::Tensor rate, torch::Tensor stoich, torch::Tensor conc,
    torch::optional<torch::Tensor> rc_ddT = torch::nullopt,
    torch::optional<ThermoOptions> op = torch::nullopt);

torch::Tensor jacobian_evaporation(torch::Tensor rate, torch::Tensor stoich,
                                   torch::Tensor conc, ThermoOptions op,
                                   double ftol = 1.e-6);

}  // namespace kintera
