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
    torch::optional<torch::Tensor> temp = torch::nullopt,
    torch::optional<torch::Tensor> logrc_ddT = torch::nullopt,
    torch::optional<SpeciesThermo> op = torch::nullopt);

torch::Tensor jacobian_evaporation(torch::Tensor rate, torch::Tensor stoich,
                                   torch::Tensor conc, torch::Tensor temp,
                                   ThermoOptions const& op);

}  // namespace kintera
