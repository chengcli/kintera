#pragma once

// kintera
#include "thermo.hpp"

namespace kintera {

torch::Tensor relative_humidity(torch::Tensor temp, torch::Tensor conc,
                                torch::Tensor stoich, ThermoOptions const& op);

}  // namespace kintera
