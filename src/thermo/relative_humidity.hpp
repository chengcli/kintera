#pragma once

// kintera
#include "thermo.hpp"

namespace kintera {

torch::Tensor relative_humidity(torch::Tensor temp, torch::Tensor pres,
                                torch::Tensor conc, torch::Tensor stoich,
                                ThermoOptions op);

}  // namespace kintera
