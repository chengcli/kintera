#pragma once

// kintera
#include "nucleation.hpp"

namespace kintera {

torch::Tensor relative_humidity(torch::Tensor temp, torch::Tensor conc,
                                torch::Tensor stoich,
                                NucleationOptions const& op);

}  // namespace kintera
