#pragma once

// torch
#include <torch/torch.h>

namespace kintera {

// Forward declaration
struct SpeciesThermo;
struct ThermoOptions;

torch::Tensor eval_cv_R(torch::Tensor temp, torch::Tensor conc,
                        SpeciesThermo const& op);

torch::Tensor eval_cp_R(torch::Tensor temp, torch::Tensor conc,
                        SpeciesThermo const& op);

torch::Tensor eval_czh(torch::Tensor temp, torch::Tensor conc,
                       SpeciesThermo const& op);

torch::Tensor eval_czh_ddC(torch::Tensor temp, torch::Tensor conc,
                           SpeciesThermo const& op);

torch::Tensor eval_intEng_R(torch::Tensor temp, torch::Tensor conc,
                            SpeciesThermo const& op);

torch::Tensor eval_enthalpy_R(torch::Tensor temp, torch::Tensor conc,
                              SpeciesThermo const& op);

torch::Tensor eval_entropy_R(torch::Tensor temp, torch::Tensor pres,
                             torch::Tensor conc, torch::Tensor stoich,
                             SpeciesThermo const& op);

torch::Tensor eval_entropy_R(torch::Tensor temp, torch::Tensor pres,
                             torch::Tensor conc, torch::Tensor stoich,
                             ThermoOptions const& op);

}  // namespace kintera
