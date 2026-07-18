#pragma once

// torch
#include <torch/torch.h>

namespace kintera {

// Forward declaration
struct SpeciesThermoImpl;
struct ThermoOptionsImpl;

using SpeciesThermo = std::shared_ptr<SpeciesThermoImpl>;
using ThermoOptions = std::shared_ptr<ThermoOptionsImpl>;

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

//! Per-device+dtype cached CONTIGUOUS (2,3,9) H2/H/He NASA-9 coefficient block
//! (universal constants). Shared by the h2diss torch path and the fused scalar
//! kernels (thermo_y.cpp), which would otherwise rebuild it per solve.
torch::Tensor h2diss_coeffs_cached(torch::Tensor const& like);

torch::Tensor eval_entropy_R(torch::Tensor temp, torch::Tensor pres,
                             torch::Tensor conc, torch::Tensor stoich,
                             SpeciesThermo const& op);

torch::Tensor eval_entropy_R(torch::Tensor temp, torch::Tensor pres,
                             torch::Tensor conc, torch::Tensor stoich,
                             ThermoOptions const& op);

}  // namespace kintera
