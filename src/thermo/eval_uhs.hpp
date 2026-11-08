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

torch::Tensor eval_entropy_R(torch::Tensor temp, torch::Tensor pres,
                             torch::Tensor conc, torch::Tensor stoich,
                             SpeciesThermo const& op);

torch::Tensor eval_entropy_R(torch::Tensor temp, torch::Tensor pres,
                             torch::Tensor conc, torch::Tensor stoich,
                             ThermoOptions const& op);

//! Pressure change of a constant-cv ideal mixture without subtracting totals.
//! thermal0 excludes species energy offsets; gas0 = sum(rho_i R_i) over
//! gas species, cv0 = sum(rho_i cv_i) over all species. Deltas include changes
//! in composition and thermal energy. Inputs broadcast; dtype/device follow
//! ordinary tensor arithmetic. The caller ensures cv0 and cv0 + dcv > 0.
//! This does not implement temperature-dependent heat capacities or nonideal
//! gas corrections; those require their own thermodynamic inversion.
inline torch::Tensor eval_const_cv_pressure_delta(torch::Tensor const& thermal0,
                                                  torch::Tensor const& gas0,
                                                  torch::Tensor const& cv0,
                                                  torch::Tensor const& dthermal,
                                                  torch::Tensor const& dgas,
                                                  torch::Tensor const& dcv) {
  auto p0 = thermal0 * gas0 / cv0;
  return (dthermal * (gas0 + dgas) + thermal0 * dgas - p0 * dcv) / (cv0 + dcv);
}

}  // namespace kintera
