#pragma once

#include "thermo.hpp"

namespace kintera {

torch::Tensor effective_cp_mole(torch::Tensor temp, torch::Tensor pres,
                                torch::Tensor xfrac, ThermoX& thermo,
                                double dT = 0.01);

//! \brief Extrapolate state TPX to a new pressure along an adiabat
/*!
 * Extrapolates the state variables (temperature, pressure, and mole fractions)
 *
 * \param[in,out] temp Temperature tensor (K)
 * \param[in,out] pres Pressure tensor (Pa)
 * \param[in,out] xfrac Mole fraction tensor
 * \param[in] thermo ThermoX object containing the thermodynamic model
 * \param[in] dlnp Logarithmic change in pressure (dlnp = ln(p_new / p_old))
 */
void extrapolate_ad_(torch::Tensor temp, torch::Tensor pres,
                     torch::Tensor xfrac, ThermoX& thermo, double dlnp,
                     double ftol = 1.e-5);

}  // namespace kintera
