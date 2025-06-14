#pragma once

#include "thermo.hpp"

namespace kintera {

//! \brief Calculate effective heat capacity at constant pressure
/*!
 *
 * \param[in] temp Temperature tensor (K)
 * \param[in] pres Pressure tensor (Pa)
 * \param[in] xfrac Mole fraction tensor
 * \param[in] weight Weight tensor
 * \param[in] thermo ThermoX object containing the thermodynamic model
 * \return Equivalent heat capacity at constant pressure (Cp) tensor [J/(mol K)]
 */
torch::Tensor effective_cp_mole(torch::Tensor temp, torch::Tensor pres,
                                torch::Tensor xfrac, torch::Tensor weight,
                                ThermoX& thermo);

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
                     torch::Tensor xfrac, ThermoX& thermo, double dlnp);

}  // namespace kintera
