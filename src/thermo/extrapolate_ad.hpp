#pragma once

#include "thermo.hpp"

namespace kintera {

//! \brief Calculate effective heat capacity at constant pressure
/*!
 *
 * \param temp Temperature tensor (K)
 * \param pres Pressure tensor (Pa)
 * \param xfrac Mole fraction tensor
 * \param gain Gain tensor
 * \param thermo ThermoX object containing the thermodynamic model
 * \param conc Optional concentration tensor, if not provided it will be
 * computed
 * \return Equivalent heat capacity at constant pressure (Cp) tensor [J/(mol K)]
 */
torch::Tensor effective_cp_mole(
    torch::Tensor temp, torch::Tensor pres, torch::Tensor xfrac,
    torch::Tensor gain, ThermoX& thermo,
    torch::optional<torch::Tensor> conc = torch::nullopt);

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
