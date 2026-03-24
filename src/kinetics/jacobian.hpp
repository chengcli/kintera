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
    torch::Tensor temp, SpeciesThermo const& op,
    torch::optional<torch::Tensor> logrc_ddT = torch::nullopt);

torch::Tensor jacobian_evaporation(torch::Tensor rate, torch::Tensor stoich,
                                   torch::Tensor conc, torch::Tensor temp,
                                   ThermoOptions const& op);

//! Compute Jacobian for photolysis reactions
/*!
 * For photolysis reactions A + hν -> products, the rate law is:
 *   d[A]/dt = -k * [A]
 *
 * where k is the photolysis rate constant (depends on actinic flux and
 * cross-section, but not on concentration).
 *
 * The Jacobian entry for d(d[A]/dt)/d[A] = -k
 *
 * \param rate       Photolysis rate constant [s^-1], shape (..., nreaction)
 * \param stoich     Stoichiometry matrix, shape (nspecies, nreaction)
 * \param conc       Concentration [mol/m^3], shape (..., nspecies)
 * \param rc_ddC     Rate constant derivative w.r.t. concentration,
 *                   shape (..., nspecies, nreaction) - typically zero for
 *                   photolysis
 * \param rc_ddT     Optional rate constant derivative w.r.t. temperature
 * \return           Jacobian matrix, shape (..., nreaction, nspecies)
 */
torch::Tensor jacobian_photolysis(
    torch::Tensor rate, torch::Tensor stoich, torch::Tensor conc,
    torch::Tensor rc_ddC,
    torch::optional<torch::Tensor> rc_ddT = torch::nullopt);

}  // namespace kintera
