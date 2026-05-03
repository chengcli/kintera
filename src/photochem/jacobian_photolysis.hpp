#pragma once

// C/C++
#include <optional>

// torch
#include <torch/torch.h>

namespace kintera {

//! Compute the species-space Jacobian for photolysis reactions.
/*!
 * For photolysis reactions A + hν -> products, the rate law is:
 *   d[A]/dt = -k * [A]
 *
 * where k is the photolysis rate constant (depends on actinic flux and
 * cross-section, but not on concentration).
 *
 * This helper maps photolysis reaction rates into species production/loss
 * derivatives, i.e. d(dC_i/dt)/dC_j.
 *
 * \param rate       Photolysis rate constant [s^-1], shape (..., nreaction)
 * \param stoich     Stoichiometry matrix, shape (nspecies, nreaction)
 * \param conc       Concentration [mol/m^3], shape (..., nspecies)
 * \param rc_ddC     Rate constant derivative w.r.t. concentration,
 *                   shape (..., nspecies, nreaction) - typically zero for
 *                   photolysis
 * \param rc_ddT     Optional rate constant derivative w.r.t. temperature;
 *                   currently unused
 * \return           Species-space Jacobian d(dC_i/dt)/dC_j,
 *                   shape (..., nspecies, nspecies)
 */
torch::Tensor jacobian_photolysis_species(
    torch::Tensor rate, torch::Tensor stoich, torch::Tensor conc,
    torch::Tensor rc_ddC,
    torch::optional<torch::Tensor> rc_ddT = torch::nullopt);

}  // namespace kintera
