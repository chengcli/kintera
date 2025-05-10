#pragma once

#include <torch/torch.h>

namespace kintera {

/**
 * @brief Explicit solver for chemical kinetics
 *
 * @param C Concentration tensor with shape (..., n_species)
 * @param rates Reaction rates tensor with shape (..., n_reactions)
 * @param stoich_matrix Stoichiometry matrix with shape (n_reactions, n_species)
 * @param dt Timestep
 * @param max_rel_change Maximum allowed relative change in concentration
 * (default: 0.1)
 */
void explicit_solve(torch::Tensor& C, const torch::Tensor& rates,
                    const torch::Tensor& stoich_matrix, double dt,
                    double max_rel_change = 0.1);

}  // namespace kintera
