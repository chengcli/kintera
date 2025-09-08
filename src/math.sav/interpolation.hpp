#pragma once

// torch
#include <torch/torch.h>

namespace kintera {

//! Multidimensional linear interpolation
/*!
 * \param query_coords Query coordinates (nbatch, ndim)
 * \param coords Coordinate arrays, len = ndim, each tensor has shape (nx1,),
 * (nx2,) ...
 * \param lookup Lookup tensor (nx1, nx2, ..., nval)
 * \return Interpolated values (nbatch, nval)
 */
torch::Tensor interpn(std::vector<torch::Tensor> const& query_coords,
                      std::vector<torch::Tensor> const& coords,
                      torch::Tensor const& lookup, bool extrapolate = false);

}  // namespace kintera
