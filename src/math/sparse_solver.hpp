#pragma once

#include <torch/types.h>

namespace kintera {

torch::Tensor cuda_csr_solve_cusolver(const torch::Tensor& crow_indices,
                                      const torch::Tensor& col_indices,
                                      const torch::Tensor& values,
                                      const torch::Tensor& rhs, double tol,
                                      int reorder);

}  // namespace kintera
