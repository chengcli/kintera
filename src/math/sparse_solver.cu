// torch
#include <memory>

#include <torch/types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

// cuda
#include <cuda_runtime_api.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include "sparse_solver.hpp"

namespace kintera {

namespace {

void check_cuda(cudaError_t status, const char* msg) {
  TORCH_CHECK(status == cudaSuccess, msg, ": ", cudaGetErrorString(status));
}

void check_cusolver(cusolverStatus_t status, const char* msg) {
  TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS, msg, " failed with status ",
              static_cast<int>(status));
}

void check_cusparse(cusparseStatus_t status, const char* msg) {
  TORCH_CHECK(status == CUSPARSE_STATUS_SUCCESS, msg, " failed with status ",
              static_cast<int>(status));
}

struct CuSolverSpHandleDeleter {
  void operator()(std::remove_pointer_t<cusolverSpHandle_t>* handle) const {
    if (handle != nullptr) {
      cusolverSpDestroy(handle);
    }
  }
};

struct CuSparseMatDescrDeleter {
  void operator()(std::remove_pointer_t<cusparseMatDescr_t>* descr) const {
    if (descr != nullptr) {
      cusparseDestroyMatDescr(descr);
    }
  }
};

}  // namespace

torch::Tensor cuda_csr_solve_cusolver(const torch::Tensor& crow_indices,
                                      const torch::Tensor& col_indices,
                                      const torch::Tensor& values,
                                      const torch::Tensor& rhs, double tol,
                                      int reorder) {
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(rhs.is_cuda(), "rhs must be a CUDA tensor");
  TORCH_CHECK(crow_indices.is_cuda(), "crow_indices must be a CUDA tensor");
  TORCH_CHECK(col_indices.is_cuda(), "col_indices must be a CUDA tensor");
  TORCH_CHECK(values.dim() == 1, "values must be 1D");
  TORCH_CHECK(rhs.dim() == 1, "rhs must be 1D");
  TORCH_CHECK(crow_indices.dim() == 1, "crow_indices must be 1D");
  TORCH_CHECK(col_indices.dim() == 1, "col_indices must be 1D");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32 ||
                  values.scalar_type() == torch::kFloat64,
              "values must be float32 or float64");
  TORCH_CHECK(rhs.scalar_type() == values.scalar_type(),
              "rhs dtype must match values dtype");
  TORCH_CHECK(crow_indices.scalar_type() == torch::kInt32,
              "crow_indices must be int32");
  TORCH_CHECK(col_indices.scalar_type() == torch::kInt32,
              "col_indices must be int32");

  auto device = values.device();
  c10::cuda::CUDAGuard device_guard(device);
  auto stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();

  auto crow = crow_indices.contiguous();
  auto col = col_indices.contiguous();
  auto val = values.contiguous();
  auto b = rhs.contiguous();

  const auto n = static_cast<int>(b.numel());
  TORCH_CHECK(crow.numel() == n + 1, "crow_indices must have length n + 1");
  TORCH_CHECK(col.numel() == val.numel(),
              "col_indices must have the same length as values");
  TORCH_CHECK(crow.device() == device && col.device() == device &&
                  b.device() == device,
              "all tensors must live on the same CUDA device");
  TORCH_CHECK(crow[crow.numel() - 1].item<int>() == val.numel(),
              "CSR nnz mismatch");
  auto crow_host = crow.cpu();
  auto crow_ptr = crow_host.data_ptr<int>();
  for (int i = 0; i < crow_host.numel() - 1; ++i) {
    TORCH_CHECK(crow_ptr[i] <= crow_ptr[i + 1],
                "crow_indices must be nondecreasing");
  }

  auto x = torch::zeros_like(b);

  cusolverSpHandle_t solver_handle_raw = nullptr;
  cusparseMatDescr_t descr_raw = nullptr;
  check_cusolver(cusolverSpCreate(&solver_handle_raw), "cusolverSpCreate");
  std::unique_ptr<std::remove_pointer_t<cusolverSpHandle_t>,
                  CuSolverSpHandleDeleter>
      solver_handle(solver_handle_raw);
  check_cusolver(cusolverSpSetStream(solver_handle.get(), stream),
                 "cusolverSpSetStream");
  check_cusparse(cusparseCreateMatDescr(&descr_raw), "cusparseCreateMatDescr");
  std::unique_ptr<std::remove_pointer_t<cusparseMatDescr_t>,
                  CuSparseMatDescrDeleter>
      descr(descr_raw);
  check_cusparse(cusparseSetMatType(descr.get(), CUSPARSE_MATRIX_TYPE_GENERAL),
                 "cusparseSetMatType");
  check_cusparse(cusparseSetMatIndexBase(descr.get(), CUSPARSE_INDEX_BASE_ZERO),
                 "cusparseSetMatIndexBase");

  int singularity = -1;
  if (val.scalar_type() == torch::kFloat64) {
    check_cusolver(cusolverSpDcsrlsvqr(
                       solver_handle.get(), n, static_cast<int>(val.numel()),
                       descr.get(),
                       val.data_ptr<double>(), crow.data_ptr<int>(),
                       col.data_ptr<int>(), b.data_ptr<double>(), tol, reorder,
                       x.data_ptr<double>(), &singularity),
                   "cusolverSpDcsrlsvqr");
  } else {
    check_cusolver(cusolverSpScsrlsvqr(
                       solver_handle.get(), n, static_cast<int>(val.numel()),
                       descr.get(),
                       val.data_ptr<float>(), crow.data_ptr<int>(),
                       col.data_ptr<int>(), b.data_ptr<float>(),
                       static_cast<float>(tol), reorder, x.data_ptr<float>(),
                       &singularity),
                   "cusolverSpScsrlsvqr");
  }

  check_cuda(cudaGetLastError(), "CUDA sparse solve");
  TORCH_CHECK(singularity < 0,
              "cusolver sparse solve detected singularity at row ",
              singularity);
  return x;
}

}  // namespace kintera

extern "C" torch::Tensor kintera_cuda_csr_solve_cusolver(
    const torch::Tensor& crow_indices, const torch::Tensor& col_indices,
    const torch::Tensor& values, const torch::Tensor& rhs, double tol,
    int reorder) {
  return kintera::cuda_csr_solve_cusolver(crow_indices, col_indices, values,
                                          rhs, tol, reorder);
}
