// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// kintera
#include <kintera/loops.cuh>
#include "utils_dispatch.hpp"

namespace kintera {

// TODO(cli): Implmenet these functions

void call_func1_cuda(at::TensorIterator &iter, std::vector<std::string> const& funcs) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_func1_cuda", [&] {
      // do nothing
  });
}

void call_func2_cuda(at::TensorIterator &iter, user_func2 const *func) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_func2_cuda", [&] {
      // do nothing
  });
}

void call_func3_cuda(at::TensorIterator &iter, user_func3 const *func) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_func3_cuda", [&] {
      // do nothing
  });
}

}  // namespace kintera

namespace at::native {

REGISTER_CUDA_DISPATCH(call_func1, &kintera::call_func1_cuda);
REGISTER_CUDA_DISPATCH(call_func2, &kintera::call_func2_cuda);
REGISTER_CUDA_DISPATCH(call_func3, &kintera::call_func3_cuda);

}  // namespace at::native
