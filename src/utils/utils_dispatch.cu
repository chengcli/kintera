// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// kintera
#include <kintera/utils/func1.hpp>
#include <kintera/loops.cuh>
#include "utils_dispatch.hpp"

namespace kintera {

// TODO(cli): Implmenet these functions

void call_func1_cuda(at::TensorIterator &iter, std::vector<std::string> const& funcs) {
  at::cuda::CUDAGuard device_guard(iter.device());

  auto f1 = get_device_func1(funcs);
  auto f1_ptrs = f1.data().get();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_func1_cuda", [&] {
    int nout = at::native::ensure_nonempty_size(iter.output(), -1);

    native::gpu_kernel<2>(
        iter, [=] GPU_LAMBDA (char* const data[2], unsigned int strides[2]) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          // temp 
          auto arg1 = reinterpret_cast<scalar_t*>(data[1] + strides[1]);

          for (int j = 0; j < nout; ++j) {
            if (f1_ptrs[j]) out[j] += f1_ptrs[j](*arg1);
          }
        });

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
