// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// kintera
#include <kintera/utils/user_funcs.hpp>
#include <kintera/loops.cuh>
#include "utils_dispatch.hpp"

namespace kintera {

extern std::vector<std::string> func1_names;
extern std::vector<std::string> func2_names;
extern std::vector<std::string> func3_names;

void call_func1_cuda(at::TensorIterator &iter,
                     std::vector<std::string> const& funcs) {
  at::cuda::CUDAGuard device_guard(iter.device());

  auto f1 = get_device_func1(funcs, func1_names);
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

void call_func2_cuda(at::TensorIterator &iter,
                     std::vector<std::string> const& funcs) {
  at::cuda::CUDAGuard device_guard(iter.device());

  auto f2 = get_device_func2(funcs, func2_names);
  auto f2_ptrs = f2.data().get();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_func2_cuda", [&] {
      int nout = at::native::ensure_nonempty_size(iter.output(), -1);

      native::gpu_kernel<3>(
          iter, [=] GPU_LAMBDA (char* const data[3], unsigned int strides[3]) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
            // temp
            auto arg1 = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
            // conc
            auto arg2 = reinterpret_cast<scalar_t*>(data[2] + strides[2]);

            for (int j = 0; j < nout; ++j) {
              if (f2_ptrs[j]) out[j] += f2_ptrs[j](*arg1, arg2[j]);
            }
          });
  });
}

void call_func3_cuda(at::TensorIterator &iter, std::vector<std::string> const& funcs) {
  at::cuda::CUDAGuard device_guard(iter.device());

  auto f3 = get_device_func3(funcs, func3_names);
  auto f3_ptrs = f3.data().get();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_func3_cuda", [&] {
      int nout = at::native::ensure_nonempty_size(iter.output(), -1);

      native::gpu_kernel<4>(
          iter, [=] GPU_LAMBDA (char* const data[4], unsigned int strides[4]) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
            // temp
            auto arg1 = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
            // pres
            auto arg2 = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
            // conc
            auto arg3 = reinterpret_cast<scalar_t*>(data[3] + strides[3]);

            for (int j = 0; j < nout; ++j) {
              if (f3_ptrs[j]) out[j] += f3_ptrs[j](*arg1, *arg2, arg3[j]);
            }
          });
  });
}

}  // namespace kintera

namespace at::native {

DEFINE_DISPATCH(call_func1);
DEFINE_DISPATCH(call_func2);
DEFINE_DISPATCH(call_func3);

REGISTER_CUDA_DISPATCH(call_func1, &kintera::call_func1_cuda);
REGISTER_CUDA_DISPATCH(call_func2, &kintera::call_func2_cuda);
REGISTER_CUDA_DISPATCH(call_func3, &kintera::call_func3_cuda);

}  // namespace at::native
