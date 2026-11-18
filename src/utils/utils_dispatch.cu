// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// kintera
#include <kintera/loops.cuh>
#include <kintera/thermo/thermo.hpp>
#include "user_funcs.hpp"
#include "utils_dispatch.hpp"

namespace kintera {

void call_func1_cuda(at::TensorIterator &iter, at::Tensor const& funcs) {
  at::cuda::CUDAGuard device_guard(iter.device());
  auto f1_ptrs = static_cast<user_func1*>(funcs.data_ptr());

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

void call_func2_cuda(at::TensorIterator &iter, at::Tensor const& funcs) {
  at::cuda::CUDAGuard device_guard(iter.device());
  auto f2_ptrs = static_cast<user_func2*>(funcs.data_ptr());

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

void call_func3_cuda(at::TensorIterator &iter, at::Tensor const& funcs) {
  at::cuda::CUDAGuard device_guard(iter.device());
  auto f3_ptrs = static_cast<user_func3*>(funcs.data_ptr());

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

FuncTable get_device_funcs(std::vector<std::string> const& names, int arity,
                           at::Device device);  // user_funcs.cu

// the CUDA kernels of the two stubs; a table build also runs the lazy
// shared-memory query of loops.cuh, so it happens before any capture
FuncTable fill_func_table_cuda(std::vector<std::string> const& names,
                               int arity, at::Device device) {
  at::cuda::CUDAGuard device_guard(device);
  native::get_max_dynamic_shared_memory(device.index());
  return get_device_funcs(names, arity, device);
}

bool capture_active_cuda(at::Device device) {
  at::cuda::CUDAGuard device_guard(device);
  return at::cuda::currentStreamCaptureStatusMayInitCtx() !=
         at::cuda::CaptureStatus::None;
}

}  // namespace kintera

namespace at::native {

REGISTER_CUDA_DISPATCH(call_func1, &kintera::call_func1_cuda);
REGISTER_CUDA_DISPATCH(call_func2, &kintera::call_func2_cuda);
REGISTER_CUDA_DISPATCH(call_func3, &kintera::call_func3_cuda);
REGISTER_CUDA_DISPATCH(fill_func_table, &kintera::fill_func_table_cuda);
REGISTER_CUDA_DISPATCH(capture_active, &kintera::capture_active_cuda);

}  // namespace at::native
