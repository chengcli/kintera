// torch
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// kintera
#include "utils_dispatch.hpp"

namespace kintera {

void call_func1_cpu(at::TensorIterator &iter,
                    std::vector<std::string> const &funcs) {
  int grain_size = iter.numel() / at::get_num_threads();
  auto f1 = get_host_func1(funcs);
  auto f1_ptrs = f1.data();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_func1_cpu", [&] {
    int nout = at::native::ensure_nonempty_size(iter.output(), -1);
    iter.for_each(
        [&](char **data, const int64_t *strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
            // temp
            auto arg1 = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
            for (int j = 0; j < nout; ++j)
              if (f1_ptrs[j]) out[j] += f1_ptrs[j](*arg1);
          }
        },
        grain_size);
  });
}

void call_func2_cpu(at::TensorIterator &iter,
                    std::vector<std::string> const &funcs) {
  int grain_size = iter.numel() / at::get_num_threads();
  auto f2 = get_host_func2(funcs);
  auto f2_ptrs = f2.data();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_func2_cpu", [&] {
    int nout = at::native::ensure_nonempty_size(iter.output(), -1);
    iter.for_each(
        [&](char **data, const int64_t *strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
            // temp
            auto arg1 = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
            // conc
            auto arg2 = reinterpret_cast<scalar_t *>(data[2] + i * strides[2]);
            for (int j = 0; j < nout; ++j)
              if (f2_ptrs[j]) out[j] += f2_ptrs[j](*arg1, arg2[j]);
          }
        },
        grain_size);
  });
}

void call_func3_cpu(at::TensorIterator &iter,
                    std::vector<std::string> const &funcs) {
  int grain_size = iter.numel() / at::get_num_threads();
  auto f3 = get_host_func3(funcs);
  auto f3_ptrs = f3.data();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_func3_cpu", [&] {
    int nout = at::native::ensure_nonempty_size(iter.output(), -1);
    iter.for_each(
        [&](char **data, const int64_t *strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
            // temp
            auto arg1 = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
            // pres
            auto arg2 = reinterpret_cast<scalar_t *>(data[2] + i * strides[2]);
            // conc
            auto arg3 = reinterpret_cast<scalar_t *>(data[3] + i * strides[3]);
            for (int j = 0; j < nout; ++j)
              if (f3_ptrs[j]) out[j] += f3_ptrs[j](*arg1, *arg2, arg3[j]);
          }
        },
        grain_size);
  });
}

}  // namespace kintera

namespace at::native {

DEFINE_DISPATCH(call_func1);
DEFINE_DISPATCH(call_func2);
DEFINE_DISPATCH(call_func3);

REGISTER_ALL_CPU_DISPATCH(call_func1, &kintera::call_func1_cpu);
REGISTER_ALL_CPU_DISPATCH(call_func2, &kintera::call_func2_cpu);
REGISTER_ALL_CPU_DISPATCH(call_func3, &kintera::call_func3_cpu);

}  // namespace at::native
