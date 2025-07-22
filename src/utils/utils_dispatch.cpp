// torch
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// kintera
#include "utils_dispatch.hpp"

namespace kintera {

void call_func1_cpu(at::TensorIterator &iter, user_func1 const *func) {
  int grain_size = iter.numel() / at::get_num_threads();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_func1_cpu", [&] {
    int nout = at::native::ensure_nonempty_size(iter.output(), -1);
    iter.for_each(
        [&](char **data, const int64_t *strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
            // temp
            auto arg1 = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
            for (int j = 0; j < nout; ++j)
              if (func[j]) out[j] += func[j](*arg1);
          }
        },
        grain_size);
  });
}

void call_func2_cpu(at::TensorIterator &iter, user_func2 const *func) {
  int grain_size = iter.numel() / at::get_num_threads();

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
              if (func[j]) out[j] += func[j](*arg1, arg2[j]);
          }
        },
        grain_size);
  });
}

void call_func3_cpu(at::TensorIterator &iter, user_func3 const *func) {
  int grain_size = iter.numel() / at::get_num_threads();

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
              if (func[j]) out[j] += func[j](*arg1, *arg2, arg3[j]);
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
