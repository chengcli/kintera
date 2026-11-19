// torch
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// kintera
#include <kintera/thermo/thermo.hpp>

#include "user_funcs.hpp"
#include "utils_dispatch.hpp"

namespace kintera {

extern user_func1 func1_table_cpu[];
extern std::vector<std::string> func1_names;
extern user_func2 func2_table_cpu[];
extern std::vector<std::string> func2_names;
extern user_func3 func3_table_cpu[];
extern std::vector<std::string> func3_names;

void call_func1_cpu(at::TensorIterator& iter, at::Tensor const& funcs) {
  int grain_size = iter.numel() / at::get_num_threads();
  auto f1_ptrs = static_cast<user_func1*>(funcs.data_ptr());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_func1_cpu", [&] {
    int nout = at::native::ensure_nonempty_size(iter.output(), -1);
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            // temp
            auto arg1 = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            for (int j = 0; j < nout; ++j)
              if (f1_ptrs[j]) out[j] += f1_ptrs[j](*arg1);
          }
        },
        grain_size);
  });
}

void call_func2_cpu(at::TensorIterator& iter, at::Tensor const& funcs) {
  int grain_size = iter.numel() / at::get_num_threads();
  auto f2_ptrs = static_cast<user_func2*>(funcs.data_ptr());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_func2_cpu", [&] {
    int nout = at::native::ensure_nonempty_size(iter.output(), -1);
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            // temp
            auto arg1 = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            // conc
            auto arg2 = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            for (int j = 0; j < nout; ++j)
              if (f2_ptrs[j]) out[j] += f2_ptrs[j](*arg1, arg2[j]);
          }
        },
        grain_size);
  });
}

void call_func3_cpu(at::TensorIterator& iter, at::Tensor const& funcs) {
  int grain_size = iter.numel() / at::get_num_threads();
  auto f3_ptrs = static_cast<user_func3*>(funcs.data_ptr());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_func3_cpu", [&] {
    int nout = at::native::ensure_nonempty_size(iter.output(), -1);
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            // temp
            auto arg1 = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            // pres
            auto arg2 = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            // conc
            auto arg3 = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
            for (int j = 0; j < nout; ++j)
              if (f3_ptrs[j]) out[j] += f3_ptrs[j](*arg1, *arg2, arg3[j]);
          }
        },
        grain_size);
  });
}

// the stubs' CPU kernels: host pointers via get_host_func; no CPU capture
FuncTable fill_func_table_cpu(std::vector<std::string> const& names, int arity,
                              at::Device) {
  auto host = [&](auto* table, std::vector<std::string> const& all) {
    auto f = get_host_func(names, all, table);  // pointers fit in int64
    return at::from_blob(f.data(), {(int64_t)f.size()}, at::kLong).clone();
  };
  return arity == 1   ? host(func1_table_cpu, func1_names)
         : arity == 2 ? host(func2_table_cpu, func2_names)
                      : host(func3_table_cpu, func3_names);
}
bool capture_active_cpu(at::Device) { return false; }

}  // namespace kintera

namespace at::native {

DEFINE_DISPATCH(call_func1);
DEFINE_DISPATCH(call_func2);
DEFINE_DISPATCH(call_func3);
DEFINE_DISPATCH(fill_func_table);
DEFINE_DISPATCH(capture_active);
REGISTER_ALL_CPU_DISPATCH(call_func1, &kintera::call_func1_cpu);
REGISTER_ALL_CPU_DISPATCH(call_func2, &kintera::call_func2_cpu);
REGISTER_ALL_CPU_DISPATCH(call_func3, &kintera::call_func3_cpu);
REGISTER_ALL_CPU_DISPATCH(fill_func_table, &kintera::fill_func_table_cpu);
REGISTER_ALL_CPU_DISPATCH(capture_active, &kintera::capture_active_cpu);

}  // namespace at::native
