// C/C++
#include <vector>

// torch
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>

// kintera
#include "evolve_implicit_dispatch.hpp"
#include "evolve_implicit_impl.h"

namespace kintera {

void call_evolve_implicit_cpu(at::TensorIterator& iter,
                              at::Tensor const& stoich, double dt) {
  int grain_size = iter.numel() / std::max(1, at::get_num_threads());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_evolve_implicit_cpu", [&] {
    int nspecies = at::native::ensure_nonempty_size(stoich, 0);
    int nreaction = at::native::ensure_nonempty_size(stoich, 1);

    auto stoich_ptr = stoich.data_ptr<scalar_t>();
    scalar_t inv_dt = static_cast<scalar_t>(1.0 / dt);
    size_t mem_size = evolve_implicit_space<scalar_t>(nspecies, nreaction);

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          std::vector<char> work(mem_size);
          for (int i = 0; i < n; i++) {
            auto delta = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto rate = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto jac = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            evolve_implicit_cell(delta, rate, jac, stoich_ptr, nspecies,
                                 nreaction, inv_dt, work.data());
          }
        },
        grain_size);
  });
}

}  // namespace kintera

namespace at::native {

DEFINE_DISPATCH(call_evolve_implicit);

REGISTER_ALL_CPU_DISPATCH(call_evolve_implicit,
                          &kintera::call_evolve_implicit_cpu);

}  // namespace at::native
