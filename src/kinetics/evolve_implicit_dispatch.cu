// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// kintera
#include <kintera/loops.cuh>

#include "evolve_implicit_dispatch.hpp"
#include "evolve_implicit_impl.h"

namespace kintera {

void call_evolve_implicit_cuda(at::TensorIterator& iter,
                               at::Tensor const& stoich, double dt) {
  at::cuda::CUDAGuard device_guard(iter.device());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_evolve_implicit_cuda", [&] {
    int nspecies = at::native::ensure_nonempty_size(stoich, 0);
    int nreaction = at::native::ensure_nonempty_size(stoich, 1);

    auto stoich_ptr = stoich.data_ptr<scalar_t>();
    scalar_t inv_dt = static_cast<scalar_t>(1.0 / dt);
    int mem_size = evolve_implicit_space<scalar_t>(nspecies, nreaction);

    native::gpu_mem_kernel<32, 3>(
        iter, mem_size,
        [=] GPU_LAMBDA(char* const data[3], unsigned int strides[3],
                       char* work) {
          auto delta = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          auto rate = reinterpret_cast<scalar_t*>(data[1] + strides[1]);
          auto jac = reinterpret_cast<scalar_t*>(data[2] + strides[2]);
          evolve_implicit_cell(delta, rate, jac, stoich_ptr, nspecies,
                               nreaction, inv_dt, work);
        });
  });
}

}  // namespace kintera

namespace at::native {

REGISTER_CUDA_DISPATCH(call_evolve_implicit,
                       &kintera::call_evolve_implicit_cuda);

}  // namespace at::native
