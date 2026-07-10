#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>

#include <kintera/loops.cuh>

#include "phase_equilibrate_tp.h"
#include "equilibrium_dispatch.hpp"

namespace kintera {

template <typename T>
size_t equilibrium_space(int nspecies, int nreaction, int nphase) {
  size_t bytes = 0;
  auto bump = [&](size_t align, size_t nbytes) {
    bytes = static_cast<size_t>(align_up(bytes, align)) + nbytes;
  };
  bump(alignof(T), nphase * sizeof(T));
  bump(alignof(T), nphase * nreaction * sizeof(T));
  bump(alignof(T), nreaction * sizeof(T));
  bump(alignof(T), nreaction * nreaction * sizeof(T));
  bump(alignof(T), nspecies * nreaction * sizeof(T));
  bump(alignof(T), nspecies * sizeof(T));
  bump(alignof(T), nreaction * sizeof(T));
  bump(alignof(T), nspecies * sizeof(T));
  return bytes + leastsq_kkt_space<T>(nreaction, nspecies);
}

void call_equilibrium_cuda(at::TensorIterator &iter, at::Tensor const &stoich,
                           at::Tensor const &phase_ids, int nphase,
                           int gas_phase, double standard_pressure, double ftol,
                           double mole_floor, int max_iter) {
  at::cuda::CUDAGuard device_guard(iter.device());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_equilibrium_cuda", [&] {
    int nspecies = stoich.size(0);
    int nreaction = stoich.size(1);
    auto stoich_ptr = stoich.data_ptr<scalar_t>();
    auto phase_ptr = phase_ids.data_ptr<int>();
    size_t work_size = equilibrium_space<scalar_t>(nspecies, nreaction, nphase);
    // Each equilibrium cell receives an independent global-memory workspace.
    native::gpu_global_mem_kernel<32, 7>(
        iter, work_size,
        [=] GPU_LAMBDA(char *const data[7], unsigned int strides[7],
                       char *work) {
          auto gain = reinterpret_cast<scalar_t *>(data[0] + strides[0]);
          auto diag = reinterpret_cast<scalar_t *>(data[1] + strides[1]);
          auto out = reinterpret_cast<scalar_t *>(data[2] + strides[2]);
          auto temp = reinterpret_cast<scalar_t *>(data[3] + strides[3]);
          auto pres = reinterpret_cast<scalar_t *>(data[4] + strides[4]);
          auto moles = reinterpret_cast<scalar_t *>(data[5] + strides[5]);
          auto log_k = reinterpret_cast<scalar_t *>(data[6] + strides[6]);
          phase_equilibrate_tp(
              gain, diag, out, *temp, *pres, moles, log_k, stoich_ptr,
              phase_ptr, nspecies, nreaction, nphase, gas_phase,
              static_cast<scalar_t>(standard_pressure),
              static_cast<scalar_t>(ftol), static_cast<scalar_t>(mole_floor),
              max_iter, work);
        });
  });
}

} // namespace kintera

namespace at::native {

REGISTER_CUDA_DISPATCH(call_equilibrium, &kintera::call_equilibrium_cuda);

} // namespace at::native
