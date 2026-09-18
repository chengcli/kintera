#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>

#include <disort/loops.cuh>

#include "phase_equilibrate_tp.h"
#include "equilibrium_dispatch.hpp"

namespace kintera {

template <typename T>
size_t equilibrium_space(int nspecies, int nreaction, int nphase) {
  size_t bytes = 0;
  auto bump = [&](size_t, size_t nbytes) {
    bytes += pool_allocation_bytes(nbytes);
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
    size_t work_size = pool_workspace_bytes(
        pmem::STATE_SIZE + equilibrium_space<scalar_t>(nspecies, nreaction,
                                                        nphase));
    at::Tensor workspace;
    disort::native::gpu_chunk_kernel<7>(
        iter, work_size, &workspace,
        [=] GPU_LAMBDA(char *const data[7], unsigned int strides[7], int64_t) {
          pmem::pool_init();
          auto gain = reinterpret_cast<scalar_t *>(data[0] + strides[0]);
          auto diag = reinterpret_cast<scalar_t *>(data[1] + strides[1]);
          auto out = reinterpret_cast<scalar_t *>(data[2] + strides[2]);
          auto temp = reinterpret_cast<scalar_t *>(data[3] + strides[3]);
          auto pres = reinterpret_cast<scalar_t *>(data[4] + strides[4]);
          auto moles = reinterpret_cast<scalar_t *>(data[5] + strides[5]);
          auto log_k = reinterpret_cast<scalar_t *>(data[6] + strides[6]);
          phase_equilibrate_tp<scalar_t, PoolBackend::DisortGlobal>(
              gain, diag, out, *temp, *pres, moles, log_k, stoich_ptr,
              phase_ptr, nspecies, nreaction, nphase, gas_phase,
              static_cast<scalar_t>(standard_pressure),
              static_cast<scalar_t>(ftol), static_cast<scalar_t>(mole_floor),
              max_iter);
        });
  });
}

} // namespace kintera

namespace at::native {

REGISTER_CUDA_DISPATCH(call_equilibrium, &kintera::call_equilibrium_cuda);

} // namespace at::native
