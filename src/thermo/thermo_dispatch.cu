// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/cuda/CUDAGuard.h>

// kintera
#include <kintera/utils/user_funcs.hpp>
#include <kintera/loops.cuh>
#include "equilibrate_tp.h"
#include "equilibrate_uv.h"
#include "thermo_dispatch.hpp"

namespace kintera {

void call_equilibrate_tp_cuda(at::TensorIterator &iter, int ngas,
                              at::Tensor const& stoich,
                              ThermoTables const& t,
                              double logsvp_eps, int max_iter) {
  at::cuda::CUDAGuard device_guard(iter.device());

  // device function pointers resolved when the table was built
  auto logsvp_ptrs = static_cast<user_func1*>(t.logsvp.data_ptr());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_equilibrate_tp_cuda", [&] {
    int nspecies = at::native::ensure_nonempty_size(stoich, 0);
    int nreaction = at::native::ensure_nonempty_size(stoich, 1);

    auto stoich_ptr = stoich.data_ptr<scalar_t>();
    auto svp_kind_ptr = t.svp_kind.data_ptr<int>();
    auto svp_params_ptr = t.svp_params.data_ptr<double>();

    size_t mem_size =
        pool_workspace_bytes(equilibrate_tp_space<scalar_t>(nspecies, nreaction));
    native::gpu_mem_kernel<32, 7>(
        iter, mem_size,
        [=] GPU_LAMBDA(char* const data[7], unsigned int strides[7],
                       char* work) {
        auto gain = reinterpret_cast<scalar_t *>(data[0] + strides[0]);
        auto diag = reinterpret_cast<scalar_t *>(data[1] + strides[1]);
        auto xfrac = reinterpret_cast<scalar_t *>(data[2] + strides[2]);
        auto temp = reinterpret_cast<scalar_t *>(data[3] + strides[3]);
        auto pres = reinterpret_cast<scalar_t *>(data[4] + strides[4]);
        auto reaction_set = reinterpret_cast<int*>(data[5] + strides[5]);
        auto nactive = reinterpret_cast<int*>(data[6] + strides[6]);
        int max_iter_i = max_iter;
        equilibrate_tp(gain, diag, xfrac, *temp, *pres,
                       stoich_ptr, nspecies,
                       nreaction, ngas, logsvp_ptrs,
                       svp_kind_ptr, svp_params_ptr,
                       logsvp_eps, &max_iter_i, reaction_set,
                       nactive, work);
      });
  });
}

void call_equilibrate_uv_cuda(at::TensorIterator &iter, int ngas,
                             at::Tensor const& stoich,
                             at::Tensor const& intEng_offset,
                             at::Tensor const& cv_const,
                             ThermoTables const& t,
                             double logsvp_eps, int max_iter, int uv_solver) {
  at::cuda::CUDAGuard device_guard(iter.device());

  /////  (1) svp and intEng_extra functions, resolved when the table was built
  auto logsvp_ptrs = static_cast<user_func1*>(t.logsvp.data_ptr());
  auto logsvp_ddT_ptrs = static_cast<user_func1*>(t.logsvp_ddT.data_ptr());
  auto intEng_extra_ptrs = static_cast<user_func2*>(t.intEng_extra.data_ptr());
  auto intEng_extra_ddT_ptrs =
      static_cast<user_func2*>(t.intEng_extra_ddT.data_ptr());

  /////  (2) Launch kernel calculation    /////

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_equilibrate_uv_cuda", [&] {
    int nspecies = at::native::ensure_nonempty_size(stoich, 0);
    int nreaction = at::native::ensure_nonempty_size(stoich, 1);

    auto stoich_ptr = stoich.data_ptr<scalar_t>();
    auto intEng_offset_ptr = intEng_offset.data_ptr<scalar_t>();
    auto cv_const_ptr = cv_const.data_ptr<scalar_t>();
    auto svp_kind_ptr = t.svp_kind.data_ptr<int>();
    auto svp_params_ptr = t.svp_params.data_ptr<double>();

    size_t mem_size = pool_workspace_bytes(
        uv_solver == 2
            ? equilibrate_uv_partition_space<scalar_t>(nspecies, nreaction)
            : equilibrate_uv_space<scalar_t>(nspecies, nreaction));
    native::gpu_mem_kernel<32, 7>(
        iter, mem_size,
        [=] GPU_LAMBDA(char* const data[7], unsigned int strides[7],
                       char* work) {
        auto gain = reinterpret_cast<scalar_t *>(data[0] + strides[0]);
        auto diag = reinterpret_cast<scalar_t *>(data[1] + strides[1]);
        auto conc = reinterpret_cast<scalar_t *>(data[2] + strides[2]);
        auto temp = reinterpret_cast<scalar_t *>(data[3] + strides[3]);
        auto intEng = reinterpret_cast<scalar_t *>(data[4] + strides[4]);
        auto reaction_set = reinterpret_cast<int*>(data[5] + strides[5]);
        auto nactive = reinterpret_cast<int*>(data[6] + strides[6]);
        int max_iter_i = max_iter;
        equilibrate_uv(gain, diag, temp, conc, *intEng,
                       stoich_ptr, nspecies, nreaction, ngas,
                       intEng_offset_ptr, cv_const_ptr,
                       logsvp_ptrs, logsvp_ddT_ptrs,
                       svp_kind_ptr, svp_params_ptr,
                       intEng_extra_ptrs, intEng_extra_ddT_ptrs,
                       logsvp_eps, &max_iter_i, reaction_set,
                       nactive, uv_solver, work);
      });
  });
}

void call_logsvp_inline_cuda(at::TensorIterator &iter,
                             at::Tensor const& svp_kind,
                             at::Tensor const& svp_params, bool deriv) {
  at::cuda::CUDAGuard device_guard(iter.device());

  auto svp_kind_ptr = svp_kind.data_ptr<int>();
  auto svp_params_ptr = svp_params.data_ptr<double>();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_logsvp_inline_cuda", [&] {
    int nout = at::native::ensure_nonempty_size(iter.output(), -1);

    native::gpu_kernel<2>(
        iter, [=] GPU_LAMBDA (char* const data[2], unsigned int strides[2]) {
          auto out = reinterpret_cast<scalar_t*>(data[0] + strides[0]);
          // temp
          auto arg1 = reinterpret_cast<scalar_t*>(data[1] + strides[1]);

          for (int j = 0; j < nout; ++j) {
            if (svp_kind_ptr[j] == 0) continue;
            double const* p = svp_params_ptr + j * KSVP_NPARAM;
            out[j] = deriv ? eval_logsvp_ddT(svp_kind_ptr[j], p, nullptr, *arg1)
                           : eval_logsvp(svp_kind_ptr[j], p, nullptr, *arg1);
          }
        });
  });
}

}  // namespace kintera

namespace at::native {

REGISTER_CUDA_DISPATCH(call_equilibrate_tp,
                       &kintera::call_equilibrate_tp_cuda);

REGISTER_CUDA_DISPATCH(call_equilibrate_uv,
                       &kintera::call_equilibrate_uv_cuda);

REGISTER_CUDA_DISPATCH(call_logsvp_inline,
                       &kintera::call_logsvp_inline_cuda);

}  // namespace at::native
