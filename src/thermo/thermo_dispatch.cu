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

extern std::vector<std::string> func1_names;
extern std::vector<std::string> func2_names;

void call_equilibrate_tp_cuda(at::TensorIterator &iter, int ngas,
                              at::Tensor const& stoich,
                              std::vector<std::string> const &logsvp_func,
                              double logsvp_eps, int max_iter) {
  at::cuda::CUDAGuard device_guard(iter.device());

  auto f1 = get_device_func1(logsvp_func, func1_names);
  auto logsvp_ptrs = f1.data().get();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_equilibrate_tp_cuda", [&] {
    int nspecies = at::native::ensure_nonempty_size(stoich, 0);
    int nreaction = at::native::ensure_nonempty_size(stoich, 1);

    auto stoich_ptr = stoich.data_ptr<scalar_t>();

    int mem_size = equilibrate_tp_space<scalar_t>(nspecies, nreaction);
    //std::cout << "mem size (bytes) = " << mem_size << std::endl;

    native::gpu_mem_kernel<32, 7>(
        iter, mem_size, [=] GPU_LAMBDA(
          char* const data[7], unsigned int strides[7], char* work) {
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
                       logsvp_eps, &max_iter_i, reaction_set,
                       nactive, work);
      });
  });
}

void call_equilibrate_uv_cuda(at::TensorIterator &iter, int ngas,
                             at::Tensor const& stoich,
                             at::Tensor const& intEng_offset,
                             at::Tensor const& cv_const,
                             std::vector<std::string> const &logsvp_func,
                             std::vector<std::string> const &intEng_extra_func,
                             double logsvp_eps, int max_iter) {
  at::cuda::CUDAGuard device_guard(iter.device());

  /////  (1) Get svp functions   /////
  auto f1a = get_device_func1(logsvp_func, func1_names);
  auto logsvp_ptrs = f1a.data().get();

  // transform the name of logsvp_func by appending "_ddT"
  auto logsvp_ddT_func = logsvp_func;
  for (auto &name : logsvp_ddT_func) name += "_ddT";

  auto f1b = get_device_func1(logsvp_ddT_func, func1_names);
  auto logsvp_ddT_ptrs = f1b.data().get();

  /////  (2) Get intEng_extra functions   /////
  auto f2a = get_device_func2(intEng_extra_func, func2_names);
  auto intEng_extra_ptrs = f2a.data().get();

  auto intEng_extra_ddT_func = intEng_extra_func;
  for (auto &name : intEng_extra_ddT_func) {
    if (!name.empty()) name += "_ddT";
  }
  auto f2b = get_device_func2(intEng_extra_ddT_func, func2_names);
  auto intEng_extra_ddT_ptrs = f2b.data().get();

  /////  (3) Launch kernel calculation    /////

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_equilibrate_uv_cuda", [&] {
    int nspecies = at::native::ensure_nonempty_size(stoich, 0);
    int nreaction = at::native::ensure_nonempty_size(stoich, 1);

    auto stoich_ptr = stoich.data_ptr<scalar_t>();
    auto intEng_offset_ptr = intEng_offset.data_ptr<scalar_t>();
    auto cv_const_ptr = cv_const.data_ptr<scalar_t>();

    int mem_size = equilibrate_uv_space<scalar_t>(nspecies, nreaction);
    //std::cout << "mem size (bytes) = " << mem_size << std::endl;

    native::gpu_mem_kernel<32, 7>(
        iter, mem_size, [=] GPU_LAMBDA(
          char* const data[7], unsigned int strides[7], char* work) {
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
                       intEng_extra_ptrs, intEng_extra_ddT_ptrs,
                       logsvp_eps, &max_iter_i, reaction_set,
                       nactive, work);
      });
  });
}

}  // namespace kintera

namespace at::native {

REGISTER_CUDA_DISPATCH(call_equilibrate_tp,
                       &kintera::call_equilibrate_tp_cuda);

REGISTER_CUDA_DISPATCH(call_equilibrate_uv,
                       &kintera::call_equilibrate_uv_cuda);

}  // namespace at::native
