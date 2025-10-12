// C/C++
#include <algorithm>

// torch
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>

// kintera
#include <kintera/utils/user_funcs.hpp>

#include "equilibrate_tp.h"
#include "equilibrate_uv.h"
#include "thermo_dispatch.hpp"

namespace kintera {

extern user_func1 func1_table_cpu[];
extern std::vector<std::string> func1_names;

extern user_func2 func2_table_cpu[];
extern std::vector<std::string> func2_names;

void call_equilibrate_tp_cpu(at::TensorIterator &iter, int ngas,
                             at::Tensor const &stoich,
                             std::vector<std::string> const &logsvp_func,
                             double logsvp_eps, int max_iter) {
  int grain_size = iter.numel() / at::get_num_threads();

  auto f1 = get_host_func(logsvp_func, func1_names, func1_table_cpu);
  auto logsvp_ptrs = f1.data();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_equilibrate_tp_cpu", [&] {
    int nspecies = at::native::ensure_nonempty_size(stoich, 0);
    int nreaction = at::native::ensure_nonempty_size(stoich, 1);

    auto stoich_ptr = stoich.data_ptr<scalar_t>();

    iter.for_each(
        [&](char **data, const int64_t *strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto gain = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
            auto diag = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
            auto xfrac = reinterpret_cast<scalar_t *>(data[2] + i * strides[2]);
            auto temp = reinterpret_cast<scalar_t *>(data[3] + i * strides[3]);
            auto pres = reinterpret_cast<scalar_t *>(data[4] + i * strides[4]);
            auto reaction_set =
                reinterpret_cast<int *>(data[5] + i * strides[5]);
            auto nactive = reinterpret_cast<int *>(data[6] + i * strides[6]);
            int max_iter_i = max_iter;
            equilibrate_tp(gain, diag, xfrac, *temp, *pres, stoich_ptr,
                           nspecies, nreaction, ngas, logsvp_ptrs, logsvp_eps,
                           &max_iter_i, reaction_set, nactive);
          }
        },
        grain_size);
  });
}

void call_equilibrate_uv_cpu(at::TensorIterator &iter, int ngas,
                             at::Tensor const &stoich,
                             at::Tensor const &intEng_offset,
                             at::Tensor const &cv_const,
                             std::vector<std::string> const &logsvp_func,
                             std::vector<std::string> const &intEng_extra_func,
                             double logsvp_eps, int max_iter) {
  int grain_size = iter.numel() / at::get_num_threads();

  /////  (1) Get svp functions   /////
  auto f1a = get_host_func(logsvp_func, func1_names, func1_table_cpu);
  auto logsvp_ptrs = f1a.data();

  // transform the name of logsvp_func by appending "_ddT"
  auto logsvp_ddT_func = logsvp_func;
  for (auto &name : logsvp_ddT_func) name += "_ddT";

  auto f1b = get_host_func(logsvp_ddT_func, func1_names, func1_table_cpu);
  auto logsvp_ddT_ptrs = f1b.data();

  /////  (2) Get intEng_extra functions   /////
  auto f2a = get_host_func(intEng_extra_func, func2_names, func2_table_cpu);
  auto intEng_extra_ptrs = f2a.data();

  // transform the name of intEng_extra_func by appending "_ddT"
  auto intEng_extra_ddT_func = intEng_extra_func;
  for (auto &name : intEng_extra_ddT_func) {
    if (!name.empty()) name += "_ddT";
  }

  auto f2b = get_host_func(intEng_extra_ddT_func, func2_names, func2_table_cpu);
  auto intEng_extra_ddT_ptrs = f2b.data();

  /////  (3) Launch kernel calculation    /////

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_equilibrate_uv_cpu", [&] {
    int nspecies = at::native::ensure_nonempty_size(stoich, 0);
    int nreaction = at::native::ensure_nonempty_size(stoich, 1);

    auto stoich_ptr = stoich.data_ptr<scalar_t>();
    auto intEng_offset_ptr = intEng_offset.data_ptr<scalar_t>();
    auto cv_const_ptr = cv_const.data_ptr<scalar_t>();

    iter.for_each(
        [&](char **data, const int64_t *strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto gain = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
            auto diag = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
            auto conc = reinterpret_cast<scalar_t *>(data[2] + i * strides[2]);
            auto temp = reinterpret_cast<scalar_t *>(data[3] + i * strides[3]);
            auto intEng =
                reinterpret_cast<scalar_t *>(data[4] + i * strides[4]);
            auto reaction_set =
                reinterpret_cast<int *>(data[5] + i * strides[5]);
            auto nactive = reinterpret_cast<int *>(data[6] + i * strides[6]);

            int max_iter_i = max_iter;
            equilibrate_uv(gain, diag, temp, conc, *intEng, stoich_ptr,
                           nspecies, nreaction, ngas, intEng_offset_ptr,
                           cv_const_ptr, logsvp_ptrs, logsvp_ddT_ptrs,
                           intEng_extra_ptrs, intEng_extra_ddT_ptrs, logsvp_eps,
                           &max_iter_i, reaction_set, nactive);
          }
        },
        grain_size);
  });
}

}  // namespace kintera

namespace at::native {

DEFINE_DISPATCH(call_equilibrate_tp);
DEFINE_DISPATCH(call_equilibrate_uv);

REGISTER_ALL_CPU_DISPATCH(call_equilibrate_tp,
                          &kintera::call_equilibrate_tp_cpu);
REGISTER_ALL_CPU_DISPATCH(call_equilibrate_uv,
                          &kintera::call_equilibrate_uv_cpu);

}  // namespace at::native
