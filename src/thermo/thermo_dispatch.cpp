// torch
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// kintera
#include "equilibrate_tp.h"
#include "equilibrate_uv.h"

namespace kintera {

void call_equilibrate_tp_cpu(at::TensorIterator &iter, int ngas,
                             user_func1 const *logsvp_func,
                             double logsvp_eps, int max_iter)
{
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "equilibrate_tp_cpu", [&] {
    int nspecies = at::native::ensure_nonempty_size(iter.input(2), 0);
    int nreaction = at::native::ensure_nonempty_size(iter.input(2), 1);

    iter.for_each([&](char **data, const int64_t *strides, int64_t n) {
      auto stoich = reinterpret_cast<scalar_t *>(data[3]);
      for (int i = 0; i < n; i++) {
        auto out = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
        auto temp = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
        auto pres = reinterpret_cast<scalar_t *>(data[2] + i * strides[2]);
        int max_iter_i = max_iter;
        equilibrate_tp(out, *temp, *pres, stoich, nspecies, nreaction, ngas,
                       logsvp_func, logsvp_eps, &max_iter_i);
      }
    });
  });
}

void call_equilibrate_uv_cpu(at::TensorIterator &iter, double h0,
                             user_func1 const *logsvp_func,
                             user_func1 const *logsvp_func_ddT,
                             user_func1 const *enthalpy_extra,
                             user_func1 const *enthalpy_extra_ddT,
                             double logsvp_eps, int max_iter)
{
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "equilibrate_uv_cpu", [&] {
    int nspecies = at::native::ensure_nonempty_size(iter.input(2), 0);
    int nreaction = at::native::ensure_nonempty_size(iter.input(2), 1);

    iter.for_each([&](char **data, const int64_t *strides, int64_t n) {
      auto stoich = reinterpret_cast<scalar_t *>(data[2]);
      auto enthalpy_offset = reinterpret_cast<scalar_t *>(data[3]);
      auto cp_const = reinterpret_cast<scalar_t *>(data[4]);

      for (int i = 0; i < n; i++) {
        auto out = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
        auto conc = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
        int max_iter_i = max_iter;
        equilibrate_uv(out, conc, h0, stoich, nspecies, nreaction, enthalpy_offset,
                       cp_const, logsvp_func, logsvp_func_ddT, 
                       enthalpy_extra, enthalpy_extra_ddT,
                       logsvp_eps, &max_iter_i);
      }
    });
  });
}

} // namespace kintera
