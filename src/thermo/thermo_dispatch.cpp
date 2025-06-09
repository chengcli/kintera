// torch
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>
#include <torch/torch.h>

// kintera
#include "equilibrate_tp.h"
#include "equilibrate_uv.h"
#include "integrate_z.h"
#include "thermo_dispatch.hpp"

namespace kintera {

void call_equilibrate_tp_cpu(at::TensorIterator &iter, int ngas,
                             user_func1 const *logsvp_func, float logsvp_eps,
                             int max_iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_equilibrate_tp_cpu", [&] {
    int nspecies = at::native::ensure_nonempty_size(iter.input(2), 0);
    int nreaction = at::native::ensure_nonempty_size(iter.input(2), 1);
    auto stoich = iter.input(2).data_ptr<scalar_t>();

    iter.for_each([&](char **data, const int64_t *strides, int64_t n) {
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

void call_equilibrate_uv_cpu(at::TensorIterator &iter,
                             user_func1 const *logsvp_func,
                             user_func1 const *logsvp_func_ddT,
                             user_func2 const *intEng_extra,
                             user_func2 const *intEng_extra_ddT,
                             float logsvp_eps, int max_iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_equilibrate_uv_cpu", [&] {
    int nspecies = at::native::ensure_nonempty_size(iter.input(1), 0);
    int nreaction = at::native::ensure_nonempty_size(iter.input(1), 1);
    auto stoich = iter.input(1).data_ptr<scalar_t>();
    auto intEng_offset = iter.input(2).data_ptr<scalar_t>();
    auto cv_const = iter.input(3).data_ptr<scalar_t>();

    iter.for_each([&](char **data, const int64_t *strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto conc = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
        auto temp = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
        auto intEng = reinterpret_cast<scalar_t *>(data[2] + i * strides[2]);
        int max_iter_i = max_iter;
        equilibrate_uv(temp, conc, *intEng, stoich, nspecies, nreaction,
                       intEng_offset, cv_const, logsvp_func, logsvp_func_ddT,
                       intEng_extra, intEng_extra_ddT, logsvp_eps, &max_iter_i);
      }
    });
  });
}

void call_integrate_z_cpu(at::TensorIterator &iter, float dz,
                          char const *method, float grav, float adTdz,
                          user_func1 const *logsvp_func, float logsvp_eps,
                          int max_iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_integrate_z_cpu", [&] {
    int nspecies = at::native::ensure_nonempty_size(iter.input(2), 0);
    int ngas = at::native::ensure_nonempty_size(iter.input(2), 1);
    auto stoich = iter.input(2).data_ptr<scalar_t>();
    auto mu = iter.input(3).data_ptr<scalar_t>();

    iter.for_each([&](char **data, const int64_t *strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto xfrac = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
        auto temp = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
        auto pres = reinterpret_cast<scalar_t *>(data[2] + i * strides[2]);
        auto enthalpy = reinterpret_cast<scalar_t *>(data[3] + i * strides[3]);
        auto cp = reinterpret_cast<scalar_t *>(data[4] + i * strides[4]);
        int max_iter_i = max_iter;
        integrate_z(xfrac, temp, pres, mu, (scalar_t)dz, method, (scalar_t)grav,
                    (scalar_t)adTdz, stoich, nspecies, ngas, enthalpy, cp,
                    logsvp_func, logsvp_eps, &max_iter_i);
      }
    });
  });
}

void call_with_TC_cpu(at::TensorIterator &iter, user_func2 const *func) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_with_TC_cpu", [&] {
    int nspecies = at::native::ensure_nonempty_size(iter.input(0), -1);
    iter.for_each([&](char **data, const int64_t *strides, int64_t n) {
      for (int i = 0; i < n; i++) {
        auto out = reinterpret_cast<scalar_t *>(data[0] + i * strides[0]);
        auto temp = reinterpret_cast<scalar_t *>(data[1] + i * strides[1]);
        auto conc = reinterpret_cast<scalar_t *>(data[2] + i * strides[2]);
        for (int j = 0; j < nspecies; ++j) {
          if (func[j] == nullptr) {
            out[j] = 0;
          } else {
            out[j] = func[j](*temp, conc[j]);
          }
        }
      }
    });
  });
}

}  // namespace kintera

namespace at::native {

DEFINE_DISPATCH(call_equilibrate_tp);
DEFINE_DISPATCH(call_equilibrate_uv);
DEFINE_DISPATCH(call_integrate_z);
DEFINE_DISPATCH(call_with_TC);

REGISTER_ALL_CPU_DISPATCH(call_equilibrate_tp,
                          &kintera::call_equilibrate_tp_cpu);
REGISTER_ALL_CPU_DISPATCH(call_equilibrate_uv,
                          &kintera::call_equilibrate_uv_cpu);
REGISTER_ALL_CPU_DISPATCH(call_integrate_z, &kintera::call_integrate_z_cpu);
REGISTER_ALL_CPU_DISPATCH(call_with_TC, &kintera::call_with_TC_cpu);

}  // namespace at::native
