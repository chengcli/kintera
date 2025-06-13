#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using equilibrate_tp_fn = void (*)(at::TensorIterator &iter, int ngas,
                                   user_func1 const *logsvp_func,
                                   float logsvp_eps, int max_iter);

using equilibrate_uv_fn = void (*)(at::TensorIterator &iter,
                                   user_func1 const *logsvp_func,
                                   user_func1 const *logsvp_func_ddT,
                                   user_func2 const *intEng_extra,
                                   user_func2 const *intEng_extra_ddT,
                                   float logsvp_eps, int max_iter);

using integrate_z_fn = void (*)(at::TensorIterator &iter, float dz,
                                char const *method, float grav, float adTdz,
                                user_func1 const *logsvp_func, float logsvp_eps,
                                int max_iter);

DECLARE_DISPATCH(equilibrate_tp_fn, call_equilibrate_tp);
DECLARE_DISPATCH(equilibrate_uv_fn, call_equilibrate_uv);
DECLARE_DISPATCH(integrate_z_fn, call_integrate_z);

}  // namespace at::native
