#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

// kintera
#include <kintera/utils/user_funcs.hpp>

namespace at::native {

using equilibrate_tp_fn = void (*)(at::TensorIterator& iter, int ngas,
                                   at::Tensor const& stoich,
                                   kintera::ThermoTables const& tables,
                                   double logsvp_eps, int max_iter);

using equilibrate_uv_fn = void (*)(at::TensorIterator& iter, int ngas,
                                   at::Tensor const& stoich,
                                   at::Tensor const& intEng_offset,
                                   at::Tensor const& cv_const,
                                   kintera::ThermoTables const& tables,
                                   double logsvp_eps, int max_iter,
                                   int uv_solver);

using logsvp_inline_fn = void (*)(at::TensorIterator& iter,
                                  at::Tensor const& svp_kind,
                                  at::Tensor const& svp_params, bool deriv);

DECLARE_DISPATCH(equilibrate_tp_fn, call_equilibrate_tp);
DECLARE_DISPATCH(equilibrate_uv_fn, call_equilibrate_uv);
DECLARE_DISPATCH(logsvp_inline_fn, call_logsvp_inline);

}  // namespace at::native
