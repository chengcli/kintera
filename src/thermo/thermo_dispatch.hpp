#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

using user_func1 = double (*)(double);
using user_func2 = double (*)(double, double);

namespace at::native {

using equilibrate_tp_fn = void (*)(at::TensorIterator &iter, int ngas,
                                   at::Tensor const &stoich,
                                   std::vector<std::string> const &logsvp_func,
                                   double logsvp_eps, int max_iter);

using equilibrate_uv_fn =
    void (*)(at::TensorIterator &iter, int ngas, at::Tensor const &stoich,
             at::Tensor const &intEng_offset, at::Tensor const &cv_const,
             std::vector<std::string> const &logsvp_func,
             std::vector<std::string> const &intEng_extra_func,
             double logsvp_eps, int max_iter);

DECLARE_DISPATCH(equilibrate_tp_fn, call_equilibrate_tp);
DECLARE_DISPATCH(equilibrate_uv_fn, call_equilibrate_uv);

}  // namespace at::native
