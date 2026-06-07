#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using evolve_implicit_fn = void (*)(at::TensorIterator& iter,
                                    at::Tensor const& stoich, double dt);

DECLARE_DISPATCH(evolve_implicit_fn, call_evolve_implicit);

}  // namespace at::native
