#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using fn_iter = void (*)(at::TensorIterator& iter,
                         at::Tensor const& funcs);  // a kintera::FuncTable

DECLARE_DISPATCH(fn_iter, call_func1);
DECLARE_DISPATCH(fn_iter, call_func2);
DECLARE_DISPATCH(fn_iter, call_func3);

}  // namespace at::native
