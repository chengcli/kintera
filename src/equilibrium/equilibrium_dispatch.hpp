#pragma once

#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using equilibrium_fn = void (*)(at::TensorIterator& iter,
                                at::Tensor const& stoich,
                                at::Tensor const& phase_ids, int nphase,
                                int gas_phase, double standard_pressure,
                                double ftol, double mole_floor, int max_iter);

DECLARE_DISPATCH(equilibrium_fn, call_equilibrium);

}  // namespace at::native
