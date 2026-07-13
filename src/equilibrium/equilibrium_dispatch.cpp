#include "equilibrium_dispatch.hpp"

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/ReduceOpsUtils.h>

#include "phase_equilibrate_tp.h"

namespace kintera {

void call_equilibrium_cpu(at::TensorIterator& iter, at::Tensor const& stoich,
                          at::Tensor const& phase_ids, int nphase,
                          int gas_phase, double standard_pressure, double ftol,
                          double mole_floor, int max_iter) {
  int grain_size = std::max<int64_t>(1, iter.numel() / at::get_num_threads());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_equilibrium_cpu", [&] {
    int nspecies = stoich.size(0);
    int nreaction = stoich.size(1);
    auto stoich_ptr = stoich.data_ptr<scalar_t>();
    auto phase_ptr = phase_ids.data_ptr<int>();
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int64_t i = 0; i < n; ++i) {
            auto gain = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto diag = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto out = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            auto temp = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
            auto pres = reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);
            auto moles = reinterpret_cast<scalar_t*>(data[5] + i * strides[5]);
            auto log_k = reinterpret_cast<scalar_t*>(data[6] + i * strides[6]);
            phase_equilibrate_tp(gain, diag, out, *temp, *pres, moles, log_k,
                                 stoich_ptr, phase_ptr, nspecies, nreaction,
                                 nphase, gas_phase,
                                 static_cast<scalar_t>(standard_pressure),
                                 static_cast<scalar_t>(ftol),
                                 static_cast<scalar_t>(mole_floor), max_iter);
          }
        },
        grain_size);
  });
}

}  // namespace kintera

namespace at::native {

DEFINE_DISPATCH(call_equilibrium);
REGISTER_ALL_CPU_DISPATCH(call_equilibrium, &kintera::call_equilibrium_cpu);

}  // namespace at::native
