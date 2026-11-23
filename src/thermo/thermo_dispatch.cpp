// C/C++
#include <map>
#include <mutex>

// torch
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Loops.h>

// kintera
#include "equilibrate_tp.h"
#include "equilibrate_uv.h"
#include "log_svp.hpp"
#include "thermo.hpp"
#include "thermo_dispatch.hpp"

namespace kintera {

void call_equilibrate_tp_cpu(at::TensorIterator& iter, int ngas,
                             at::Tensor const& stoich, ThermoTables const& t,
                             double logsvp_eps, int max_iter) {
  int grain_size = iter.numel() / at::get_num_threads();
  auto logsvp_ptrs = static_cast<user_func1*>(t.logsvp.data_ptr());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_equilibrate_tp_cpu", [&] {
    int nspecies = at::native::ensure_nonempty_size(stoich, 0);
    int nreaction = at::native::ensure_nonempty_size(stoich, 1);

    auto stoich_ptr = stoich.data_ptr<scalar_t>();
    auto svp_kind_ptr = t.svp_kind.data_ptr<int>();
    auto svp_params_ptr = t.svp_params.data_ptr<double>();

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto gain = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto diag = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto xfrac = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            auto temp = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
            auto pres = reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);
            auto reaction_set =
                reinterpret_cast<int*>(data[5] + i * strides[5]);
            auto nactive = reinterpret_cast<int*>(data[6] + i * strides[6]);
            int max_iter_i = max_iter;
            equilibrate_tp(gain, diag, xfrac, *temp, *pres, stoich_ptr,
                           nspecies, nreaction, ngas, logsvp_ptrs, svp_kind_ptr,
                           svp_params_ptr, logsvp_eps, &max_iter_i,
                           reaction_set, nactive);
          }
        },
        grain_size);
  });
}

void call_equilibrate_uv_cpu(at::TensorIterator& iter, int ngas,
                             at::Tensor const& stoich,
                             at::Tensor const& intEng_offset,
                             at::Tensor const& cv_const, ThermoTables const& t,
                             double logsvp_eps, int max_iter, int uv_solver) {
  int grain_size = iter.numel() / at::get_num_threads();

  // svp and intEng_extra functions: the table's, resolved when it was built
  auto logsvp_ptrs = static_cast<user_func1*>(t.logsvp.data_ptr());
  auto logsvp_ddT_ptrs = static_cast<user_func1*>(t.logsvp_ddT.data_ptr());
  auto intEng_extra_ptrs = static_cast<user_func2*>(t.intEng_extra.data_ptr());
  auto intEng_extra_ddT_ptrs =
      static_cast<user_func2*>(t.intEng_extra_ddT.data_ptr());

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_equilibrate_uv_cpu", [&] {
    int nspecies = at::native::ensure_nonempty_size(stoich, 0);
    int nreaction = at::native::ensure_nonempty_size(stoich, 1);

    auto stoich_ptr = stoich.data_ptr<scalar_t>();
    auto intEng_offset_ptr = intEng_offset.data_ptr<scalar_t>();
    auto cv_const_ptr = cv_const.data_ptr<scalar_t>();
    auto svp_kind_ptr = t.svp_kind.data_ptr<int>();
    auto svp_params_ptr = t.svp_params.data_ptr<double>();

    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto gain = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto diag = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            auto conc = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            auto temp = reinterpret_cast<scalar_t*>(data[3] + i * strides[3]);
            auto intEng = reinterpret_cast<scalar_t*>(data[4] + i * strides[4]);
            auto reaction_set =
                reinterpret_cast<int*>(data[5] + i * strides[5]);
            auto nactive = reinterpret_cast<int*>(data[6] + i * strides[6]);

            int max_iter_i = max_iter;
            equilibrate_uv(gain, diag, temp, conc, *intEng, stoich_ptr,
                           nspecies, nreaction, ngas, intEng_offset_ptr,
                           cv_const_ptr, logsvp_ptrs, logsvp_ddT_ptrs,
                           svp_kind_ptr, svp_params_ptr, intEng_extra_ptrs,
                           intEng_extra_ddT_ptrs, logsvp_eps, &max_iter_i,
                           reaction_set, nactive, uv_solver);
          }
        },
        grain_size);
  });
}

void call_logsvp_inline_cpu(at::TensorIterator& iter,
                            at::Tensor const& svp_kind,
                            at::Tensor const& svp_params, bool deriv) {
  int grain_size = iter.numel() / at::get_num_threads();

  auto svp_kind_ptr = svp_kind.data_ptr<int>();
  auto svp_params_ptr = svp_params.data_ptr<double>();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "call_logsvp_inline_cpu", [&] {
    int nout = at::native::ensure_nonempty_size(iter.output(), -1);
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          for (int i = 0; i < n; i++) {
            auto out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            // temp
            auto arg1 = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            for (int j = 0; j < nout; ++j) {
              if (svp_kind_ptr[j] == 0) continue;
              double const* p = svp_params_ptr + j * KSVP_NPARAM;
              out[j] = deriv
                           ? eval_logsvp_ddT(svp_kind_ptr[j], p, nullptr, *arg1)
                           : eval_logsvp(svp_kind_ptr[j], p, nullptr, *arg1);
            }
          }
        },
        grain_size);
  });
}

uint64_t options_digest(SpeciesThermo const& op) {  // of the content
  uint64_t h = 1469598103934665603ull;
  auto add = [&](auto const&... v) { (fnv1a(h, v), ...); };
  add(op->vapor_ids(), op->cloud_ids(), op->names(), op->mu(), op->cref_R(),
      op->uref_R(), op->sref_R(), op->intEng_R_extra(), op->cp_R_extra(),
      op->entropy_R_extra(), op->czh(), op->czh_ddC(), op->nasa9_low(),
      op->nasa9_high(), op->nasa9_Tmid(), op->use_nasa9_cp(), op->use_h2_cp(),
      op->h2_cp_mode());
  auto th = std::dynamic_pointer_cast<ThermoOptionsImpl>(op);
  if (th && th->nucleation())
    add(th->nucleation()->logsvp(), th->nucleation()->svp_params());
  return h;
}

// the store: one table per (options digest, device, dtype), kept, so it is
// built once; a missing one is built by species_tables (thermo.hpp) and
// make_svp_spec. Never freed: no device tensor is released at process exit
ThermoTablesPtr tables_for(SpeciesThermo const& op, at::Tensor const& like) {
  static std::mutex mutex;
  static auto& store =
      *new std::map<std::tuple<uint64_t, int, int, int>, ThermoTablesPtr>;
  static int64_t epoch = 0;
  auto h = options_digest(op);
  auto dev = like.device();
  std::lock_guard<std::mutex> lock(mutex);
  auto& slot =
      store[{h, (int)dev.type(), dev.index(), (int)like.scalar_type()}];
  if (slot) return NoTableBuild::record(slot), slot;
  TORCH_CHECK(!NoTableBuild::top() && !capture_active(dev),
              "[kintera] no thermo table for ", dev, " ", like.scalar_type(),
              " inside capture or NoTableBuild: prepare() first");
  c10::InferenceMode off(false);  // normal tensors, usable in both modes
  auto t = species_tables(op, like);
  auto th = std::dynamic_pointer_cast<ThermoOptionsImpl>(op);
  LogSVPFunc::make_svp_spec(th ? th->nucleation() : nullptr, *t,
                            thermo_callback_device(dev));
  t->digest = h, t->epoch = ++epoch;  // the epoch never changes after
  slot = t;
  NoTableBuild::record(t);
  return t;
}

}  // namespace kintera

namespace at::native {

DEFINE_DISPATCH(call_equilibrate_tp);
DEFINE_DISPATCH(call_equilibrate_uv);
DEFINE_DISPATCH(call_logsvp_inline);

REGISTER_ALL_CPU_DISPATCH(call_equilibrate_tp,
                          &kintera::call_equilibrate_tp_cpu);
REGISTER_ALL_CPU_DISPATCH(call_equilibrate_uv,
                          &kintera::call_equilibrate_uv_cpu);
REGISTER_ALL_CPU_DISPATCH(call_logsvp_inline, &kintera::call_logsvp_inline_cpu);

}  // namespace at::native
