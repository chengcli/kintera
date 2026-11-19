#pragma once

// C/C++
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// torch
#include <ATen/core/Tensor.h>

namespace kintera {

using user_func1 = double (*)(double);
using user_func2 = double (*)(double, double);
using user_func3 = double (*)(double, double, double);

template <typename T>
std::vector<T> get_host_func(std::vector<std::string> const& names,
                             std::vector<std::string> const& func_names,
                             T* func_table) {
  std::vector<T> funcs;
  for (const auto& name : names) {
    auto it = std::find(func_names.begin(), func_names.end(), name);
    if (it != func_names.end()) {
      int id = static_cast<int>(std::distance(func_names.begin(), it));
      funcs.push_back(func_table[id]);
    } else if (name.empty() || name == "null") {
      funcs.push_back(nullptr);
    } else {
      throw std::runtime_error("Function " + name + " not registered.");
    }
  }
  return funcs;
}

//! int64 tensor of host or device function pointers, one per column; 0: none
using FuncTable = at::Tensor;
//! one per (options digest, device, compute dtype), immutable once built
struct ThermoTables {
  uint64_t digest = 0;      // of the options content
  int64_t epoch = 0;        // set when built; a rebuild is a new table
  bool svp_inline = false;  // some svp column is 'ideal' or 'antoine'
  at::Tensor cref_R, uref_R, sref_R, cv0, u0, inv_mu, h2_mask, nasa9_low,
      nasa9_high, nasa9_Tmid, svp_kind, svp_params;  // int32 kind, FP64 params
  FuncTable logsvp, logsvp_ddT, intEng_extra, intEng_extra_ddT, cp_R_extra, czh,
      czh_ddC, entropy_extra, h2_entropy;  // intEng_extra has H2's
};
using ThermoTablesPtr = std::shared_ptr<ThermoTables const>;
struct SpeciesThermoImpl;
uint64_t options_digest(std::shared_ptr<SpeciesThermoImpl> const& op);
//! op's table for like's device and dtype; built outside capture/NoTableBuild
ThermoTablesPtr tables_for(std::shared_ptr<SpeciesThermoImpl> const& op,
                           at::Tensor const& like);

}  // namespace kintera
