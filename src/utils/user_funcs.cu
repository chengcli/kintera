// torch
#include <ATen/ATen.h>

// kintera
#include "user_funcs.hpp"

namespace kintera {

extern __device__ __constant__ user_func1 *func1_table_device_ptr;
extern __device__ __constant__ user_func2 *func2_table_device_ptr;
extern __device__ __constant__ user_func3 *func3_table_device_ptr;

extern std::vector<std::string> func1_names;
extern std::vector<std::string> func2_names;
extern std::vector<std::string> func3_names;

// The device pointers of `names` as a FuncTable: one symbol read and one
// gather per table, not per call. Entry 0 of a device table is null.
template <typename T>
FuncTable gather_device_funcs(std::vector<std::string> const& names,
                              std::vector<std::string> const& func_names,
                              T* const& symbol, at::Device device) {
  std::vector<int64_t> h_idx;
  for (const auto& name : names) {
    auto it = std::find(func_names.begin(), func_names.end(), name);
    if (it != func_names.end()) {
      h_idx.push_back(std::distance(func_names.begin(), it) + 1);
    } else if (name.empty() || name == "null") {
      h_idx.push_back(0);
    } else {
      throw std::runtime_error("Function " + name + " not registered.");
    }
  }

  T* d_full_table = nullptr;
  cudaMemcpyFromSymbol(&d_full_table, symbol, sizeof(T*));
  auto opts = at::TensorOptions().dtype(at::kLong).device(device);
  auto full = at::from_blob(
      d_full_table, {static_cast<int64_t>(func_names.size()) + 1}, opts);
  return full.index_select(0, at::tensor(h_idx, at::kLong).to(device));
}

// called by the CUDA kernel of the fill_func_table stub (utils_dispatch.cu)
FuncTable get_device_funcs(std::vector<std::string> const& names, int arity,
                           at::Device device) {
  if (arity == 1)
    return gather_device_funcs(names, func1_names, func1_table_device_ptr,
                               device);
  if (arity == 2)
    return gather_device_funcs(names, func2_names, func2_table_device_ptr,
                               device);
  return gather_device_funcs(names, func3_names, func3_table_device_ptr,
                             device);
}

}  // namespace kintera
