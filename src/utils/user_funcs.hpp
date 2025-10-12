#pragma once

// C/C++
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

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
    } else if (name == "null") {
      funcs.push_back(nullptr);
    } else {
      throw std::runtime_error("Function " + name + " not registered.");
    }
  }
  return funcs;
}

#ifdef __CUDACC__
#include <thrust/device_vector.h>

template <typename T>
thrust::device_vector<T> get_device_func(
    std::vector<std::string> const& names,
    std::vector<std::string> const& func_names, T* func_table) {
  // (1) Get full device function table
  T* d_full_table = nullptr;
  cudaMemcpyFromSymbol(&d_full_table, func1_table_device_ptr, sizeof(T*));

  // (2) Build a host‐side index list
  std::vector<int> h_idx;

  for (const auto& name : names) {
    auto it = std::find(func_names.begin(), func_names.end(), name);
    if (it != func_names.end()) {
      int id = static_cast<int>(std::distance(func_names.begin(), it));
      h_idx.push_back(id + 1);
    } else if (name == "null") {
      h_idx.push_back(0);
    } else {
      throw std::runtime_error("Function " + name + " not registered.");
    }
  }

  // (3) Copy indices to device
  thrust::device_vector<int> d_idx = h_idx;

  // (4) Wrap the raw table pointer
  thrust::device_ptr<T> full_ptr(d_full_table);

  // (5) Allocate result and do one gather
  thrust::device_vector<T> result(names.size());
  thrust::gather(d_idx.begin(),  // where to read your indices
                 d_idx.end(),
                 full_ptr,       // base array to gather from
                 result.begin()  // write results here
  );

  return result;
}

#endif

}  // namespace kintera
