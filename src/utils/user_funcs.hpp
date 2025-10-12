#pragma once

// C/C++
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __CUDACC__
// thrust
#include <thrust/device_vector.h>

#endif

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

#ifdef __CUDACC__

thrust::device_vector<user_func1> get_device_func1(
    std::vector<std::string> const& names,
    std::vector<std::string> const& func_names);

thrust::device_vector<user_func2> get_device_func2(
    std::vector<std::string> const& names,
    std::vector<std::string> const& func_names);

thrust::device_vector<user_func3> get_device_func3(
    std::vector<std::string> const& names,
    std::vector<std::string> const& func_names);

#endif

}  // namespace kintera
