// C/C++
#include <thrust/device_vector.h>
#include <thrust/gather.h>

// kintera
#include "user_funcs.hpp"

namespace kintera {

extern __device__ __constant__ user_func1 *func1_table_device_ptr;
extern __device__ __constant__ user_func2 *func2_table_device_ptr;
extern __device__ __constant__ user_func3 *func3_table_device_ptr;

thrust::device_vector<user_func1> get_device_func1(
    std::vector<std::string> const& names,
    std::vector<std::string> const& func_names) {
  // (1) Get full device function table
  user_func1* d_full_table = nullptr;
  cudaMemcpyFromSymbol(&d_full_table, func1_table_device_ptr,
                       sizeof(user_func1*));

  // (2) Build a host‐side index list
  std::vector<int> h_idx;

  for (const auto& name : names) {
    auto it = std::find(func_names.begin(), func_names.end(), name);
    if (it != func_names.end()) {
      int id = static_cast<int>(std::distance(func_names.begin(), it));
      h_idx.push_back(id + 1);
    } else if (name.empty() || name == "null") {
      h_idx.push_back(0);
    } else {
      throw std::runtime_error("Function " + name + " not registered.");
    }
  }

  // (3) Copy indices to device
  thrust::device_vector<int> d_idx = h_idx;

  // (4) Wrap the raw table pointer
  thrust::device_ptr<user_func1> full_ptr(d_full_table);

  // (5) Allocate result and do one gather
  thrust::device_vector<user_func1> result(names.size());
  thrust::gather(d_idx.begin(),  // where to read your indices
                 d_idx.end(),
                 full_ptr,       // base array to gather from
                 result.begin()  // write results here
  );

  return result;
}

thrust::device_vector<user_func2> get_device_func2(
    std::vector<std::string> const& names,
    std::vector<std::string> const& func_names) {
  // (1) Get full device function table
  user_func2* d_full_table = nullptr;
  cudaMemcpyFromSymbol(&d_full_table, func2_table_device_ptr,
                       sizeof(user_func2*));

  // (2) Build a host‐side index list
  std::vector<int> h_idx;

  for (const auto& name : names) {
    auto it = std::find(func_names.begin(), func_names.end(), name);
    if (it != func_names.end()) {
      int id = static_cast<int>(std::distance(func_names.begin(), it));
      h_idx.push_back(id + 1);
    } else if (name.empty() || name == "null") {
      h_idx.push_back(0);
    } else {
      throw std::runtime_error("Function " + name + " not registered.");
    }
  }

  // (3) Copy indices to device
  thrust::device_vector<int> d_idx = h_idx;

  // (4) Wrap the raw table pointer
  thrust::device_ptr<user_func2> full_ptr(d_full_table);

  // (5) Allocate result and do one gather
  thrust::device_vector<user_func2> result(names.size());
  thrust::gather(d_idx.begin(),  // where to read your indices
                 d_idx.end(),
                 full_ptr,       // base array to gather from
                 result.begin()  // write results here
  );

  return result;
}

thrust::device_vector<user_func3> get_device_func3(
    std::vector<std::string> const& names,
    std::vector<std::string> const& func_names) {
  // (1) Get full device function table
  user_func3* d_full_table = nullptr;
  cudaMemcpyFromSymbol(&d_full_table, func3_table_device_ptr,
                       sizeof(user_func3*));

  // (2) Build a host‐side index list
  std::vector<int> h_idx;

  for (const auto& name : names) {
    auto it = std::find(func_names.begin(), func_names.end(), name);
    if (it != func_names.end()) {
      int id = static_cast<int>(std::distance(func_names.begin(), it));
      h_idx.push_back(id + 1);
    } else if (name.empty() || name == "null") {
      h_idx.push_back(0);
    } else {
      throw std::runtime_error("Function " + name + " not registered.");
    }
  }

  // (3) Copy indices to device
  thrust::device_vector<int> d_idx = h_idx;

  // (4) Wrap the raw table pointer
  thrust::device_ptr<user_func3> full_ptr(d_full_table);

  // (5) Allocate result and do one gather
  thrust::device_vector<user_func3> result(names.size());
  thrust::gather(d_idx.begin(),  // where to read your indices
                 d_idx.end(),
                 full_ptr,       // base array to gather from
                 result.begin()  // write results here
  );

  return result;
}

}  // namespace kintera
