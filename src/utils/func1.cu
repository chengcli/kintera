// thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// kintera
#include "func1.hpp"

extern __device__ __constant__ user_func1* func1_table_device_ptr;

thrust::device_vector<user_func1> get_device_func1(
    std::vector<std::string> const& names)
{
  // (1) Get full device function table
  user_func1* d_full_table = nullptr;
  cudaMemcpyFromSymbol(&d_full_table, func1_table_device_ptr, sizeof(user_func1*));

  // (2) Build a host‐side index list
  std::vector<int> h_idx(names.size());

  for (size_t i = 0; i < names.size(); ++i) {
    int id = Func1Registrar::get_id(names[i]);
    h_idx[i] = (id < 0 ? 0 : id + 1);
  }

  // (3) Copy indices to device
  thrust::device_vector<int> d_idx = h_idx;

  // (4) Wrap the raw table pointer
  thrust::device_ptr<user_func1> full_ptr(d_full_table);

  // (5) Allocate your result and do one gather
  thrust::device_vector<user_func1> result(names.size());
  thrust::gather(
    d_idx.begin(),           // where to read your indices
    d_idx.end(),
    full_ptr,                // base array to gather from
    result.begin()           // write results here
  );

  return result;
}
