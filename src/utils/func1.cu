// thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// kintera
#include "func1.hpp"

extern __device__ __constant__ user_func1* func1_table_device_ptr;

thrust::device_vector<user_func1> get_device_func1(
    std::vector<std::string> const& names)
{
  // Get full device function table
  user_func1* d_full_table = nullptr;
  cudaMemcpyFromSymbol(&d_full_table, func1_table_device_ptr, sizeof(user_func1*));

  // Create thrust host vector for selected function pointers
  thrust::host_vector<user_func1> h_ptrs(names.size());

  for (size_t i = 0; i < names.size(); ++i) {
    int idx = Func1Registrar::get_id(names[i]);

    if (idx == -1) {  // null-op
      h_ptrs[i] = nullptr;
      continue;
    }

    // Copy individual device function pointer to host
    cudaMemcpy(&h_ptrs[i], d_full_table + idx, sizeof(user_func1), cudaMemcpyDeviceToHost);
  }

  // Copy to thrust device vector
  return thrust::device_vector<user_func1>(h_ptrs);
}
