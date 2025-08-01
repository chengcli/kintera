// thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// kintera
#include "func3.hpp"

extern __device__ __constant__ user_func3* func3_table_device_ptr;

thrust::device_vector<user_func3> get_device_func3(
    std::vector<std::string> const& names)
{
  // Get full device function table
  user_func3* d_full_table = nullptr;
  cudaMemcpyFromSymbol(&d_full_table, func3_table_device_ptr, sizeof(user_func3*));

  // Create thrust host vector for selected function pointers
  thrust::host_vector<user_func3> h_ptrs(names.size());

  for (size_t i = 0; i < names.size(); ++i) {
    int idx = Func3Registrar::get_id(names[i]);

    if (idx == -1) {  // null-op
      h_ptrs[i] = nullptr;
      continue;
    }

    // Copy individual device function pointer to host
    cudaMemcpy(&h_ptrs[i], d_full_table + idx, sizeof(user_func3), cudaMemcpyDeviceToHost);
  }

  // Copy to thrust device vector
  return thrust::device_vector<user_func3>(h_ptrs);
}
