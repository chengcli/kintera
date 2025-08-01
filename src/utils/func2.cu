// thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// kintera
#include "func2.hpp"

extern __device__ __constant__ user_func2* func2_table_device_ptr;

thrust::device_vector<user_func2> get_device_func2(
    std::vector<std::string> const& names)
{
  // Get full device function table
  user_func2* d_full_table = nullptr;
  cudaMemcpyFromSymbol(&d_full_table, func2_table_device_ptr, sizeof(user_func2*));

  // Create thrust host vector for selected function pointers
  thrust::host_vector<user_func2> h_ptrs(names.size());

  for (size_t i = 0; i < names.size(); ++i) {
    int idx = Func2Registrar::get_id(names[i]);

    if (idx == -1) {  // null-op
      h_ptrs[i] = nullptr;
      continue;
    }

    // Copy individual device function pointer to host
    cudaMemcpy(&h_ptrs[i], d_full_table + idx, sizeof(user_func2), cudaMemcpyDeviceToHost);
  }

  // Copy to thrust device vector
  return thrust::device_vector<user_func2>(h_ptrs);
}
