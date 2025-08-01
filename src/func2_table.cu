
// base
#include <configure.h>

using user_func2 = double (*)(double, double);

__device__ __constant__ user_func2* func2_table_device_ptr = nullptr;
