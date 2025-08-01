
// base
#include <configure.h>

using user_func3 = double (*)(double, double, double);

__device__ __constant__ user_func3* func3_table_device_ptr = nullptr;
