// kintera
#include <kintera/thermo/h2_rotation.h>
#include <kintera/vapors/vapor_functions.h>

#include <kintera/utils/user_funcs.hpp>

#include "func_list.h"

namespace kintera {

/////  func1 registry  /////

__device__ user_func1 func1_table_cuda[] = {
    nullptr, KINTERA_FUNC1_LIST(KINTERA_FUNC_PTR)};

__device__ __constant__ user_func1* func1_table_device_ptr = func1_table_cuda;

/////  func2 registry  /////

__device__ user_func2 func2_table_cuda[] = {
    nullptr, KINTERA_FUNC2_LIST(KINTERA_FUNC_PTR)};

__device__ __constant__ user_func2* func2_table_device_ptr = func2_table_cuda;

/////  func3 registry  /////

__device__ user_func3 func3_table_cuda[] = {
    nullptr KINTERA_FUNC3_LIST(KINTERA_FUNC_PTR)};

__device__ __constant__ user_func3* func3_table_device_ptr = func3_table_cuda;

}  // namespace kintera
