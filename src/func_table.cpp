// kintera
#include <kintera/thermo/h2_rotation.h>
#include <kintera/vapors/vapor_functions.h>

#include <kintera/utils/user_funcs.hpp>

#include "func_list.h"

namespace kintera {

/////  func1 registry  /////

user_func1 func1_table_cpu[] = {KINTERA_FUNC1_LIST(KINTERA_FUNC_PTR)};

std::vector<std::string> func1_names = {KINTERA_FUNC1_LIST(KINTERA_FUNC_NAME)};

/////  func2 registry  /////

user_func2 func2_table_cpu[] = {KINTERA_FUNC2_LIST(KINTERA_FUNC_PTR)};

std::vector<std::string> func2_names = {KINTERA_FUNC2_LIST(KINTERA_FUNC_NAME)};

/////  func3 registry  /////

user_func3 func3_table_cpu[] = {KINTERA_FUNC3_LIST(KINTERA_FUNC_PTR)};

std::vector<std::string> func3_names = {KINTERA_FUNC3_LIST(KINTERA_FUNC_NAME)};

}  // namespace kintera
