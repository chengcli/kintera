// kintera
#include <kintera/vapors/vapor_functions.h>

#include <kintera/utils/user_funcs.hpp>

namespace kintera {

/////  func1 registry  /////

user_func1 func1_table_cpu[] = {
    h2o_ideal,     h2o_ideal_ddT,     nh3_ideal,   nh3_ideal_ddT,
    nh3_h2s_lewis, nh3_h2s_lewis_ddT, h2s_ideal,   h2s_ideal_ddT,
    h2s_antoine,   h2s_antoine_ddT,   ch4_ideal,   ch4_ideal_ddT,
    so2_antoine,   so2_antoine_ddT,   co2_antoine, co2_antoine_ddT};

std::vector<std::string> func1_names = {
    "h2o_ideal",     "h2o_ideal_ddT",     "nh3_ideal",   "nh3_ideal_ddT",
    "nh3_h2s_lewis", "nh3_h2s_lewis_ddT", "h2s_ideal",   "h2s_ideal_ddT",
    "h2s_antoine",   "h2s_antoine_ddT",   "ch4_ideal",   "ch4_ideal_ddT",
    "so2_antoine",   "so2_antoine_ddT",   "co2_antoine", "co2_antoine_ddT"};

/////  func2 registry  /////

user_func2 func2_table_cpu[] = {};

std::vector<std::string> func2_names = {};

/////  func3 registry  /////

user_func3 func3_table_cpu[] = {};

std::vector<std::string> func3_names = {};

}  // namespace kintera
