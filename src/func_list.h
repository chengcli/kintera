#pragma once

// The registry of user functions, spelled once: func_table.cpp (host pointer
// tables and the name vectors) and func_table.cu (device tables, with a
// leading nullptr) expand these lists in the same order.

#define KINTERA_FUNC1_LIST(X) \
  X(h2o_ideal)                \
  X(h2o_ideal_ddT)            \
  X(h2o_bryan)                \
  X(h2o_bryan_ddT)            \
  X(nh3_ideal)                \
  X(nh3_ideal_ddT)            \
  X(nh3_h2s_lewis)            \
  X(nh3_h2s_lewis_ddT)        \
  X(h2s_ideal)                \
  X(h2s_ideal_ddT)            \
  X(h2s_antoine)              \
  X(h2s_antoine_ddT)          \
  X(ch4_ideal)                \
  X(ch4_ideal_ddT)            \
  X(so2_antoine)              \
  X(so2_antoine_ddT)          \
  X(co2_antoine)              \
  X(co2_antoine_ddT)          \
  X(kcl_lodders)              \
  X(kcl_lodders_ddT)          \
  X(na_h2s_visscher)          \
  X(na_h2s_visscher_ddT)

#define KINTERA_FUNC2_LIST(X) \
  X(h2_rot_normal)            \
  X(h2_rot_normal_ddT)        \
  X(h2_rot_normal_s)          \
  X(h2_rot_equilibrium)       \
  X(h2_rot_equilibrium_ddT)   \
  X(h2_rot_equilibrium_s)

#define KINTERA_FUNC3_LIST(X)

#define KINTERA_FUNC_PTR(f) f,
#define KINTERA_FUNC_NAME(f) #f,
