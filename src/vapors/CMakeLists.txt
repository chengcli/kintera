set(namel vapors)
string(TOUPPER ${namel} nameu)

if (CUDAToolkit_FOUND)
  file(GLOB src_files vapor_functions.cu)

  add_library(${namel}_${buildl}
      OBJECT
      ${src_files}
      )

  set_target_properties(${namel}_${buildl}
      PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      )
else()
  file(GLOB src_files vapor_functions.cpp)

  add_library(${namel}_${buildl}
      OBJECT
      ${src_files}
      )

  set_target_properties(${namel}_${buildl}
      PROPERTIES
      COMPILE_FLAGS "${CMAKE_CXX_FLAGS_${buildu}}"
      )
endif()

add_library(kintera::vapors ALIAS ${namel}_${buildl})
