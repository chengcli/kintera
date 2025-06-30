# Install kintera library
#
#   library: libkintera_debug.o
#   library: libkintera_release.o
#
# Define the following symbols
#
#   KINTERA_INCLUDE_DIR
#   kintera::kintera
#   kintera::vapors
#   kintera::kintera_cu
#
# Normal usage would be:
#
#   include_directories(${KINTERA_INCLUDE_DIR})
#   target_link_libraries(kintera::kintera)

include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  ${CMAKE_BINARY_DIR}
  SYSTEM ${TORCH_API_INCLUDE_DIR}
  SYSTEM ${TORCH_INCLUDE_DIR}
  SYSTEM ${HARP_INCLUDE_DIR}
  )

string(TOLOWER ${CMAKE_BUILD_TYPE} buildl)
string(TOUPPER ${CMAKE_BUILD_TYPE} buildu)

set(namel kintera)
string(TOUPPER ${namel} nameu)

file(GLOB src_files
  *.cpp
  xsection/*.cpp
  utils/*.cpp
  kinetics/*.cpp
  thermo/*.cpp
  math/*.cpp
  eos/*.cpp
  units/*.cpp
  )

add_library(${namel}_${buildl}
    SHARED
    ${src_files}
    )

set_target_properties(${namel}_${buildl}
    PROPERTIES
    COMPILE_FLAGS "${CMAKE_CXX_FLAGS_${buildu}}"
    )

target_link_libraries(${namel}_${buildl}
    PUBLIC
    fmt::fmt
    yaml-cpp::yaml-cpp
    ${TORCH_LIBRARY}
    ${TORCH_CPU_LIBRARY}
    ${C10_LIBRARY}
    ${HARP_LIBRARY}
    ${DISORT_LIBRARY}
    )

add_library(kintera::kintera ALIAS ${namel}_${buildl})

# CUDA source codes
if (CUDAToolkit_FOUND)
  file(GLOB cu_src_files
    *.cu
    thermo/*.cu
    utils/*.cu
    )

  add_library(${namel}_cuda_${buildl}
      SHARED
      ${cu_src_files}
      )

  set_target_properties(${namel}_cuda_${buildl}
      PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      )

  target_link_libraries(${namel}_cuda_${buildl}
      PUBLIC
      CUDA::cudart
      CUDA::cudart_static
      ${TORCH_CUDA_LIBRARY}
      ${C10_CUDA_LIBRARY}
      )

  add_library(kintera::kintera_cu ALIAS ${namel}_cuda_${buildl})
endif()

set(KINTERA_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/.."
  CACHE INTERNAL "kintera include directory")

add_subdirectory(vapors)
