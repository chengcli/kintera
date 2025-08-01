# General setup for GCC and compilers sufficiently close to GCC:
#

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Add the --extended-lambda flag to CUDA compilation
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_CXX_FLAGS_RELEASE
      "-O3 -funroll-loops -funroll-all-loops -fstrict-aliasing")
  set(CMAKE_C_FLAGS_RELEASE
      "-O3 -funroll-loops -funroll-all-loops -fstrict-aliasing")

  set(CMAKE_CXX_FLAGS_DEBUG "-g3 -fsanitize=address")
  set(CMAKE_C_FLAGS_DEBUG "-g3 -fsanitize=address")

  set(KNOWN_COMPILER TRUE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -fstrict-aliasing")
  set(CMAKE_C_FLAGS_RELEASE "-O3 -funroll-loops -fstrict-aliasing")

  set(CMAKE_CXX_FLAGS_DEBUG "-g3 -fsanitize=address")
  set(CMAKE_C_FLAGS_DEBUG "-g3 -fsanitize=address")

  set(KNOWN_COMPILER TRUE)
endif()

#
# Setup for ICC compiler (version >= 10):
#
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  add_link_options("-fuse-ld=lld")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -lstdc++")
  set(CMAKE_C_FLAGS_RELEASE "-O3 -lstdc++")

  set(CMAKE_CXX_FLAGS_DEBUG "-g3")
  set(CMAKE_C_FLAGS_DEBUG "-g3")

  set(KNOWN_COMPILER TRUE)
endif()

#
# Setup for MSVC compiler (version >= 2012):
#
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  message(FATAL_ERROR "\n" "MSVC compiler not implemented.\n\n")
  # set(_flags ${CMAKE_SOURCE_DIR}/cmake/compiler_flags_msvc.cmake)
  # message(STATUS "Include ${_flags}") include(${_flags}) set(KNOWN_COMPILER
  # TRUE)
endif()

if(NOT KNOWN_COMPILER)
  message(
    FATAL_ERROR
      "\n" "Unknown compiler!\n"
      "If you're serious about it, set SETUP_DEFAULT_COMPILER_FLAGS=OFF "
      "and set the relevant compiler options by hand.\n\n")
endif()
