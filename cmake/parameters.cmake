# define default parameters

# netcdf options
if(NOT NETCDF OR NOT DEFINED NETCDF)
  set(NETCDF_OPTION "NO_NETCDFOUTPUT")
else()
  set(NETCDF_OPTION "NETCDFOUTPUT")
  find_package(NetCDF REQUIRED)
endif()

# pnetcdf options
if(NOT PNETCDF OR NOT DEFINED PNETCDF)
  set(PNETCDF_OPTION "NO_PNETCDFOUTPUT")
else()
  set(PNETCDF_OPTION "PNETCDFOUTPUT")
  find_package(PNetCDF REQUIRED)
endif()

# CUDA flags
if(NOT CUDA OR NOT DEFINED CUDA)
  set(CUDA_OPTION "DISABLE_CUDA")
else()
  set(CUDA_OPTION "ENABLE_CUDA")
endif()

# maximum number of branches of photolysis reactions
if(NOT MAX_PHOTO_BRANCHES OR NOT DEFINED MAX_PHOTO_BRANCHES)
  set(MAX_PHOTO_BRANCHES 3)
endif()
