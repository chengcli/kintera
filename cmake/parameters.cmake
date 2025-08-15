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

# maximum number of branches of photolysis reactions
if(NOT MAX_PHOTO_BRANCHES OR NOT DEFINED MAX_PHOTO_BRANCHES)
  set(MAX_PHOTO_BRANCHES 3)
endif()

# memory pool options
if(NOT USE_MEMORY_POOL OR NOT DEFINED USE_MEMORY_POOL)
  set(MEMORY_POOL_OPTION "NO_USE_MEMORY_POOL")
else()
  set(MEMORY_POOL_OPTION "USE_MEMORY_POOL")
endif()
