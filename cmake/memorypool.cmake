include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

set(PACKAGE_NAME memorypool)
set(REPO_URL "https://github.com/tongshengw/memorypool")
set(REPO_TAG "v0.1.17")

add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "" ON)
set(MEMORYPOOL_INCLUDE_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/_deps/${PACKAGE_NAME}-src/gpu"
  CACHE PATH "memorypool include directory")
