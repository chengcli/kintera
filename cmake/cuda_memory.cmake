include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

set(PACKAGE_NAME cumem)
set(REPO_URL "https://github.com/gcereska/cuda-memory")
set(REPO_TAG "v0.1")

add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "" ON)
include_directories(${cumem_SOURCE_DIR})
