include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

set(patch_command
    git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/02.gtest.patch
    )

set(PACKAGE_NAME gtest)
set(REPO_URL "https://github.com/google/googletest")
set(REPO_TAG "v1.13.0")
set(INSTALL_GTEST OFF)

add_package(${PACKAGE_NAME} ${REPO_URL} ${REPO_TAG} "${patch_command}" ON)
include_directories(${gtest_SOURCE_DIR}/include
                    ${gtest_SOURCE_DIR}/googletest/include)
