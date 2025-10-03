# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
cmake_minimum_required(VERSION 3.19)

set(EXAMPLE_NAME simple_residual)

project("cuDSS_${EXAMPLE_NAME}_example"
        DESCRIPTION  "cuDSS"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cudss/index.html"
        LANGUAGES    CXX CUDA)

set(CMAKE_CUDA_STANDARD          11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS        OFF)

option(BUILD_STATIC "Building statically linked examples" OFF)

# Find cuDSS
find_package(cudss 0.7.0 REQUIRED)

set_source_files_properties(${EXAMPLE_NAME}.cpp PROPERTIES LANGUAGE CUDA)

add_executable(${EXAMPLE_NAME}_example)

target_sources(${EXAMPLE_NAME}_example
    PUBLIC ${PROJECT_SOURCE_DIR}/${EXAMPLE_NAME}.cpp
)

target_include_directories(${EXAMPLE_NAME}_example PUBLIC ${cudss_INCLUDE_DIR})
target_link_directories(${EXAMPLE_NAME}_example PUBLIC ${cudss_LIBRARY_DIR})

target_compile_options(${EXAMPLE_NAME}_example PRIVATE
    "$<$<CONFIG:Debug>:-lineinfo -g>"
    "$<$<CONFIG:RelWithDebInfo>:-lineinfo -g>"
)

target_link_libraries(${EXAMPLE_NAME}_example PUBLIC
    cudss
    cusparse
)

if(BUILD_STATIC)
    add_executable(${EXAMPLE_NAME}_example_static)

    target_sources(${EXAMPLE_NAME}_example_static
        PUBLIC ${PROJECT_SOURCE_DIR}/${EXAMPLE_NAME}.cpp
    )

    target_include_directories(${EXAMPLE_NAME}_example_static PUBLIC ${cudss_INCLUDE_DIR})
    target_link_directories(${EXAMPLE_NAME}_example_static PUBLIC ${cudss_LIBRARY_DIR})

    if (WIN32)
        target_link_libraries(${EXAMPLE_NAME}_example_static PUBLIC
        cudss
    )
    else()
        target_link_libraries(${EXAMPLE_NAME}_example_static PUBLIC
            cudss_static
        )
    endif()

endif()
