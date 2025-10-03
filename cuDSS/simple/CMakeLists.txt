# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.19)

set(EXAMPLE_NAME simple)

project("cuDSS_${EXAMPLE_NAME}_example"
        DESCRIPTION  "cuDSS"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cudss/index.html"
        LANGUAGES    CXX CUDA)

set(CMAKE_CUDA_STANDARD          11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS        OFF)

option(BUILD_STATIC "Building statically linked examples" OFF)

# Find cuDSS
find_package(cudss 0.6.0 REQUIRED)

set_source_files_properties(${EXAMPLE_NAME}.cpp PROPERTIES LANGUAGE CUDA)

add_executable(${EXAMPLE_NAME}_example)

target_sources(${EXAMPLE_NAME}_example
    PUBLIC ${PROJECT_SOURCE_DIR}/${EXAMPLE_NAME}.cpp
)

if (WIN32)
    target_include_directories(${EXAMPLE_NAME}_example PUBLIC ${cudss_INCLUDE_DIR})
    target_link_directories(${EXAMPLE_NAME}_example PUBLIC ${cudss_LIBRARY_DIR})
endif()

target_compile_options(${EXAMPLE_NAME}_example PRIVATE
    "$<$<CONFIG:Debug>:-lineinfo -g>"
    "$<$<CONFIG:RelWithDebInfo>:-lineinfo -g>"
)

target_link_libraries(${EXAMPLE_NAME}_example PUBLIC
    cudss
)

if(BUILD_STATIC)
    add_executable(${EXAMPLE_NAME}_example_static)

    target_sources(${EXAMPLE_NAME}_example_static
        PUBLIC ${PROJECT_SOURCE_DIR}/${EXAMPLE_NAME}.cpp
    )

    if (WIN32)
        target_include_directories(${EXAMPLE_NAME}_example_static PUBLIC ${cudss_INCLUDE_DIR})
        target_link_directories(${EXAMPLE_NAME}_example_static PUBLIC ${cudss_LIBRARY_DIR})
        target_link_libraries(${EXAMPLE_NAME}_example_static PUBLIC
        cudss
    )
    else()
        target_link_libraries(${EXAMPLE_NAME}_example_static PUBLIC
            cudss_static
        )
    endif()

endif()