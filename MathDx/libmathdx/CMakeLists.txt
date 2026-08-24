# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.31)

# libmathdx Public Examples project
project(libmathdx_public_examples LANGUAGES CXX)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find libmathdx package (uses the installed libmathdx-config.cmake)
find_package(libmathdx CONFIG REQUIRED)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/sample.cmake")

foreach(example ${LIBMATHDX_EXAMPLES})
    add_executable(${example} "${example}.cpp")

    # Include common headers
    target_include_directories(${example} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/common")

    # Link libraries (static linking)
    if(NOT WIN32)
        target_link_libraries(${example} PRIVATE -static-libstdc++ -static-libgcc)
    endif()
    target_link_libraries(${example} PRIVATE libmathdx::libmathdx CUDA::nvrtc CUDA::nvJitLink CUDA::cuda_driver CUDA::cudart)

    # Add CTest tests
    add_test(NAME ${example} COMMAND ${example})
endforeach()

message(STATUS "Configured public examples: ${LIBMATHDX_EXAMPLES}")
