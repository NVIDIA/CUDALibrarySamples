# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.18)

# Define project
project(cufft_device_api_lto_helper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CUFFTDX_CUFFT_MIN_VERSION)
    set(CUFFTDX_CUFFT_MIN_VERSION "12.1" CACHE STRING "Minimum required cuFFT library version")
endif()

# Possible variables that can be set externally
# cufft_ROOT: points to the cuFFT library root directory
list(APPEND CMAKE_MODULE_PATH "${cufftdx_cufft_MODULE_PATH}")

find_package(cufft ${CUFFTDX_CUFFT_MIN_VERSION} REQUIRED)

# Define the helper executable
add_executable(cufft_device_api_lto_helper cufft_device_api_lto_helper.cpp)
target_compile_definitions(cufft_device_api_lto_helper
    PRIVATE
        CUFFT_ENABLE_EXPERIMENTAL_API
)
target_compile_definitions(cufft_device_api_lto_helper
    PRIVATE
        CUFFTDX_ENABLE_CUFFT_DEPENDENCY # To include cufftdx::utils::check_cufft_device_api_version helper function
)

target_link_libraries(cufft_device_api_lto_helper
    PUBLIC
        cufft::cufft_static
)

target_include_directories(cufft_device_api_lto_helper
    PRIVATE
        "${cufftdx_INCLUDE_DIRS}"
        "${cufftdx_commondx_INCLUDE_DIR}"
)
