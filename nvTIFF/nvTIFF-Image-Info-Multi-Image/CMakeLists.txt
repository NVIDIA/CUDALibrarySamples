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

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(nvtiff_image_info LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# NVTIFF_PATH points at an nvTIFF install prefix or an extracted archive
# directory, e.g. nvtiff-linux-x86_64-<version>/cuda13.
find_library(NVTIFF_LIB
    NAMES nvtiff
    HINTS ${NVTIFF_PATH}/lib64/${CUDAToolkit_VERSION_MAJOR}
          ${NVTIFF_PATH}/lib/${CUDAToolkit_VERSION_MAJOR}
          ${NVTIFF_PATH}/lib64
          ${NVTIFF_PATH}/lib)
if(NOT NVTIFF_LIB)
    message(FATAL_ERROR "nvTIFF library not found. Set -DNVTIFF_PATH to an nvTIFF install or extracted archive (see README).")
endif()
message(STATUS "Using nvTIFF library: ${NVTIFF_LIB}")

add_executable(nvtiff_image_info nvtiff_image_info.cpp)
target_include_directories(nvtiff_image_info PRIVATE
    ${NVTIFF_PATH}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(nvtiff_image_info PRIVATE ${NVTIFF_LIB})
if(UNIX)
    target_compile_options(nvtiff_image_info PRIVATE -Wall)
endif()
