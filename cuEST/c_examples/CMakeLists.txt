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
cmake_minimum_required(VERSION 3.20)

project(cuest_c_examples LANGUAGES C)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

if(NOT DEFINED CUEST_INSTALL_DIR)
    message(FATAL_ERROR
        "CUEST_INSTALL_DIR is not set.\n"
        "Re-run cmake with: -DCUEST_INSTALL_DIR=<path to cuest installation>")
endif()

set(CUEST_INCLUDE_DIR ${CUEST_INSTALL_DIR}/include)

if(NOT EXISTS ${CUEST_INCLUDE_DIR})
    message(FATAL_ERROR "CUEST include directory not found: ${CUEST_INCLUDE_DIR}")
endif()

# Support both lib/libcuest.so and lib/<cuda_major>/libcuest.so layouts
if(EXISTS ${CUEST_INSTALL_DIR}/lib/libcuest.so)
    set(CUEST_LIBRARY ${CUEST_INSTALL_DIR}/lib/libcuest.so)
elseif(EXISTS ${CUEST_INSTALL_DIR}/lib/${CUDAToolkit_VERSION_MAJOR}/libcuest.so)
    set(CUEST_LIBRARY ${CUEST_INSTALL_DIR}/lib/${CUDAToolkit_VERSION_MAJOR}/libcuest.so)
else()
    message(FATAL_ERROR "cuest library not found under ${CUEST_INSTALL_DIR}/lib/")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 90)
endif()

add_subdirectory(examples)
