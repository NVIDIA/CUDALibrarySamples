# SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.18.0 FATAL_ERROR)

# ---[ Routine name
set(ROUTINE1 mtgp32_uniform)
set(ROUTINE2 mtgp32_normal)
set(ROUTINE3 mtgp32_lognormal)
set(ROUTINE4 mtgp32_poisson)

# ---[ Project specification.
project(curand_examples LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

include(GNUInstallDirs)

# ##########################################
# curand_examples build mode
# ##########################################

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# ##########################################
# curand_examples building flags
# ##########################################

# Global CXX/CUDA flags

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Debug options
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -O0 -g -lineinfo")

# ##########################################
# curand_examples target
# ##########################################
include(../../cmake/curand_example.cmake)

include_directories("${CMAKE_SOURCE_DIR}/../../utils")

# ADD 3 ROUTINES for uniform, normal, & lognormal
add_curand_example(curand_examples "curand_${ROUTINE1}_example" curand_${ROUTINE1}_example.cpp)
add_curand_example(curand_examples "curand_${ROUTINE2}_example" curand_${ROUTINE2}_example.cpp)
add_curand_example(curand_examples "curand_${ROUTINE3}_example" curand_${ROUTINE3}_example.cpp)
add_curand_example(curand_examples "curand_${ROUTINE4}_example" curand_${ROUTINE4}_example.cpp)

# ##########################################
# curand_examples directories
# ##########################################

# By default put binaries in build/bin (pre-install)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Installation directories
set(CURAND_EXAMPLES_BINARY_INSTALL_DIR "curand_examples/bin")

# ##########################################
# Install examples
# ##########################################

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
ENDIF()