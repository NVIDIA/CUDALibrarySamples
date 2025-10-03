# SPDX-FileCopyrightText: Copyright (c) 1993-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(ROUTINE 3d_mgpu_c2c)

project(
  "${ROUTINE}_example"
  DESCRIPTION "GPU-Accelerated Fast Fourier Transforms"
  HOMEPAGE_URL "https://docs.nvidia.com/cuda/cufft/index.html"
  LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CUDA_ARCHITECTURES LESS 75)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 90 100 120)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_source_files_properties(${ROUTINE}_example.cpp PROPERTIES LANGUAGE CUDA)
add_executable(${ROUTINE}_example)

target_include_directories(${ROUTINE}_example
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           ${CMAKE_SOURCE_DIR}/../utils)

target_sources(${ROUTINE}_example
               PRIVATE ${PROJECT_SOURCE_DIR}/${ROUTINE}_example.cpp)

target_link_libraries(${ROUTINE}_example PRIVATE CUDA::cufft CUDA::cudart)