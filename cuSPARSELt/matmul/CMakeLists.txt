# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(ROUTINE matmul)

project("${ROUTINE}_example"
        DESCRIPTION  "cuSPARSELt"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cusparselt/index.html"
        LANGUAGES    CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD           14)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CXX_EXTENSIONS         OFF)
set(CMAKE_CUDA_STANDARD          14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS        OFF)

add_executable(${ROUTINE}_example)
add_executable(${ROUTINE}_example_static)

target_sources(${ROUTINE}_example
    PUBLIC ${PROJECT_SOURCE_DIR}/${ROUTINE}_example.cpp
)

target_sources(${ROUTINE}_example_static
    PUBLIC ${PROJECT_SOURCE_DIR}/${ROUTINE}_example.cpp
)

target_include_directories(${ROUTINE}_example
    PUBLIC ${CUDAToolkit_INCLUDE_DIRS}
    PUBLIC ${CUSPARSELT_PATH}/include
)

target_include_directories(${ROUTINE}_example_static
    PUBLIC ${CUDAToolkit_INCLUDE_DIRS}
    PUBLIC ${CUSPARSELT_PATH}/include
)

target_link_libraries(${ROUTINE}_example
    PUBLIC CUDA::cudart
    PUBLIC CUDA::cusparse
    PUBLIC ${CUSPARSELT_PATH}/lib64/libcusparseLt.so
)

target_link_libraries(${ROUTINE}_example_static
    PUBLIC CUDA::cudart
    PUBLIC CUDA::cusparse
    PUBLIC ${CUSPARSELT_PATH}/lib64/libcusparseLt_static.a
    PUBLIC ${CMAKE_DL_LIBS}
)