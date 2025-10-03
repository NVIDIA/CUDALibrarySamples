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

cmake_minimum_required(VERSION 3.12)
project(cufftmp_jax LANGUAGES CXX CUDA)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/src)

message(STATUS "Using ${NVSHMEM_HOME} for NVSHMEM_HOME and ${CUFFTMP_HOME} for CUFFTMP_HOME")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUFFTMP_HOME}/include ${NVSHMEM_HOME}/include)
link_directories(${CUFFTMP_HOME}/lib ${NVSHMEM_HOME}/lib)

pybind11_add_module(gpu_ops 
    ${CMAKE_CURRENT_LIST_DIR}/src/kernels.cu 
    ${CMAKE_CURRENT_LIST_DIR}/src/gpu_ops.cpp
)

target_link_libraries(gpu_ops 
    PRIVATE 
        cufftMp
        nvshmem_host
        nvshmem_device
)

set_target_properties(gpu_ops 
    PROPERTIES
        CUDA_STANDARD 17
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
)

install(TARGETS gpu_ops DESTINATION cufftmp_jax)