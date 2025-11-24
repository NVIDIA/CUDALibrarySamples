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

cmake_minimum_required(VERSION 3.16)

project(cublasmp_samples LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_RUNTIME_LIBRARY shared)

function(build_sample sample_name)
    add_executable(${sample_name} ${sample_name}.cu)

    set_target_properties(${sample_name} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    target_include_directories(${sample_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${MPI_CXX_INCLUDE_DIRS}
        ${CUBLASMP_INCLUDE_DIRECTORIES}
        ${NCCL_INCLUDE_DIRECTORIES}
        ${NVSHMEM_INCLUDE_DIRECTORIES}
    )

    target_link_directories(${sample_name} PUBLIC
        ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
        ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}/stubs
    )

    target_link_libraries(${sample_name} PUBLIC
        cudart
        ${CUBLASMP_LIBRARIES}
        ${MPI_CXX_LIBRARIES}
        ${NCCL_LIBRARIES}
        ${NVSHMEM_DEVICE_LIBRARIES}
        ${NVSHMEM_HOST_LIBRARIES}
        nvidia-ml
        m
    )

    install(TARGETS ${sample_name}
        DESTINATION "bin/"
    )
endfunction(build_sample)

build_sample("gemm")
build_sample("tp_matmul")
build_sample("matmul_ag")
build_sample("matmul_rs")
build_sample("matmul_ar")
build_sample("trsm")
build_sample("syrk")
build_sample("geadd")
build_sample("tradd")
build_sample("gemr2d")
