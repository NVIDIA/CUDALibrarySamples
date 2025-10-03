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

project(cusolvermp_samples LANGUAGES C CUDA)

find_package(MPI REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_RUNTIME_LIBRARY shared)

function(build_sample sample_name)
    add_executable(${sample_name} ${sample_name}.c)

    target_include_directories(${sample_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${MPI_C_INCLUDE_DIRS}
        ${NCCL_INCLUDE_DIR}
        ${CUSOLVERMP_INCLUDE_DIRECTORIES}
    )

    target_link_directories(${sample_name} PUBLIC
        ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
        ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}/stubs
        ${CUSOLVERMP_LINK_DIRECTORIES}
    )

    target_link_libraries(${sample_name} PUBLIC
        ${MPI_C_LIBRARIES}
        ${NCCL_LIBRARIES}
        cudart
        cublas
        cublasLt
        cusolver
        cusolverMp
        cuda
        nvidia-ml
        m
    )

    install(TARGETS ${sample_name}
        DESTINATION "bin/"
    )
endfunction(build_sample)

build_sample("mp_getrf_getrs")
build_sample("mp_potrf_potrs")
build_sample("mp_potrf_potrs_fp32")
build_sample("mp_potrf_potrs_fp32emulation")
build_sample("mp_geqrf")
build_sample("mp_syevd")
build_sample("mp_sygvd")
build_sample("mp_gels")