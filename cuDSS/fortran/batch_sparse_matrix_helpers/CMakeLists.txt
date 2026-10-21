# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.19)

set(EXAMPLE_NAME batch_sparse_matrix_helpers)

project("cuDSS_${EXAMPLE_NAME}_fortran_example"
        DESCRIPTION  "cuDSS Fortran example"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cudss/index.html"
        LANGUAGES    Fortran)

# The examples call the CUDA runtime and cuBLAS directly (through the cudautils
# module), so find the CUDA Toolkit before cuDSS -- this also lets the cuDSS
# package verify CUDA Toolkit compatibility at find_package() time.
find_package(CUDAToolkit REQUIRED)
find_package(cudss 1.0.0 REQUIRED)

# The Fortran interface ships as source (a .mod is compiler-specific), so it is
# compiled here together with the support modules and the example. CMake derives
# the Fortran module build order from the use/module statements.
add_executable(${EXAMPLE_NAME}_example
    ${cudss_INCLUDE_DIR}/fortran/cudss.f90
    ${CMAKE_CURRENT_SOURCE_DIR}/cudautils.f90
    ${CMAKE_CURRENT_SOURCE_DIR}/cudssutils.f90
    ${CMAKE_CURRENT_SOURCE_DIR}/validation.f90
    ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}.f90)

target_link_libraries(${EXAMPLE_NAME}_example PRIVATE
    cudss
    CUDA::cudart
    CUDA::cublas)
