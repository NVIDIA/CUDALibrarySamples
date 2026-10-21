# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.19)

set(EXAMPLE_NAME uniform_batch)

project("cuDSS_${EXAMPLE_NAME}_nvfortran_example"
        DESCRIPTION  "cuDSS CUDA Fortran example"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cudss/index.html"
        LANGUAGES    Fortran)

# CUDA Fortran (.cuf) requires the NVIDIA HPC compiler nvfortran.
if(NOT CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    message(FATAL_ERROR "The cuDSS CUDA Fortran examples require nvfortran "
                        "(configure with -DCMAKE_Fortran_COMPILER=nvfortran).")
endif()

find_package(cudss 1.0.0 REQUIRED)

# .cuf is CUDA Fortran; tell CMake it is a Fortran source and enable it via -cuda.
set(CUF_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cudssutils.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}.cuf)
set_source_files_properties(${CUF_SOURCES} PROPERTIES LANGUAGE Fortran)

add_executable(${EXAMPLE_NAME}_example
    ${cudss_INCLUDE_DIR}/fortran/cudss.f90
    ${CMAKE_CURRENT_SOURCE_DIR}/validation.f90
    ${CUF_SOURCES})

# nvfortran: -cuda enables CUDA Fortran and links the CUDA runtime; cuBLAS via -cudalib.
target_compile_options(${EXAMPLE_NAME}_example PRIVATE -cuda)
target_link_options(${EXAMPLE_NAME}_example PRIVATE -cuda -cudalib=cublas)
target_link_libraries(${EXAMPLE_NAME}_example PRIVATE cudss)
