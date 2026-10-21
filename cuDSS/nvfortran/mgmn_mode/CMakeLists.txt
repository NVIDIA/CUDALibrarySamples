# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.19)

set(EXAMPLE_NAME mgmn_mode)

project("cuDSS_${EXAMPLE_NAME}_nvfortran_example"
        DESCRIPTION  "cuDSS CUDA Fortran MGMN example"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cudss/index.html"
        LANGUAGES    Fortran C)

if(NOT CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
    message(FATAL_ERROR "The cuDSS CUDA Fortran examples require nvfortran "
                        "(configure with -DCMAKE_Fortran_COMPILER=nvfortran).")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(cudss 1.0.0 REQUIRED)
find_package(MPI REQUIRED COMPONENTS Fortran C)

# .cuf is CUDA Fortran; the C thunk stays C, so -cuda is applied to Fortran only.
set(CUF_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cudssutils.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}.cuf)
set_source_files_properties(${CUF_SOURCES} PROPERTIES LANGUAGE Fortran)

# At run time this example needs a cuDSS communication-layer library and mpirun.
add_executable(${EXAMPLE_NAME}_example
    ${cudss_INCLUDE_DIR}/fortran/cudss.f90
    ${cudss_INCLUDE_DIR}/fortran/cudss_mpi.f90
    ${cudss_INCLUDE_DIR}/fortran/cudss_fortran_mpi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/validation.f90
    ${CUF_SOURCES})

target_include_directories(${EXAMPLE_NAME}_example PRIVATE ${cudss_INCLUDE_DIR})
target_compile_options(${EXAMPLE_NAME}_example PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-cuda>)
target_link_options(${EXAMPLE_NAME}_example PRIVATE -cuda -cudalib=cublas)
target_link_libraries(${EXAMPLE_NAME}_example PRIVATE
    cudss
    CUDA::cudart
    MPI::MPI_Fortran
    MPI::MPI_C)
