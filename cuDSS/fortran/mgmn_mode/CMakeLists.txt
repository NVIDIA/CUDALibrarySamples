# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.19)

set(EXAMPLE_NAME mgmn_mode)

project("cuDSS_${EXAMPLE_NAME}_fortran_example"
        DESCRIPTION  "cuDSS Fortran MGMN example"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cudss/index.html"
        LANGUAGES    Fortran C)

find_package(CUDAToolkit REQUIRED)
find_package(cudss 1.0.0 REQUIRED)
find_package(MPI REQUIRED COMPONENTS Fortran C)

# MGMN also compiles the cudss_mpi module and its C thunk (which converts the
# Fortran MPI communicator via MPI_Comm_f2c). At run time this example needs a
# cuDSS communication-layer library (e.g. libcudss_commlayer_openmpi.so) and must
# be launched with mpirun; see the cuDSS documentation.
add_executable(${EXAMPLE_NAME}_example
    ${cudss_INCLUDE_DIR}/fortran/cudss.f90
    ${cudss_INCLUDE_DIR}/fortran/cudss_mpi.f90
    ${cudss_INCLUDE_DIR}/fortran/cudss_fortran_mpi.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cudautils.f90
    ${CMAKE_CURRENT_SOURCE_DIR}/cudssutils.f90
    ${CMAKE_CURRENT_SOURCE_DIR}/validation.f90
    ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_NAME}.f90)

# The C thunk includes cudss.h and <cuda_runtime.h>.
target_include_directories(${EXAMPLE_NAME}_example PRIVATE ${cudss_INCLUDE_DIR})

target_link_libraries(${EXAMPLE_NAME}_example PRIVATE
    cudss
    CUDA::cudart
    CUDA::cublas
    MPI::MPI_Fortran
    MPI::MPI_C)
