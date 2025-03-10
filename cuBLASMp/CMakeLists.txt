# 
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

cmake_minimum_required(VERSION 3.16)

project(cublasmp_samples LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_RUNTIME_LIBRARY shared)

if (CAL_MPI_INCLUDE_PATH AND CAL_MPI_LIBRARIES)
    add_definitions(-DUSE_CAL_MPI)
endif ()

function(build_sample sample_name)
    add_executable(${sample_name} ${sample_name}.cu)

    set_target_properties(${sample_name} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    target_include_directories(${sample_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${MPI_CXX_INCLUDE_DIRS}
        ${CAL_INCLUDE_DIRECTORIES}
        ${CUBLASMP_INCLUDE_DIRECTORIES}
        ${CAL_MPI_INCLUDE_PATH}
        ${NVSHMEM_INCLUDE_DIRECTORIES}
    )

    target_link_directories(${sample_name} PUBLIC
        ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
        ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}/stubs
    )

    target_link_libraries(${sample_name} PUBLIC
        cudart
        ${CUBLASMP_LIBRARIES}
        ${CAL_LIBRARIES}
        ${MPI_CXX_LIBRARIES}
        ${NVSHMEM_HOST_LIBRARIES}
        ${NVSHMEM_DEVICE_LIBRARIES}
        nvidia-ml
        m
        ${CAL_MPI_LIBRARIES}
    )

    install(TARGETS ${sample_name}
        DESTINATION "bin/"
    )
endfunction(build_sample)

build_sample("pgemm")
build_sample("pmatmul")
build_sample("pmatmul_ar")
build_sample("ptrsm")
build_sample("psyrk")
build_sample("pgeadd")
build_sample("ptradd")
