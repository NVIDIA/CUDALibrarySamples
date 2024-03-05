#******************************************************************************
# * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# *
# * NVIDIA CORPORATION and its licensors retain all intellectual property
# * and proprietary rights in and to this software, related documentation
# * and any modifications thereto.  Any use, reproduction, disclosure or
# * distribution of this software and related documentation without an express
# * license agreement from NVIDIA CORPORATION is strictly prohibited.
# ******************************************************************************

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
    PUBLIC CUDA::cuda_driver
    PUBLIC ${CMAKE_DL_LIBS}
)
