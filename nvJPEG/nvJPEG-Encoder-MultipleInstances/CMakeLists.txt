# 
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)


option(CROSS_COMPILE_AARCH64 "Cross compile for ARM64" OFF)

if(CROSS_COMPILE_AARCH64)
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(CMAKE_CUDA_HOST_COMPILER aarch64-linux-gnu-g++)
endif()

project(nvJPEGEncMultipleInstances LANGUAGES CXX CUDA)

find_package(Threads REQUIRED)
find_package(CUDAToolkit 12.9 REQUIRED)

add_executable(nvJPEGEncMultipleInstances "nvJPEGEncMultipleInstances.cpp")

target_compile_features(nvJPEGEncMultipleInstances PRIVATE
    cxx_std_17
    cuda_std_17)

target_link_libraries(nvJPEGEncMultipleInstances PUBLIC
    CUDA::nvjpeg
    CUDA::cudart_static
    Threads::Threads)