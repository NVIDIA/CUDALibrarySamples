# 
# Copyright (c) 2020 - 2023, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(nvjpeg2000_sample LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)


find_library(NVJPEG2K_LIB
     NAMES nvjpeg2k
     PATHS ${NVJPEG2K_PATH}/lib64/${CUDAToolkit_VERSION_MAJOR}
           ${NVJPEG2K_PATH}/lib/${CUDAToolkit_VERSION_MAJOR})

if(NOT NVJPEG2K_LIB)
  message(FATAL_ERROR, " nvJPEG2000 library not found. Make sure the library is installed. If using the tar.xz/zip packages set -DNVJPEG2K_PATH")
else()
  message(INFO, " nvJPEG2000 library found here: ${NVJPEG2K_LIB}")
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

include_directories(
  ${NVJPEG2K_PATH}/include
  SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)


set(EXAMPLES_DESCRIPTOR_SOURCES "nvjpeg2000DecodeSample.cpp")

add_executable(nvjpeg2000_decode_sample ${EXAMPLES_DESCRIPTOR_SOURCES})

if(UNIX)
  set(FILESYS -lstdc++fs)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

target_link_libraries(nvjpeg2000_decode_sample PUBLIC ${NVJPEG2K_LIB} CUDA::cudart ${FILESYS})

