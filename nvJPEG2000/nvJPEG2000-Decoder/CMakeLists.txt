# 
# Copyright (c) 2020 - 2021, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(nvjpeg2000_sample LANGUAGES CXX CUDA)

if(NOT NVJPEG2K_PATH)
message(FATAL_ERROR, "NVJPEG2K_PATH not set")
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


SET(EXAMPLES_DESCRIPTOR_SOURCES "nvjpeg2000DecodeSample.cpp")

add_executable(nvjpeg2000_decode_sample ${EXAMPLES_DESCRIPTOR_SOURCES})

find_library(NVJPEG2K_LIB
    NAMES nvjpeg2k
    PATHS ${NVJPEG2K_PATH}/lib64)

find_library(CUDART_LIB 
    NAMES cudart
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

if(UNIX)
  set(FILESYS -lstdc++fs)
endif()

target_link_libraries(nvjpeg2000_decode_sample PUBLIC ${NVJPEG2K_LIB} ${CUDART_LIB} ${FILESYS})

