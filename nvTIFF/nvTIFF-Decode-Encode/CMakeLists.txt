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

project(nvTiff_example LANGUAGES C CXX CUDA)

if(NOT NVTIFF_PATH)
message(FATAL_ERROR, "NVTIFF_PATH not set")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

include_directories(
  ${NVTIFF_PATH}/include
  SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(EXAMPLES_DESCRIPTOR_SOURCES
  nvtiff_example.cu
  nvTiff_utils.cpp
  )

add_executable(nvTiff_example ${EXAMPLES_DESCRIPTOR_SOURCES})

find_library(nvTIFF_LIB
    NAMES nvtiff
    PATHS ${NVTIFF_PATH}/lib64)

find_library(CUDART_LIB 
    NAMES cudart
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

if(UNIX)
  set(FILESYS -lstdc++fs)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

target_link_libraries(nvTiff_example PUBLIC ${nvTIFF_LIB} ${CUDART_LIB} ${FILESYS})

