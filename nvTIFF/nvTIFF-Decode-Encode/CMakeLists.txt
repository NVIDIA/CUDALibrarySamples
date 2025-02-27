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

project(nvTiff_example LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)


find_library(NVTIFF_LIB
     NAMES nvtiff
     PATHS ${NVTIFF_PATH}/lib64/${CUDAToolkit_VERSION_MAJOR}
           ${NVTIFF_PATH}/lib)


if(NOT NVTIFF_LIB)
  message(FATAL_ERROR, " nvTIFF library not found. Make sure the library is installed. If using the tar.xz/zip packages set -DNVTIFF_PATH")
else()
  message(INFO, " nvTIFF library found here: ${NVTIFF_LIB}")
  endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

include_directories(
  ${NVTIFF_PATH}/include
  SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(EXAMPLES_DESCRIPTOR_SOURCES
  nvtiff_example.cpp
  )

add_executable(nvTiff_example ${EXAMPLES_DESCRIPTOR_SOURCES})


if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

target_link_libraries(nvTiff_example PUBLIC ${NVTIFF_LIB} CUDA::cudart)

