# 
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(nvJPEGDecMultipleInstances LANGUAGES CXX CUDA)

# ---[ Project specIFication.
SET(PROJECT_NAME nvJPEGDecMultipleInstances)
PROJECT(${PROJECT_NAME} LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)    
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


include_directories(
  SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)


SET(EXAMPLES_DESCRIPTOR_SOURCES "nvJPEGDecMultipleInstances.cpp")

add_executable(nvJPEGDecMultipleInstances ${EXAMPLES_DESCRIPTOR_SOURCES})

find_library(NVJPEG_LIB
    NAMES nvjpeg
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

find_library(CUDART_LIB 
    NAMES cudart
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

target_link_libraries(nvJPEGDecMultipleInstances PUBLIC ${NVJPEG_LIB} ${CUDART_LIB} pthread)

