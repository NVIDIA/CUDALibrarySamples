# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(nvJPEGROIDecode LANGUAGES CXX CUDA)

# ---[ Project specIFication.
SET(PROJECT_NAME nvJPEGROIDecode)
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


SET(EXAMPLES_DESCRIPTOR_SOURCES "nvJPEGROIDecode.cpp")

add_executable(nvJPEGROIDecode ${EXAMPLES_DESCRIPTOR_SOURCES})

find_library(NVJPEG_LIB
    NAMES nvjpeg
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

find_library(CUDART_LIB 
    NAMES cudart
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

target_link_libraries(nvJPEGROIDecode PUBLIC ${NVJPEG_LIB} ${CUDART_LIB} pthread)