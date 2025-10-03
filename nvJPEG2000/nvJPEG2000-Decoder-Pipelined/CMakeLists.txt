# SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(nvjpeg2k_sample LANGUAGES CXX CUDA)

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

set(EXAMPLES_DESCRIPTOR_SOURCES "nvjpeg2k_dec_pipelined.cpp")

add_executable(nvjpeg2k_dec_pipelined ${EXAMPLES_DESCRIPTOR_SOURCES})


if(UNIX)
  set(FILESYS -lstdc++fs)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

target_link_libraries(nvjpeg2k_dec_pipelined PUBLIC ${NVJPEG2K_LIB} CUDA::cudart ${FILESYS})
