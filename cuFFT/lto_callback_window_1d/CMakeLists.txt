# SPDX-FileCopyrightText: Copyright (c) 1993-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.18)

project(
  lto_ea_sample
  DESCRIPTION "cuFFT LTO EA preview sample"
  HOMEPAGE_URL "https://docs.nvidia.com/cuda/cufft/index.html"
  LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CUDA_ARCHITECTURES LESS 75)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 90 100 120)
endif()

option(CB_USE_CONSTANT_MEMORY "Use constant memory for windowing callback parameters" OFF)

set(CUDA_LTO_ARCHITECTURE 75)
set(GENCODE_FLAGS_LTO --generate-code arch=compute_${CUDA_LTO_ARCHITECTURE},code=lto_${CUDA_LTO_ARCHITECTURE})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(r2c_c2r_lto_callback_example)
add_executable(r2c_c2r_lto_nvrtc_callback_example)
add_executable(r2c_c2r_legacy_callback_example)

target_include_directories(r2c_c2r_lto_callback_example
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_BINARY_DIR})
target_include_directories(r2c_c2r_lto_nvrtc_callback_example
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_BINARY_DIR})
target_include_directories(r2c_c2r_legacy_callback_example
                           PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                           ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_BINARY_DIR})

target_sources(r2c_c2r_lto_callback_example
               PRIVATE ${PROJECT_SOURCE_DIR}/src/r2c_c2r_lto_callback_example.cpp
                       ${PROJECT_SOURCE_DIR}/src/common.cpp
                       ${PROJECT_SOURCE_DIR}/src/r2c_c2r_reference.cu
                       ${CMAKE_BINARY_DIR}/r2c_c2r_lto_callback_device_fatbin.h)

target_sources(r2c_c2r_lto_nvrtc_callback_example
               PRIVATE ${PROJECT_SOURCE_DIR}/src/r2c_c2r_lto_nvrtc_callback_example.cpp
                       ${PROJECT_SOURCE_DIR}/src/common.cpp
                       ${PROJECT_SOURCE_DIR}/src/r2c_c2r_reference.cu)

target_sources(r2c_c2r_legacy_callback_example
               PRIVATE ${PROJECT_SOURCE_DIR}/src/r2c_c2r_legacy_callback_example.cu
                       ${PROJECT_SOURCE_DIR}/src/common.cpp
                       ${PROJECT_SOURCE_DIR}/src/r2c_c2r_reference.cu)

if(CB_USE_CONSTANT_MEMORY)
  target_compile_definitions(r2c_c2r_legacy_callback_example PRIVATE CB_USE_CONSTANT_MEMORY)
  target_compile_definitions(r2c_c2r_lto_callback_example PRIVATE CB_USE_CONSTANT_MEMORY)
  target_compile_definitions(r2c_c2r_lto_nvrtc_callback_example PRIVATE CB_USE_CONSTANT_MEMORY)
endif()

target_compile_definitions(r2c_c2r_lto_nvrtc_callback_example PRIVATE CUDA_ARCH=${CUDA_LTO_ARCHITECTURE} CUDA_PATH=${CUDAToolkit_BIN_DIR}/.. -DSOURCE_PATH=${CMAKE_SOURCE_DIR}/src)

set_target_properties(r2c_c2r_legacy_callback_example PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                                                 CUDA_SEPARABLE_COMPILATION ON)

message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
add_custom_command(
    OUTPUT  ${CMAKE_BINARY_DIR}/r2c_c2r_lto_callback_device.fatbin ${CMAKE_BINARY_DIR}/r2c_c2r_lto_callback_device_fatbin.h
    DEPENDS ${CMAKE_SOURCE_DIR}/src/r2c_c2r_lto_callback_device.cu
    COMMAND ${CMAKE_CUDA_COMPILER} --std=c++11 ${GENCODE_FLAGS_LTO} -dc -fatbin ${CMAKE_SOURCE_DIR}/src/r2c_c2r_lto_callback_device.cu -o ${CMAKE_BINARY_DIR}/r2c_c2r_lto_callback_device.fatbin
    COMMAND ${CUDAToolkit_BIN_DIR}/bin2c --name window_callback --type longlong ${CMAKE_BINARY_DIR}/r2c_c2r_lto_callback_device.fatbin > ${CMAKE_BINARY_DIR}/r2c_c2r_lto_callback_device_fatbin.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(r2c_c2r_lto_callback_example PRIVATE CUDA::cufft CUDA::cudart CUDA::nvJitLink)
target_link_libraries(r2c_c2r_lto_nvrtc_callback_example PRIVATE CUDA::cufft CUDA::cudart CUDA::nvJitLink CUDA::nvrtc)
target_link_libraries(r2c_c2r_legacy_callback_example PRIVATE CUDA::cufft_static CUDA::cudart CUDA::culibos)