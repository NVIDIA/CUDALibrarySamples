# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.18)
  project (nvcomp-benchmarks LANGUAGES CUDA CXX)

  message(STATUS "Building standalone nvCOMP benchmarks.")

  # Options
  option(BUILD_STATIC "Use the static build of nvCOMP" OFF)

  # Compilation settings
  set(NVCOMP_NAMESPACE "nvcomp::")
  set(GPU_ARCHS "75;80;86;89;90")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13")
    set(GPU_ARCHS ${GPU_ARCHS} "70")
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
    set(GPU_ARCHS ${GPU_ARCHS} "100;120")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13")
      set(GPU_ARCHS ${GPU_ARCHS} "110")
    else()
      set(GPU_ARCHS ${GPU_ARCHS} "101")
    endif()
  endif()
  if (BUILD_STATIC)
    add_compile_definitions("NVCOMP_STATIC_DEFINE")
  endif()

  find_package(nvcomp REQUIRED)
else()
  message(STATUS "Building benchmarks as part of the nvCOMP library build.")

  # Compilation settings
  set(NVCOMP_NAMESPACE "")
endif()

# Making the compilation consistent between standalone and library build modes
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13")
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CUDA_STANDARD 11)
else()
  # Note: Thrust requires C++17
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
endif()
set(CMAKE_CXX_EXTENSIONS OFF)

if (BUILD_STATIC)
  set(NVCOMP_LIBRARY_NAME "${NVCOMP_NAMESPACE}nvcomp_static")
else()
  set(NVCOMP_LIBRARY_NAME "${NVCOMP_NAMESPACE}nvcomp")
endif()

file(GLOB BENCHMARK_SOURCES *.cpp *.cu)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

foreach(BENCHMARK_SOURCE ${BENCHMARK_SOURCES})
  # cut off suffixes
  string(REPLACE ".cpp" "" BENCHMARK_NAME ${BENCHMARK_SOURCE})
  string(REPLACE ".cu" "" BENCHMARK_NAME ${BENCHMARK_NAME})
  get_filename_component(BARE_NAME ${BENCHMARK_NAME} NAME)
  if (BARE_NAME STREQUAL "benchmark_crc32")
    # Handle CRC32 benchmark separately
    continue()
  endif()
  add_executable(${BARE_NAME} ${BENCHMARK_SOURCE})
  set_property(TARGET ${BARE_NAME} PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  target_link_libraries(${BARE_NAME} PRIVATE ${NVCOMP_LIBRARY_NAME} CUDA::cudart_static)

  target_include_directories(${BARE_NAME} PRIVATE
      "$<BUILD_INTERFACE:${nvcomp_SOURCE_DIR}/include>")

  install(TARGETS ${BARE_NAME}
    RUNTIME DESTINATION bin
    COMPONENT tests 
    EXCLUDE_FROM_ALL)
endforeach(BENCHMARK_SOURCE ${BENCHMARK_SOURCES})

if (TARGET CUDA::curand_static)
  message(STATUS "Found CUDA::curand_static, building CRC32 benchmark")
  # CRC32 benchmark requires curand
  add_executable(benchmark_crc32 benchmark_crc32.cpp)
  set_property(TARGET benchmark_crc32 PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  target_link_libraries(benchmark_crc32 PRIVATE ${NVCOMP_LIBRARY_NAME} CUDA::cudart_static CUDA::curand_static)
  target_include_directories(benchmark_crc32 PRIVATE "$<BUILD_INTERFACE:${nvcomp_SOURCE_DIR}/include>")
  install(TARGETS benchmark_crc32 RUNTIME DESTINATION bin COMPONENT tests EXCLUDE_FROM_ALL)
else()
  message(STATUS "Skipping CRC32 benchmark, CUDA::curand_static not found")
endif()