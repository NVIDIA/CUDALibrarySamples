# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.18)
  project (nvcomp-benchmarks LANGUAGES CUDA CXX)

  message(STATUS "Building standalone nvCOMP benchmarks.")

  # Options
  option(BUILD_STATIC "Use the static build of nvCOMP" OFF)

  # Compilation settings
  set(NVCOMP_NAMESPACE "nvcomp::")
  set(GPU_ARCHS "70;75;80;86;89;90")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.8")
    set(GPU_ARCHS ${GPU_ARCHS} "100" "120")
  endif()
  add_compile_definitions("THRUST_CUB_WRAPPED_NAMESPACE=nvcomp")
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
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
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
