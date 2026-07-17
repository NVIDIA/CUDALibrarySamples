# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.22) # Version available with Ubuntu 22.04
  project(nvcomp-benchmarks LANGUAGES CUDA CXX)

  message(STATUS "Building standalone nvCOMP benchmarks.")

  option(
    SKIP_LINKING_WITH_STATIC_NVCOMP
    "By default, the executables are compiled once with the static and once with the dynamic nvCOMP library.\
     If ON, don't create executables that are linked against the static nvCOMP library."
    OFF)

  option(
    SKIP_LINKING_WITH_DYNAMIC_NVCOMP
    "By default, the executables are compiled once with the static and once with the dynamic nvCOMP library.\
     If ON, don't create executables that are linked against the dynamic nvCOMP library."
    OFF)

  set(NVCOMP_LINK_LIBS "")
  if(NOT SKIP_LINKING_WITH_STATIC_NVCOMP)
    list(APPEND NVCOMP_LINK_LIBS "static")
  endif()
  if(NOT SKIP_LINKING_WITH_DYNAMIC_NVCOMP)
    list(APPEND NVCOMP_LINK_LIBS "dynamic")
  endif()

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

  find_package(nvcomp REQUIRED)

  # Bootstrap CPM for standalone builds
  if(NOT COMMAND cpmaddpackage)
    file(
      DOWNLOAD
      https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.3/CPM.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
      EXPECTED_HASH
        SHA256=cc155ce02e7945e7b8967ddfaff0b050e958a723ef7aad3766d368940cb15494)
    include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
  endif()
else()
  message(STATUS "Building benchmarks as part of the nvCOMP library build.")
endif()

# NVBench and its bundled RAPIDS.cmake declare cmake_minimum_required(3.30.4),
# but that ceiling comes from rapids-cmake - not from any 3.30-era feature
# NVBench itself uses. Use rapids-cmake 24.12 in order to lower the minimum
# required version.
include(FetchContent)
FetchContent_GetProperties(rapids-cmake)
if(NOT rapids-cmake_POPULATED)
  # Pinned to the branch-24.12 HEAD commit so the download is reproducible
  # rather than tracking a moving branch ref.
  FetchContent_Declare(
    rapids-cmake
    URL https://github.com/rapidsai/rapids-cmake/archive/cb1a30c8f12923fe1f56b1948cb8b26d5676f6b9.zip
  )
  FetchContent_Populate(rapids-cmake)
  execute_process(
    COMMAND
      ${CMAKE_COMMAND} -DNVCOMP_CMAKE_VERSION=${CMAKE_VERSION}
      -DNVCOMP_CMAKE_MIN_VERSION=${CMAKE_MINIMUM_REQUIRED_VERSION} -P
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_nvbench_cmake_version.cmake
    WORKING_DIRECTORY ${rapids-cmake_SOURCE_DIR})
  add_subdirectory(${rapids-cmake_SOURCE_DIR} ${rapids-cmake_BINARY_DIR}
                   EXCLUDE_FROM_ALL)
endif()

# Disable CUPTI when it is known to cause problems: - Cross-compile toolchains
# (e.g. arm-sbsa) find the host's x86_64 libcupti.so instead of a target-arch
# one, causing a linker "file in wrong format" error. - On Windows, NVBench's
# imported nvbench::cupti target is missing IMPORTED_IMPLIB, which the Visual
# Studio generator requires.
if(NOT DEFINED NVBench_ENABLE_CUPTI)
  if(CMAKE_CROSSCOMPILING OR WIN32)
    set(NVBench_ENABLE_CUPTI OFF)
  else()
    set(NVBench_ENABLE_CUPTI ON)
  endif()
endif()

# Without this benchmark executables result in a dependency on libcupti.so
if(NVBench_ENABLE_CUPTI AND NOT WIN32)
  find_package(CUDAToolkit REQUIRED)
  find_library(
    NVCOMP_CUPTI_STATIC_LIBRARY
    NAMES cupti_static
    HINTS "${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/lib64"
          "${CUDAToolkit_LIBRARY_DIR}")
  if(NVCOMP_CUPTI_STATIC_LIBRARY)
    set(NVBench_CUPTI_LIBRARY
        "${NVCOMP_CUPTI_STATIC_LIBRARY}"
        CACHE FILEPATH "CUPTI library NVBench links against" FORCE)
    message(
      STATUS
        "nvCOMP benchmarks: linking CUPTI statically (${NVCOMP_CUPTI_STATIC_LIBRARY})"
    )
  else()
    set(NVBench_ENABLE_CUPTI OFF)
    message(
      STATUS
        "nvCOMP benchmarks: libcupti_static.a not found; disabling CUPTI in NVBench."
    )
  endif()
endif()

# NVBench's /Wall (MSVC) combined with nvcomp's /WX and --Werror=all-warnings
# promotes CUDA SDK header warnings (C4668 in host_config.h) to fatal errors.
# Temporarily strip warnings-as-errors for the NVBench scope.
set(_nvcomp_cxx_flags "${CMAKE_CXX_FLAGS}")
set(_nvcomp_c_flags "${CMAKE_C_FLAGS}")
set(_nvcomp_cuda_flags "${CMAKE_CUDA_FLAGS}")
if(MSVC)
  string(REPLACE "/WX" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "/WX" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
else()
  string(APPEND CMAKE_CXX_FLAGS " -Wno-effc++ -Wno-error")
endif()
string(REPLACE "--Werror=all-warnings" "" CMAKE_CUDA_FLAGS
               "${CMAKE_CUDA_FLAGS}")

# Build NVBench for nvCOMP's GPU_ARCHS. Set in scope rather than via CPM
# OPTIONS, whose "KEY VALUE" parsing splits GPU_ARCHS' ';'.
set(CMAKE_CUDA_ARCHITECTURES "${GPU_ARCHS}")

cpmaddpackage(
  NAME
  NVBench
  GITHUB_REPOSITORY
  NVIDIA/nvbench
  GIT_TAG
  4a33a61591666f48cb5fd714d00d7d2df5977ff1
  EXCLUDE_FROM_ALL
  YES
  PATCH_COMMAND
  ${CMAKE_COMMAND}
  -DNVCOMP_CMAKE_VERSION=${CMAKE_VERSION}
  -DNVCOMP_CMAKE_MIN_VERSION=${CMAKE_MINIMUM_REQUIRED_VERSION}
  -P
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch_nvbench_cmake_version.cmake
  OPTIONS
  # Static lib avoids runtime library path issues: nvbench.dll not found on
  # Windows (DLL built into lib/, executables in bin/), and libnvbench.so
  # requiring GLIBC_2.29 on arm-sbsa hosts that only have GLIBC_2.17.
  "BUILD_SHARED_LIBS OFF"
  "CMAKE_CXX_VISIBILITY_PRESET default"
  "CMAKE_CUDA_VISIBILITY_PRESET default"
  "CMAKE_VISIBILITY_INLINES_HIDDEN OFF"
  "NVBench_ENABLE_CUPTI ${NVBench_ENABLE_CUPTI}")

set(CMAKE_CXX_FLAGS "${_nvcomp_cxx_flags}")
set(CMAKE_C_FLAGS "${_nvcomp_c_flags}")
set(CMAKE_CUDA_FLAGS "${_nvcomp_cuda_flags}")

# NVBench requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

file(GLOB BENCHMARK_SOURCES *.cpp *.cu)

function(add_benchmark BENCHMARK_SOURCE NVCOMP_LIB_TYPE)
  set(multiValueArgs LINK_LIBRARIES)
  cmake_parse_arguments(PARSE_ARGV 2 OPTIONAL "" "" "${multiValueArgs}")

  get_filename_component(BARE_NAME "${BENCHMARK_SOURCE}" NAME_WE)

  if(NVCOMP_LIB_TYPE STREQUAL "static")
    set(TARGET_NAME ${BARE_NAME}_static)
    set(NVCOMP_LIB "nvcomp::nvcomp_static")
    set(NVCOMP_CPU_LIB "nvcomp::nvcomp_cpu_static")
  else()
    set(TARGET_NAME ${BARE_NAME}_dynamic)
    set(NVCOMP_LIB "nvcomp::nvcomp")
    set(NVCOMP_CPU_LIB "nvcomp::nvcomp_cpu")
  endif()

  add_executable(${TARGET_NAME} ${BENCHMARK_SOURCE})

  set_target_properties(
    ${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${GPU_ARCHS}" OUTPUT_NAME
                                                                ${BARE_NAME})

  # MSVC creates .exp, .lib and .pdb files for executables. They need to be
  # stored in different directories for the dynamic and static builds.
  if(NVCOMP_LIB_TYPE STREQUAL "static")
    set_target_properties(
      ${TARGET_NAME}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<0:>"
                 ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<0:>"
                 LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<0:>")
    set(INSTALL_DESTINATION bin)
    target_compile_definitions(${TARGET_NAME} PRIVATE "NVCOMP_STATIC_DEFINE")
  else()
    set_target_properties(
      ${TARGET_NAME}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                 "${CMAKE_BINARY_DIR}/dynamic_bin/$<0:>"
                 ARCHIVE_OUTPUT_DIRECTORY
                 "${CMAKE_BINARY_DIR}/dynamic_lib/$<0:>"
                 LIBRARY_OUTPUT_DIRECTORY
                 "${CMAKE_BINARY_DIR}/dynamic_lib/$<0:>")
    set(INSTALL_DESTINATION dynamic_bin)
  endif()

  target_link_libraries(
    ${TARGET_NAME} PRIVATE ${NVCOMP_LIB} ${NVCOMP_CPU_LIB} CUDA::cudart_static
                           CUDA::cuda_driver ${OPTIONAL_LINK_LIBRARIES})

  install(
    TARGETS ${TARGET_NAME}
    DESTINATION ${INSTALL_DESTINATION}
    COMPONENT tests)
endfunction()

set(NVBENCH_BENCHMARKS
    benchmark_ans_chunked
    benchmark_bitcomp_chunked
    benchmark_bitcomp_lossy
    benchmark_cascaded_chunked
    benchmark_deflate_chunked
    benchmark_gdeflate_chunked
    benchmark_gzip_chunked
    benchmark_lz4_chunked
    benchmark_snappy_chunked
    benchmark_zstd_chunked)

foreach(NVCOMP_LIB_TYPE ${NVCOMP_LINK_LIBS})
  foreach(BENCHMARK_SOURCE ${BENCHMARK_SOURCES})
    get_filename_component(BARE_NAME "${BENCHMARK_SOURCE}" NAME_WE)
    if(BARE_NAME STREQUAL "benchmark_crc32")
      if(TARGET CUDA::curand_static)
        add_benchmark(${BENCHMARK_SOURCE} ${NVCOMP_LIB_TYPE} LINK_LIBRARIES
                      CUDA::curand_static)
      else()
        message(
          STATUS "Skipping CRC32 benchmark, CUDA::curand_static not found")
      endif()
    elseif(BARE_NAME IN_LIST NVBENCH_BENCHMARKS)
      add_benchmark(${BENCHMARK_SOURCE} ${NVCOMP_LIB_TYPE} LINK_LIBRARIES
                    nvbench::nvbench)
    else()
      add_benchmark(${BENCHMARK_SOURCE} ${NVCOMP_LIB_TYPE})
    endif()
  endforeach(BENCHMARK_SOURCE)
endforeach()
