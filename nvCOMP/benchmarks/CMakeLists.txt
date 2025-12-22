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

  option(SKIP_LINKING_WITH_STATIC_NVCOMP
    "By default, the executables are compiled once with the static and once with the dynamic nvCOMP library.\
     If ON, don't create executables that are linked against the static nvCOMP library."
    OFF
  )

  option(SKIP_LINKING_WITH_DYNAMIC_NVCOMP
    "By default, the executables are compiled once with the static and once with the dynamic nvCOMP library.\
     If ON, don't create executables that are linked against the dynamic nvCOMP library."
    OFF
  )

  set(NVCOMP_LINK_LIBS "")
  if (NOT SKIP_LINKING_WITH_STATIC_NVCOMP)
    list(APPEND NVCOMP_LINK_LIBS "static")
  endif()
  if (NOT SKIP_LINKING_WITH_DYNAMIC_NVCOMP)
    list(APPEND NVCOMP_LINK_LIBS "dynamic")
  endif()

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

  find_package(nvcomp REQUIRED)
else()
  message(STATUS "Building benchmarks as part of the nvCOMP library build.")

  # Compilation settings
  set(NVCOMP_NAMESPACE "")
endif()

# Making the compilation consistent between standalone and library build modes
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13")
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CUDA_STANDARD 14)
else()
  # Note: Thrust requires C++17
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

file(GLOB BENCHMARK_SOURCES *.cpp *.cu)

function(add_benchmark BENCHMARK_SOURCE NVCOMP_LIB_TYPE)
  set(multiValueArgs LINK_LIBRARIES)
  cmake_parse_arguments(PARSE_ARGV 2 OPTIONAL "" "" "${multiValueArgs}")

  get_filename_component(BARE_NAME "${BENCHMARK_SOURCE}" NAME_WE)

  if (NVCOMP_LIB_TYPE STREQUAL "static")
    set(TARGET_NAME ${BARE_NAME}_static)
    set(NVCOMP_LIB "${NVCOMP_NAMESPACE}nvcomp_static")
  else()
    set(TARGET_NAME ${BARE_NAME}_dynamic)
    set(NVCOMP_LIB "${NVCOMP_NAMESPACE}nvcomp")
  endif()

  add_executable(${TARGET_NAME} ${BENCHMARK_SOURCE})

  set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_ARCHITECTURES "${GPU_ARCHS}"
    OUTPUT_NAME ${BARE_NAME}
  )

  # MSVC creates .exp, .lib and .pdb files for executables.
  # They need to be stored in different directories
  # for the dynamic and static builds.
  if (NVCOMP_LIB_TYPE STREQUAL "static")
    set_target_properties(${TARGET_NAME} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<0:>"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<0:>"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<0:>"
    )
    set(INSTALL_DESTINATION bin)
    target_compile_definitions(${TARGET_NAME} PRIVATE "NVCOMP_STATIC_DEFINE")
  else()
    set_target_properties(${TARGET_NAME} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dynamic_bin/$<0:>"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dynamic_lib/$<0:>"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dynamic_lib/$<0:>"
    )
    set(INSTALL_DESTINATION dynamic_bin)
  endif()

  target_link_libraries(${TARGET_NAME} PRIVATE ${NVCOMP_LIB} CUDA::cudart_static ${OPTIONAL_LINK_LIBRARIES})

  install(TARGETS ${TARGET_NAME} DESTINATION ${INSTALL_DESTINATION} COMPONENT tests)
endfunction()

foreach(NVCOMP_LIB_TYPE ${NVCOMP_LINK_LIBS})
  foreach(BENCHMARK_SOURCE ${BENCHMARK_SOURCES})
    get_filename_component(BARE_NAME "${BENCHMARK_SOURCE}" NAME_WE)
    if (BARE_NAME STREQUAL "benchmark_crc32")
      if (TARGET CUDA::curand_static)
        add_benchmark(${BENCHMARK_SOURCE} ${NVCOMP_LIB_TYPE}
          LINK_LIBRARIES CUDA::curand_static
        )
      else()
        message(STATUS "Skipping CRC32 benchmark, CUDA::curand_static not found")
      endif()
    else()
      add_benchmark(${BENCHMARK_SOURCE} ${NVCOMP_LIB_TYPE})
    endif()
  endforeach(BENCHMARK_SOURCE)
endforeach()
