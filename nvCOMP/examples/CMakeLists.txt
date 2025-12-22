# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  project (nvcomp-examples LANGUAGES CUDA CXX)

  message(STATUS "Building standalone nvCOMP examples.")

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
  message(STATUS "Building examples as part of the nvCOMP library build.")

  # Compilation settings
  set(NVCOMP_NAMESPACE "")
endif()

# Options
option(BUILD_GDS_EXAMPLE "Build nvCOMP with the GDS example" OFF)

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

function(add_example EXAMPLE_SOURCE NVCOMP_LIB_TYPE)
  set(multiValueArgs LINK_LIBRARIES INCLUDE_DIRECTORIES)
  cmake_parse_arguments(PARSE_ARGV 2 OPTIONAL "" "" "${multiValueArgs}")

  get_filename_component(BARE_NAME "${EXAMPLE_SOURCE}" NAME_WE)

  if (NVCOMP_LIB_TYPE STREQUAL "static")
    set(TARGET_NAME ${BARE_NAME}_static)
    set(NVCOMP_LIB "${NVCOMP_NAMESPACE}nvcomp_static")
  else()
    set(TARGET_NAME ${BARE_NAME}_dynamic)
    set(NVCOMP_LIB "${NVCOMP_NAMESPACE}nvcomp")
  endif()

  add_executable(${TARGET_NAME} ${EXAMPLE_SOURCE})

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
    target_compile_definitions(${TARGET_NAME} PRIVATE "NVCOMP_STATIC_DEFINE")
    set(INSTALL_DESTINATION bin)
  else()
    set_target_properties(${TARGET_NAME} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dynamic_bin/$<0:>"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dynamic_lib/$<0:>"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dynamic_lib/$<0:>"
    )
    set(INSTALL_DESTINATION dynamic_bin)
  endif()

  target_include_directories(${TARGET_NAME} PRIVATE ${OPTIONAL_INCLUDE_DIRECTORIES})

  target_link_libraries(${TARGET_NAME} PRIVATE ${NVCOMP_LIB} CUDA::cudart_static ${OPTIONAL_LINK_LIBRARIES})

  install(TARGETS ${TARGET_NAME} DESTINATION ${INSTALL_DESTINATION} COMPONENT tests)
endfunction()

foreach(NVCOMP_LIB_TYPE ${NVCOMP_LINK_LIBS})
  if (NVCOMP_LIB_TYPE STREQUAL "static")
    set(NVCOMP_SUFFIX "_static")
  else()
    set(NVCOMP_SUFFIX "")
  endif()
  # Add gdeflate CPU examples
  foreach(EXAMPLE_SOURCE gdeflate_cpu_decompression.cu gdeflate_cpu_compression.cu)
    add_example(${EXAMPLE_SOURCE} ${NVCOMP_LIB_TYPE}
      LINK_LIBRARIES "${NVCOMP_NAMESPACE}nvcomp_cpu${NVCOMP_SUFFIX}"
    )
  endforeach(EXAMPLE_SOURCE)

  # Add GDS example
  if (BUILD_GDS_EXAMPLE AND NOT MSVC)
    add_example(nvcomp_gds.cu ${NVCOMP_LIB_TYPE}
      LINK_LIBRARIES cufile
    )
  endif()

  # Add LZ4 CPU examples
  find_path(LZ4_INCLUDE_DIR NAMES lz4.h)
  find_library(LZ4_LIBRARY NAMES lz4)
  if (LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
    # lz4 CPU example requires lz4 libraries
    foreach(EXAMPLE_SOURCE lz4_cpu_decompression.cu lz4_cpu_compression.cu)
      add_example(${EXAMPLE_SOURCE} ${NVCOMP_LIB_TYPE}
        LINK_LIBRARIES ${LZ4_LIBRARY}
        INCLUDE_DIRECTORIES ${LZ4_INCLUDE_DIR}
      )
    endforeach()
  else()
    message(WARNING "Skipping building LZ4 CPU example, as no LZ4 library was found.")
  endif()

  # Add quickstart examples
  add_example(low_level_quickstart_example.cpp ${NVCOMP_LIB_TYPE})
  add_example(high_level_quickstart_example.cpp ${NVCOMP_LIB_TYPE})
  add_example(nvcomp_crc32.cu ${NVCOMP_LIB_TYPE})

  # Add Deflate CPU examples
  find_package (ZLIB)
  find_path(LIBDEFLATE_INCLUDE_DIR NAMES libdeflate.h)
  find_library(LIBDEFLATE_LIBRARY NAMES deflate)
  if (ZLIB_FOUND AND LIBDEFLATE_INCLUDE_DIR AND LIBDEFLATE_LIBRARY)
    foreach(EXAMPLE_SOURCE deflate_cpu_decompression.cu deflate_cpu_compression.cu)
      add_example(${EXAMPLE_SOURCE} ${NVCOMP_LIB_TYPE}
        LINK_LIBRARIES ${LIBDEFLATE_LIBRARY} ZLIB::ZLIB
        INCLUDE_DIRECTORIES ${LIBDEFLATE_INCLUDE_DIR}
      )
    endforeach()
  else()
    message(WARNING "Skipping building Deflate CPU example, as the zlib and/or libdeflate libraries were not found.")
  endif()

  # Add GZIP example
  if (ZLIB_FOUND)
    add_example(gzip_gpu_decompression.cu ${NVCOMP_LIB_TYPE}
      LINK_LIBRARIES ZLIB::ZLIB
    )
  else()
    message(WARNING "Skipping building Gzip GPU decompression example, as the zlib library was not found.")
  endif()

  # Add Snappy examples
  find_package(Snappy)
  if (Snappy_FOUND)
    foreach(EXAMPLE_SOURCE snappy_cpu_decompression.cu snappy_cpu_compression.cu)
      add_example(${EXAMPLE_SOURCE} ${NVCOMP_LIB_TYPE}
        LINK_LIBRARIES Snappy::snappy
      )
    endforeach()
  else()
    message(WARNING "Skipping building Snappy CPU example, as the Snappy library was not found.")
  endif()

  # Add Zstd examples
  find_path(ZSTD_INCLUDE_DIR NAMES zstd.h)
  find_library(ZSTD_LIBRARY NAMES zstd)
  if (ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
    foreach(EXAMPLE_SOURCE zstd_cpu_decompression.cu zstd_cpu_compression.cu)
      add_example(${EXAMPLE_SOURCE} ${NVCOMP_LIB_TYPE}
        LINK_LIBRARIES ${ZSTD_LIBRARY}
        INCLUDE_DIRECTORIES ${ZSTD_INCLUDE_DIR}
      )
    endforeach()
  else()
    message(WARNING "Skipping building Zstandard CPU example, as the Zstandard library was not found.")
  endif()

endforeach()
