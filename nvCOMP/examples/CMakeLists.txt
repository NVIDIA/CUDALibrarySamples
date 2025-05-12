# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.18)
  project (nvcomp-examples LANGUAGES CUDA CXX)

  message(STATUS "Building standalone nvCOMP examples.")

  # Options
  option(BUILD_GDEFLATE_CPU "Build the GDeflate CPU example" ON)
  option(BUILD_STATIC "Use the static build of nvCOMP" OFF)
  option(BUILD_GDS_EXAMPLE "Build nvCOMP with the GDS example" OFF)

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
  message(STATUS "Building examples as part of the nvCOMP library build.")

  # Options
  option(BUILD_GDS_EXAMPLE "Build nvCOMP with the GDS example" OFF)

  # Compilation settings
  set(NVCOMP_NAMESPACE "")
endif()

# Making the compilation consistent between standalone and library build modes
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

if (BUILD_STATIC)
  set(NVCOMP_LIBRARY_NAME "${NVCOMP_NAMESPACE}nvcomp_static")
  set(NVCOMP_CPU_LIBRARY_NAME "${NVCOMP_NAMESPACE}nvcomp_cpu_static")
else()
  set(NVCOMP_LIBRARY_NAME "${NVCOMP_NAMESPACE}nvcomp")
  set(NVCOMP_CPU_LIBRARY_NAME "${NVCOMP_NAMESPACE}nvcomp_cpu")
endif()

# Add gdeflate CPU examples
if(BUILD_GDEFLATE_CPU)
  file(GLOB GDEFLATE_CPU_SOURCES *gdeflate_cpu*.cpp *gdeflate_cpu*.cu)
  foreach(EXAMPLE_SOURCE ${GDEFLATE_CPU_SOURCES})
    # cut off suffixes
    string(REPLACE ".cpp" "" EXAMPLE_NAME ${EXAMPLE_SOURCE})
    string(REPLACE ".cu" "" EXAMPLE_NAME ${EXAMPLE_NAME})
    get_filename_component(BARE_NAME ${EXAMPLE_NAME} NAME)
    add_executable(${BARE_NAME} ${EXAMPLE_SOURCE})
    set_property(TARGET ${BARE_NAME} PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
    target_link_libraries(${BARE_NAME} PRIVATE ${NVCOMP_LIBRARY_NAME} ${NVCOMP_CPU_LIBRARY_NAME} CUDA::cudart_static)
  endforeach(EXAMPLE_SOURCE ${GDEFLATE_CPU_SOURCES})
endif()

# Add GDS example
if (BUILD_GDS_EXAMPLE AND NOT MSVC)
  add_executable (nvcomp_gds nvcomp_gds.cu)
  set_property(TARGET nvcomp_gds PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  target_link_libraries(nvcomp_gds PRIVATE ${NVCOMP_LIBRARY_NAME} cufile CUDA::cudart_static)
endif()

# Add LZ4 CPU examples
find_path(LZ4_INCLUDE_DIR NAMES lz4.h)
find_library(LZ4_LIBRARY NAMES lz4)
if (LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
  # lz4 CPU example requires lz4 libraries
  add_executable(lz4_cpu_compression lz4_cpu_compression.cu)
  set_property(TARGET lz4_cpu_compression PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  target_link_libraries(lz4_cpu_compression PRIVATE ${NVCOMP_LIBRARY_NAME} ${LZ4_LIBRARY} CUDA::cudart_static)
  target_include_directories(lz4_cpu_compression PRIVATE ${LZ4_INCLUDE_DIR})

  add_executable(lz4_cpu_decompression lz4_cpu_decompression.cu)
  set_property(TARGET lz4_cpu_decompression PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  target_link_libraries(lz4_cpu_decompression PRIVATE ${NVCOMP_LIBRARY_NAME} ${LZ4_LIBRARY} CUDA::cudart_static)
  target_include_directories(lz4_cpu_decompression PRIVATE ${LZ4_INCLUDE_DIR})
else()
  message(WARNING "Skipping building LZ4 CPU example, as no LZ4 library was found.")
endif()

# Add quickstart examples
add_executable(low_level_quickstart_example low_level_quickstart_example.cpp)
target_link_libraries(low_level_quickstart_example PRIVATE ${NVCOMP_LIBRARY_NAME} CUDA::cudart_static)

add_executable(high_level_quickstart_example high_level_quickstart_example.cpp)
target_link_libraries(high_level_quickstart_example PRIVATE ${NVCOMP_LIBRARY_NAME} CUDA::cudart_static)

# Add Deflate CPU examples
find_package (ZLIB)
find_path(LIBDEFLATE_INCLUDE_DIR NAMES libdeflate.h)
find_library(LIBDEFLATE_LIBRARY NAMES deflate)
if (ZLIB_FOUND AND LIBDEFLATE_INCLUDE_DIR AND LIBDEFLATE_LIBRARY)
  add_executable(deflate_cpu_compression deflate_cpu_compression.cu)
  set_property(TARGET deflate_cpu_compression PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  target_link_libraries(deflate_cpu_compression PRIVATE ${NVCOMP_LIBRARY_NAME} ${LIBDEFLATE_LIBRARY} ZLIB::ZLIB CUDA::cudart_static)
  target_include_directories(deflate_cpu_compression PRIVATE ${LIBDEFLATE_INCLUDE_DIR})

  add_executable(deflate_cpu_decompression deflate_cpu_decompression.cu)
  set_property(TARGET deflate_cpu_decompression PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  target_link_libraries(deflate_cpu_decompression PRIVATE ${NVCOMP_LIBRARY_NAME} ${LIBDEFLATE_LIBRARY} ZLIB::ZLIB CUDA::cudart_static)
  target_include_directories(deflate_cpu_decompression PRIVATE ${LIBDEFLATE_INCLUDE_DIR})
else()
  message(WARNING "Skipping building Deflate CPU example, as the zlib and/or libdeflate libraries were not found.")
endif()

# Add GZIP example
if (ZLIB_FOUND)
  add_executable(gzip_gpu_decompression gzip_gpu_decompression.cu)
  set_property(TARGET gzip_gpu_decompression PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  target_link_libraries(gzip_gpu_decompression PRIVATE ${NVCOMP_LIBRARY_NAME} ZLIB::ZLIB CUDA::cudart_static)
else()
  message(WARNING "Skipping building Gzip GPU decompression example, as the zlib library was not found.")
endif()

# Add Snappy examples
find_package(Snappy)
if (Snappy_FOUND)
  add_executable(snappy_cpu_compression snappy_cpu_compression.cu)
  set_property(TARGET snappy_cpu_compression PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  target_link_libraries(snappy_cpu_compression PRIVATE ${NVCOMP_LIBRARY_NAME} Snappy::snappy CUDA::cudart_static)

  add_executable(snappy_cpu_decompression snappy_cpu_decompression.cu)
  set_property(TARGET snappy_cpu_decompression PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  target_link_libraries(snappy_cpu_decompression PRIVATE ${NVCOMP_LIBRARY_NAME} Snappy::snappy CUDA::cudart_static)
else()
  message(WARNING "Skipping building Snappy CPU example, as the Snappy library was not found.")
endif()

# Add Zstd examples
find_path(ZSTD_INCLUDE_DIR NAMES zstd.h)
find_library(ZSTD_LIBRARY NAMES zstd)
if (ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
  add_executable(zstd_cpu_compression zstd_cpu_compression.cu)
  set_property(TARGET zstd_cpu_compression PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  target_include_directories(zstd_cpu_compression PRIVATE ${ZSTD_INCLUDE_DIR})
  target_link_libraries(zstd_cpu_compression PRIVATE ${NVCOMP_LIBRARY_NAME} ${ZSTD_LIBRARY} CUDA::cudart_static)

  add_executable(zstd_cpu_decompression zstd_cpu_decompression.cu)
  set_property(TARGET zstd_cpu_decompression PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  target_include_directories(zstd_cpu_decompression PRIVATE ${ZSTD_INCLUDE_DIR})
  target_link_libraries(zstd_cpu_decompression PRIVATE ${NVCOMP_LIBRARY_NAME} ${ZSTD_LIBRARY} CUDA::cudart_static)
else()
  message(WARNING "Skipping building Zstandard CPU example, as the Zstandard library was not found.")
endif()
