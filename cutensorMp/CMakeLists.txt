# SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#

# ---[ Check cmake version.
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# ---[ Project specification.
project(cutensormp_example LANGUAGES C CXX CUDA)

# ---[ Platform support checks.
if(WIN32)
    message(FATAL_ERROR "Windows platform is not supported for cutensormp_example.")
endif()

# ---[ Include custom functions.
include(cmake/functions.cmake)

# Make custom find modules (e.g., FindNCCL.cmake) discoverable
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ##########################################
# cuTENSOR_example build mode
# ##########################################

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# ##########################################
# cuTENSOR_example dependencies
# ##########################################
if(DEFINED ENV{NCCL_HOME})
  set(NCCL_HOME $ENV{NCCL_HOME})
endif()

find_dependency_with_fallback("MPI" "MPI_HOME" "mpi.h" "mpi;mpi_cxx;mpicxx" "MPI_INCLUDE_DIR" "MPI_C_LIBRARY;MPI_CXX_LIBRARY")

find_package(NCCL REQUIRED)
set(NCCL_INCLUDE_DIR ${NCCL_INCLUDE_DIR} CACHE INTERNAL "NCCL include directory")
set(NCCL_LIBRARY ${NCCL_LIBRARY} CACHE INTERNAL "NCCL library path")

find_package(CUDA 12.9 REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

if(NOT DEFINED ENV{CUTENSOR_ROOT} AND NOT DEFINED CUTENSOR_ROOT)
  message(FATAL_ERROR "CUTENSOR_ROOT not set!")
else()
  if(DEFINED ENV{CUTENSOR_ROOT})
    set(CUTENSOR_ROOT "$ENV{CUTENSOR_ROOT}")
  endif()
  message("-- Looking for cuTENSOR in ${CUTENSOR_ROOT}")
  if(NOT EXISTS ${CUTENSOR_ROOT})
    message(FATAL_ERROR "Cannot find CUTENSOR_ROOT")
  endif()
endif()

if((NOT TARGET cutensor) AND (NOT TARGET cutensorMp))
  add_library(cutensorMp SHARED IMPORTED)
  add_library(cutensor SHARED IMPORTED)

  set(CUTENSORMP_LIBRARY_NAME "libcutensorMp.so")
  set(CUTENSOR_LIBRARY_NAME "libcutensor.so")

  set(LIB_DIR "/lib/${CUDA_VERSION_MAJOR}")
  if(NOT EXISTS "${CUTENSOR_ROOT}${LIB_DIR}")
      set(LIB_DIR "/lib")
  endif()
  set_target_properties(cutensorMp PROPERTIES
    IMPORTED_LOCATION "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSORMP_LIBRARY_NAME}"
    IMPORTED_IMPLIB "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSORMP_LIBRARY_DEF}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_ROOT}/include")
  set_target_properties(cutensor PROPERTIES
    IMPORTED_LOCATION "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_NAME}"
    IMPORTED_IMPLIB "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_DEF}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_ROOT}/include")
endif()

# Installation directories
if(NOT DEFINED CUTENSORMP_EXAMPLE_BINARY_INSTALL_DIR)
  message(WARNING "CUTENSORMP_EXAMPLE_BINARY_INSTALL_DIR not set, setting to cutensormp_example/bin")
  set(CUTENSORMP_EXAMPLE_BINARY_INSTALL_DIR "cutensormp_example/bin")
endif()

# ##########################################
# cuTENSOR_example building flags
# ##########################################

# Global CXX/CUDA flags

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ##########################################
# cuTENSOR_example target
# ##########################################

function(add_cutensormp_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_include_directories(${EXAMPLE_TARGET}
        PUBLIC
            ${CUDA_INCLUDE_DIRS}
            ${CUTENSOR_ROOT}/include
            ${CUTENSOR_ROOT}/cutensorMp/include
            ${MPI_INCLUDE_DIR}
            ${NCCL_INCLUDE_DIR}
    )
    target_link_libraries(${EXAMPLE_TARGET}
        PUBLIC
            ${MPI_C_LIBRARY}
            ${MPI_CXX_LIBRARY}
            ${NCCL_LIBRARY}
            cudart
            cublasLt
            cutensorMp
            cutensor
    )
    set_target_properties(${EXAMPLE_TARGET} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
    )

    # Install example
    install(
        TARGETS ${EXAMPLE_TARGET}
        RUNTIME
        DESTINATION ${CUTENSOR_EXAMPLE_BINARY_INSTALL_DIR}
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
    )

    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

add_custom_target(cutensormp_examples)
add_cutensormp_example(cutensormp_examples "cuTENSORMp.example.cutensorMp_contraction" cutensorMp_contraction.cu)

# ##########################################
# cuTENSOR_example directories
# ##########################################

# By default put binaries in build/bin (pre-install)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ##########################################
# Install examples
# ##########################################

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
ENDIF()
