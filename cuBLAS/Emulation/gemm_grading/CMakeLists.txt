# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# ---[ Check cmake version.
cmake_minimum_required(VERSION 3.18.0 FATAL_ERROR)

# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------
project(jdemmel_blas3_tests LANGUAGES CXX)

include(GNUInstallDirs)

# -----------------------------------------------------------------------------
# building flags
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(ENABLE_DEBUG_PRINTS "Enable DEBUG_PRINTS compile definition" OFF)
if(ENABLE_DEBUG_PRINTS)
  add_compile_definitions(DEBUG_PRINTS)
  message(STATUS "Debug prints: ENABLED")
else()
  message(STATUS "Debug prints: DISABLED")
endif()

# ----------------------------------------------------------------------------
# Emulated GEMM requires cuBLAS 13.0u2 or newer
# ----------------------------------------------------------------------------
find_package(CUDAToolkit 13.0.2 REQUIRED COMPONENTS cublas cudart)

# ----------------------------------------------------------------------------
# BLAS discovery
# ----------------------------------------------------------------------------
find_package(BLAS REQUIRED)

# ----------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------
add_executable(test_blas3 test_blas3.cpp gemms.cpp)
target_link_libraries(test_blas3 PRIVATE BLAS::BLAS CUDA::cublas CUDA::cudart)
target_include_directories(test_blas3 PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(test_blas3 PRIVATE -O0)

add_executable(grade_blas3 grade_blas3.cpp gemms.cpp)
target_link_libraries(grade_blas3 PRIVATE BLAS::BLAS CUDA::cublas CUDA::cudart)
target_include_directories(grade_blas3 PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(grade_blas3 PRIVATE -O0)


# Pre-install build location
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Install location (under prefix)
set(CUBLAS_EXAMPLES_BINARY_INSTALL_DIR "cublas_examples/bin")

# ----------------------------------------------------------------------------
# Install
# ----------------------------------------------------------------------------
install(TARGETS test_blas3 grade_blas3
  RUNTIME DESTINATION ${CUBLAS_EXAMPLES_BINARY_INSTALL_DIR}
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
)

