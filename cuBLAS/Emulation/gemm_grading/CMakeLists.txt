# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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

