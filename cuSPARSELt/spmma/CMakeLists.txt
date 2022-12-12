# Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
cmake_minimum_required(VERSION 3.9)

set(ROUTINE spmma)

project("${ROUTINE}_example"
        DESCRIPTION  "GPU-Accelerated Sparse Linear Algebra"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cusparse/index.html"
        LANGUAGES    CXX CUDA)

set(CMAKE_CXX_STANDARD           14)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CXX_EXTENSIONS         OFF)
set(CMAKE_CUDA_STANDARD          14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS        OFF)

string(REPLACE "/bin/nvcc" "" CUDA_TOOLKIT_PATH ${CMAKE_CUDA_COMPILER})
if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64" AND
    ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(OS_ARCH_NVRTC "sbsa-linux")
elseif (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64" AND
        ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(OS_ARCH_NVRTC "x86_64-linux")
endif()
set(NVRTC_SHARED ${CUDA_TOOLKIT_PATH}/targets/${OS_ARCH_NVRTC}/lib/libnvrtc.so)

# cusparselt_example check-env utilities
function(set_with_fallback VARIABLE FALLBACK)
  if (NOT DEFINED ${VARIABLE} OR ${VARIABLE} STREQUAL "")
    set(${VARIABLE} $ENV{${VARIABLE}} CACHE INTERNAL ${VARIABLE})
    if (${VARIABLE} STREQUAL "")
      if (NOT ${FALLBACK} STREQUAL "")
        set(${VARIABLE} $ENV{${FALLBACK}} CACHE INTERNAL ${VARIABLE})
      endif ()
    endif ()
  endif ()
endfunction()

set_with_fallback(CUSPARSELT_PATH CUSPARSELT_DIR)
if (CUSPARSELT_PATH STREQUAL "")
    message(FATAL_ERROR "Please set the environment variables CUSPARSELT_DIR to the path of the cuSPARSELt installation.")
endif ()

message(STATUS "Using CUSPARSELT_PATH = ${CUSPARSELT_DIR}")

add_executable(${ROUTINE}_example)
add_executable(${ROUTINE}_example_static)

target_sources(${ROUTINE}_example
    PUBLIC ${PROJECT_SOURCE_DIR}/${ROUTINE}_example.cpp
)

target_sources(${ROUTINE}_example_static
    PUBLIC ${PROJECT_SOURCE_DIR}/${ROUTINE}_example.cpp
)

target_include_directories(${ROUTINE}_example
    PUBLIC ${CUDA_TOOLKIT_PATH}/include
    PUBLIC ${CUSPARSELT_PATH}/include
)

target_include_directories(${ROUTINE}_example_static
    PUBLIC ${CUDA_TOOLKIT_PATH}/include
    PUBLIC ${CUSPARSELT_PATH}/include
)

target_link_directories(${ROUTINE}_example
    PUBLIC ${CUSPARSELT_PATH}/lib64
    PUBLIC ${CUDA_TOOLKIT_PATH}/lib64
)

target_link_directories(${ROUTINE}_example_static
    PUBLIC ${CUSPARSELT_PATH}/lib64
    PUBLIC ${CUDA_TOOLKIT_PATH}/lib64
)

target_link_libraries(${ROUTINE}_example
    PUBLIC cudart
    PUBLIC cusparse
    PUBLIC libcusparseLt.so
    PUBLIC ${NVRTC_SHARED}
)

target_link_libraries(${ROUTINE}_example_static
    PUBLIC cudart
    PUBLIC cusparse
    PUBLIC libcusparseLt_static.a
    PUBLIC ${NVRTC_SHARED}
    PUBLIC ${CMAKE_DL_LIBS}
)
