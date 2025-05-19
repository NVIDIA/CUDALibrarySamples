#
# Copyright 2023-2025 NVIDIA Corporation.  All rights reserved.
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
#
cmake_minimum_required(VERSION 3.13)

project("cuSOLVERSp2cuDSS"
  DESCRIPTION  "A transition example from cuSOLVERSp/Rf to cuDSS"
  LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD          11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS        OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -O0 -g -lineinfo")

option(CUDSS_ROOT "Path to cudss install path") 
set(CUDSS_INCLUDE_DIRS "${CUDSS_ROOT}/include")
set(CUDSS_LIBRARY_DIRS "${CUDSS_ROOT}/lib")

#
# build_driver
#
function(build_driver)
  foreach(driver IN LISTS CUSOLVER_DRIVERS)
    foreach(eti_value IN LISTS CUSOLVER_ETI_VALUE_TYPES)
      set(exe ${driver}_${eti_value})
      set(src ${exe}.cpp)
      set_source_files_properties(${src} PROPERTIES LANGUAGE CUDA)  
      add_executable(${exe} ${src})
      
      set_property(TARGET ${exe} PROPERTY CUDA_ARCHITECTURES OFF)
      set_target_properties(${exe} PROPERTIES POSITION_INDEPENDENT_CODE ON)
      
      target_include_directories(${exe}
        PUBLIC
        ${CUDA_INCLUDE_DIRS}  
        ${CUDSS_INCLUDE_DIRS}
      )
      target_link_libraries(${exe}
        PUBLIC
        cusolver
        cublas
        cusparse
        -L${CUDSS_LIBRARY_DIRS} -lcudss
      )
    endforeach()
  endforeach()
endfunction()

#
# cuSolverSp2cuDSS
#
set(CUSOLVER_ETI_VALUE_TYPES float double scomplex dcomplex)
set(CUSOLVER_DRIVERS cuSolverSp2cuDSS) 
build_driver()

#
# cuSolverRf2cuDSS (only double)
#
set(CUSOLVER_ETI_VALUE_TYPES double)
set(CUSOLVER_DRIVERS cuSolverRf2cuDSS) 
build_driver()

#
# csreigvsi (only double)
#
set(CUSOLVER_ETI_VALUE_TYPES double)
set(CUSOLVER_DRIVERS csreigvsi2cuDSS)
build_driver()