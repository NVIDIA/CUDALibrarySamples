# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#
# csreigs (only double complex)
#
set(CUSOLVER_ETI_VALUE_TYPES dcomplex)
set(CUSOLVER_DRIVERS csreigs2cuDSS)
build_driver()

#
# Copy test matrices to the build directory
#
configure_file(test_complex.mtx test_complex.mtx)
configure_file(test_real.mtx test_real.mtx)