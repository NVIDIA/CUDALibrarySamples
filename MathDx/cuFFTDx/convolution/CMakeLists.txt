# Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.

# NOTICE TO LICENSEE:

# This source code and/or documentation ("Licensed Deliverables") are subject to
# NVIDIA intellectual property rights under U.S. and international Copyright
# laws.

# These Licensed Deliverables contained herein is PROPRIETARY and CONFIDENTIAL
# to NVIDIA and is being provided under the terms and conditions of a form of
# NVIDIA software license agreement by and between NVIDIA and Licensee ("License
# Agreement") or electronically accepted by Licensee.  Notwithstanding any terms
# or conditions to the contrary in the License Agreement, reproduction or
# disclosure of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.

# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
# AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THESE
# LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS
# OR IMPLIED WARRANTY OF ANY KIND. NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD
# TO THESE LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
# AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THESE LICENSED DELIVERABLES.

# U.S. Government End Users.  These Licensed Deliverables are a "commercial
# item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting of
# "commercial computer software" and "commercial computer software
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) and is
# provided to the U.S. Government only as a commercial end item.  Consistent
# with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),
# all U.S. Government End Users acquire the Licensed Deliverables with only
# those rights set forth herein.

# Any use of the Licensed Deliverables in individual and commercial software
# must include, in the user documentation and internal comments to the code, the
# above Disclaimer and U.S. Government End Users Notice.
cmake_minimum_required(VERSION 3.18.0)

# All supported CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70-real 72-real 75-real 80-real 86-real 86-virtual)
endif()

# cuFFTDx Examples project
project(cufftdx_examples LANGUAGES CXX CUDA)

# Find cuFFTDx from mathDx package
find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG)

# Global CXX/CUDA flags
if(NOT MSVC)
    set(CUFFTDX_CUDA_CXX_FLAGS "${CUFFTDX_CUDA_CXX_FLAGS} -Wall -Wextra -Werror")
else()
    message(FATAL_ERROR "MSVC not supported yet")
endif()

# Global CXX flags/options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUFFTDX_CUDA_CXX_FLAGS}")

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${CUFFTDX_CUDA_CXX_FLAGS}\"")

# Enable testing (ctest)
enable_testing()

add_library(cufftdx_cuda_architectures INTERFACE)
foreach(CUDA_ARCHITECTURE ${CMAKE_CUDA_ARCHITECTURES})
    string(REPLACE "-" ";" CUDA_ARCHITECTURE_LIST ${CUDA_ARCHITECTURE})
    list(GET CUDA_ARCHITECTURE_LIST 0 ARCH)
    target_compile_definitions(cufftdx_cuda_architectures INTERFACE SM_${ARCH})
endforeach()

# ###############################################################
# add_cufftdx_example
# ###############################################################
function(add_cufftdx_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE cufftdx_cuda_architectures mathdx::cufftdx)
    target_compile_options(${EXAMPLE_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>")
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
    include_directories(${CMAKE_SOURCE_DIR}/../utils)
endfunction()

add_custom_target(cufftdx_examples)

add_cufftdx_example(cufftdx_examples "mathDx.cuFFTDx.convolution" convolution.cu)
