# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.19)

set(EXAMPLE_NAME simple_mg_mode)

project("cuDSS_${EXAMPLE_NAME}_example"
        DESCRIPTION  "cuDSS"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cudss/index.html"
        LANGUAGES    CXX CUDA)

# Align CUDA host compiler with CXX (e.g. NVHPC + nvcc).
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE STRING "CUDA host compiler" FORCE)
endif()

set(CMAKE_CUDA_STANDARD          11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS        OFF)

option(BUILD_STATIC "Building statically linked examples" OFF)

# Find CUDA toolkit (for cusolver)
find_package(CUDAToolkit REQUIRED)
# Find cuDSS
find_package(cudss 0.8.0 REQUIRED)
# CUDAToolkit may expose CUDA::cusolver (CMake 3.17+) or not in some configs; fallback to library name
if(NOT TARGET CUDA::cusolver AND NOT TARGET CUDAToolkit::cusolver)
    find_library(CUSOLVER_LIBRARY cusolver
        PATHS ${CUDAToolkit_LIBRARY_DIR} "${CUDAToolkit_LIBRARY_DIR}/${CUDAToolkit_VERSION_MAJOR}"
              ${CUDAToolkit_ROOT} "$ENV{CUDA_PATH}"
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH)
    if(CUSOLVER_LIBRARY)
        add_library(cusolver_lib_imported SHARED IMPORTED)
        set_target_properties(cusolver_lib_imported PROPERTIES IMPORTED_LOCATION "${CUSOLVER_LIBRARY}")
        set(CUSOLVER_TARGET cusolver_lib_imported)
    else()
        message(FATAL_ERROR "cusolver not found; CUDAToolkit found but CUDA::cusolver target unavailable")
    endif()
elseif(TARGET CUDA::cusolver)
    set(CUSOLVER_TARGET CUDA::cusolver)
else()
    set(CUSOLVER_TARGET CUDAToolkit::cusolver)
endif()

set_source_files_properties(${EXAMPLE_NAME}.cpp PROPERTIES LANGUAGE CUDA)

add_executable(${EXAMPLE_NAME}_example)

target_sources(${EXAMPLE_NAME}_example
    PUBLIC ${PROJECT_SOURCE_DIR}/${EXAMPLE_NAME}.cpp
)

target_include_directories(${EXAMPLE_NAME}_example PUBLIC ${cudss_INCLUDE_DIR})
target_link_directories(${EXAMPLE_NAME}_example PUBLIC ${cudss_LIBRARY_DIR})

target_compile_options(${EXAMPLE_NAME}_example PRIVATE
    "$<$<CONFIG:Debug>:-lineinfo -g>"
    "$<$<CONFIG:RelWithDebInfo>:-lineinfo -g>"
)

target_link_libraries(${EXAMPLE_NAME}_example PUBLIC
    cudss
    ${CUSOLVER_TARGET}
)

if(BUILD_STATIC)
    add_executable(${EXAMPLE_NAME}_example_static)

    target_sources(${EXAMPLE_NAME}_example_static
        PUBLIC ${PROJECT_SOURCE_DIR}/${EXAMPLE_NAME}.cpp
    )

    target_include_directories(${EXAMPLE_NAME}_example_static PUBLIC ${cudss_INCLUDE_DIR})
    target_link_directories(${EXAMPLE_NAME}_example_static PUBLIC ${cudss_LIBRARY_DIR})

    if (WIN32)
        target_link_libraries(${EXAMPLE_NAME}_example_static PUBLIC
        cudss ${CUSOLVER_TARGET}
    )
    else()
        target_link_libraries(${EXAMPLE_NAME}_example_static PUBLIC
            cudss_static ${CUSOLVER_TARGET}
        )
    endif()

endif()
