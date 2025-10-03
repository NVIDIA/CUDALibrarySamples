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

cmake_minimum_required(VERSION 3.19)

set(EXAMPLE_NAME test_threading_layer)

project("cuDSS_${EXAMPLE_NAME}_example"
        DESCRIPTION  "cuDSS"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cudss/index.html"
        LANGUAGES    CXX CUDA)

set(CMAKE_CUDA_STANDARD          11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS        OFF)

option(BUILD_STATIC           "Building statically linked examples" OFF)

# Find cuDSS
find_package(cudss 0.5.0 REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)
if (NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "For building MT examples with OpenMP, OpenMP \
package must be detectable by cmake")
endif()

# Define a list of MT backends (we only have OpenMP in this example, but this
# may be extended as cuDSS supports user-define threading backends)
set(MT_BACKENDS "")


# (optional: only if samples are run with ctest)
enable_testing()

if (WIN32)
    message(FATAL_ERROR "This examples does not support Windows!")
endif()

# Define the flavor of OpenMP picked up by find_package()
if (OpenMP_CXX_LIBRARIES OR OpenMP_C_LIBRARIES) # should work on Linux
    if (OpenMP_CXX_LIBRARIES)
        string(REGEX MATCH ".*lib([A-Za-z0-9]*omp[A-Za-z0-9]*)[.](so|dll).*" dummy ${OpenMP_CXX_LIBRARIES})
    else()
        string(REGEX MATCH ".*lib([A-Za-z0-9]*omp[A-Za-z0-9]*)[.](so|dll).*" dummy ${OpenMP_C_LIBRARIES})
    endif()
    set(omp_backend_kind ${CMAKE_MATCH_1})
else() # fallback
    set(omp_backend_kind "omp")
endif()

message(STATUS "omp_backend_kind = ${omp_backend_kind}")
list(APPEND MT_BACKENDS ${omp_backend_kind})

message(STATUS "MT_BACKENDS = ${MT_BACKENDS}")

# If a user-defined MT backend
foreach(backend ${MT_BACKENDS})

    set_source_files_properties(${EXAMPLE_NAME}.cu PROPERTIES LANGUAGE CUDA)

    add_executable(${EXAMPLE_NAME}_example_${backend})

    target_sources(${EXAMPLE_NAME}_example_${backend}
        PUBLIC ${PROJECT_SOURCE_DIR}/${EXAMPLE_NAME}.cu
    )

    target_link_libraries(${EXAMPLE_NAME}_example_${backend} PUBLIC
        cudss OpenMP::OpenMP_CXX
    )

    # optional extra argument for the multithreaded examples, the threading layer library name
    set(extra_mt_input_arg ${cudss_LIBRARY_DIR}/libcudss_mtlayer_${backend}${CMAKE_SHARED_LIBRARY_SUFFIX})

    add_test(NAME ${EXAMPLE_NAME}_example_${backend} COMMAND ${EXAMPLE_NAME}_example_${backend} ${extra_mt_input_arg})

    if (BUILD_STATIC)
        add_executable(${EXAMPLE_NAME}_example_${backend}_static)

        target_sources(${EXAMPLE_NAME}_example_${backend}_static
            PUBLIC ${PROJECT_SOURCE_DIR}/${EXAMPLE_NAME}.cu
        )

        target_link_libraries(${EXAMPLE_NAME}_example_${backend}_static PUBLIC
            cudss_static  OpenMP::OpenMP_CXX
        )

        add_test(NAME ${EXAMPLE_NAME}_example_${backend}_static COMMAND ${EXAMPLE_NAME}_example_${backend}_static ${extra_mt_input_arg})

    endif() # BUILD_STATIC

endforeach() # MT_BACKENDS
