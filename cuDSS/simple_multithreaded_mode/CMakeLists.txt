#
# Copyright 2025 NVIDIA Corporation.  All rights reserved.
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
cmake_minimum_required(VERSION 3.19)

set(EXAMPLE_NAME simple_multithreaded_mode)

project("cuDSS_${EXAMPLE_NAME}_example"
        DESCRIPTION  "cuDSS"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cudss/index.html"
        LANGUAGES    CXX CUDA)

set(CMAKE_CUDA_STANDARD          11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS        OFF)

option(BUILD_STATIC           "Building statically linked examples" OFF)
option(BUILD_MT_WITH_OPENMP   "Enable OpenMP backend for the MT mode examples" ON)

# Find cuDSS
find_package(cudss 0.6.0 REQUIRED)

# Find OpenMP
if (BUILD_MT_WITH_OPENMP)
    find_package(OpenMP REQUIRED)
    if (NOT OpenMP_CXX_FOUND)
        message(FATAL_ERROR "For building MT examples with OpenMP, OpenMP \
package must be detectable by cmake")
    endif()
endif()

# (optional: only if samples are run with ctest)
enable_testing()

# Define a set of MT backends (currently, only one, OpenMP but this may be extended as
# cuDSS supports user-define threading backends)
set(MT_BACKENDS "")
if (BUILD_MT_WITH_OPENMP)
    # Define the flavor of OpenMP picked up by find_package()
    if (OpenMP_CXX_LIBRARIES OR OpenMP_C_LIBRARIES) # should work on Linux
        if (OpenMP_CXX_LIBRARIES)
            string(REGEX MATCH ".*lib([A-Za-z0-9]*omp[A-Za-z0-9]*)[.](so|dll).*" dummy ${OpenMP_CXX_LIBRARIES})
        else()
            string(REGEX MATCH ".*lib([A-Za-z0-9]*omp[A-Za-z0-9]*)[.](so|dll).*" dummy ${OpenMP_C_LIBRARIES})
        endif()
        set(omp_backend_kind ${CMAKE_MATCH_1})
    else() # fallback for Windows
        set(omp_backend_kind "omp")
    endif()
    message(STATUS "omp_backend_kind = ${omp_backend_kind}")
    set(MT_BACKENDS ${MT_BACKENDS} ${omp_backend_kind})
endif()

message(STATUS "MT_BACKENDS = ${MT_BACKENDS}")

foreach(backend ${MT_BACKENDS})

    set_source_files_properties(${EXAMPLE_NAME}.cpp PROPERTIES LANGUAGE CUDA)

    add_executable(${EXAMPLE_NAME}_example_${backend})

    target_sources(${EXAMPLE_NAME}_example_${backend}
        PUBLIC ${PROJECT_SOURCE_DIR}/${EXAMPLE_NAME}.cpp
    )

    if (WIN32)
        target_include_directories(${EXAMPLE_NAME}_example_${backend} PUBLIC
            ${cudss_INCLUDE_DIR}
        )
        target_link_directories(${EXAMPLE_NAME}_example_${backend} PUBLIC
            ${cudss_LIBRARY_DIR}
        )
    endif()

    target_link_libraries(${EXAMPLE_NAME}_example_${backend} PUBLIC
        cudss
    )

    if ("${backend}" MATCHES "omp")
        target_link_libraries(${EXAMPLE_NAME}_example_${backend} PUBLIC
            OpenMP::OpenMP_CXX
        )

        target_compile_options(${EXAMPLE_NAME}_example_${backend} PRIVATE -DUSE_OPENMP)
    endif()

    # optional extra argument for the multithreaded examples, the threading layer library name
    set(extra_mt_input_arg ${cudss_LIBRARY_DIR}/libcudss_mtlayer_${backend}${CMAKE_SHARED_LIBRARY_SUFFIX})

    add_test(NAME ${EXAMPLE_NAME}_example_${backend} COMMAND ${EXAMPLE_NAME}_example_${backend} ${extra_mt_input_arg})

    if (BUILD_STATIC)
        add_executable(${EXAMPLE_NAME}_example_${backend}_static)

        target_sources(${EXAMPLE_NAME}_example_${backend}_static
            PUBLIC ${PROJECT_SOURCE_DIR}/${EXAMPLE_NAME}.cpp
        )

        if (WIN32)
            target_include_directories(${EXAMPLE_NAME}_example_${backend}_static PUBLIC
                ${cudss_INCLUDE_DIR}
            )
            target_link_directories(${EXAMPLE_NAME}_example_${backend}_static PUBLIC
                ${cudss_LIBRARY_DIR}
            )
            target_link_libraries(${EXAMPLE_NAME}_example_${backend}_static PUBLIC
                cudss
            )
        else()
            target_link_libraries(${EXAMPLE_NAME}_example_${backend}_static PUBLIC
                cudss_static
            )
        endif()


        if ("${backend}" MATCHES "omp")
            target_link_libraries(${EXAMPLE_NAME}_example_${backend}_static PUBLIC
                OpenMP::OpenMP_CXX
            )

            target_compile_options(${EXAMPLE_NAME}_example_${backend}_static PRIVATE -DUSE_OPENMP)
        endif()


        add_test(NAME ${EXAMPLE_NAME}_example_${backend}_static COMMAND ${EXAMPLE_NAME}_example_${backend}_static ${extra_mt_input_arg})

    endif() # BUILD_STATIC

endforeach() # MT_BACKENDS

