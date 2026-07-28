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

cmake_minimum_required(VERSION 3.23...4.0)

# cuRANDDx example project
project(curanddx_example VERSION 0.2.4 LANGUAGES CXX CUDA)

if(PROJECT_IS_TOP_LEVEL)
    # Creates curanddx_compiler_options INTERFACE target with compiler flags.
    # Link it PRIVATE to example targets: target_link_libraries(t PRIVATE curanddx_compiler_options)
    add_library(curanddx_compiler_options INTERFACE)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    if(NOT MSVC)
        # Host flags applied to CXX sources directly; to CUDA sources via -Xcompiler so
        # nvcc forwards them to the host compiler instead of misinterpreting them as nvcc flags.
        target_compile_options(curanddx_compiler_options INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-fno-strict-aliasing>
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-declarations>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},aarch64>>>:-m64>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-strict-aliasing>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-deprecated-declarations>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},aarch64>>>:-Xcompiler=-m64>
        )
        if((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR (CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*"))
            target_compile_options(curanddx_compiler_options INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:--display_error_number>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=--display_error_number>
            )
        endif()
        if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER MATCHES ".*clang.*"))
            target_compile_options(curanddx_compiler_options INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-function>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-function>
            )
        endif()
    else()
        target_compile_definitions(curanddx_compiler_options INTERFACE
            _CRT_SECURE_NO_WARNINGS
            _CRT_NONSTDC_NO_WARNINGS
            _SCL_SECURE_NO_WARNINGS
            NOMINMAX
        )
        target_compile_options(curanddx_compiler_options INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:/W3>
            $<$<COMPILE_LANGUAGE:CXX>:/WX>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
        )
    endif()

    target_compile_options(curanddx_compiler_options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --display_error_number>
    )

    # Targeted CUDA Architectures, see https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
    set(CURANDDX_CUDA_ARCHITECTURES 80-real CACHE
        STRING "List of targeted cuRANDDx CUDA architectures, for example \"75-real;80-real\""
    )

    # Remove unsupported architectures below SM75 (minimum Turing).
    set(_curanddx_archs_filtered)
    foreach(_arch IN LISTS CURANDDX_CUDA_ARCHITECTURES)
        if(_arch MATCHES "^([0-9]+)")
            if(CMAKE_MATCH_1 GREATER_EQUAL 75)
                list(APPEND _curanddx_archs_filtered "${_arch}")
            else()
                message(WARNING "curanddx: removing unsupported CUDA architecture ${_arch} (minimum SM75)")
            endif()
        else()
            if(_arch MATCHES "^(all|all-major|native)$")
                list(APPEND _curanddx_archs_filtered "${_arch}")
            else()
                message(FATAL_ERROR "curanddx: unrecognized CUDA architecture '${_arch}'. "
                    "Expected SM75+ format (e.g. \"80-real\") or one of: all, all-major, native.")
            endif()
        endif()
    endforeach()
    set(CURANDDX_CUDA_ARCHITECTURES "${_curanddx_archs_filtered}" CACHE STRING
        "List of targeted cuRANDDx CUDA architectures (SM75+), for example \"75-real;80-real\""
        FORCE)
    unset(_curanddx_archs_filtered)
    unset(_arch)
    message(STATUS "Targeted cuRANDDx CUDA Architectures: ${CURANDDX_CUDA_ARCHITECTURES}")
endif()

# ******************************************
# Tests as standalone project to enable testing release package
# ******************************************
if(PROJECT_IS_TOP_LEVEL)
    find_package(mathdx REQUIRED CONFIG
        PATHS
            "${PROJECT_SOURCE_DIR}/../.." # example/curanddx
            "/opt/nvidia/mathdx/*"
    )
    set(CURANDDX_MIN_VERSION 0.2.4)
    if(curanddx_VERSION VERSION_LESS ${CURANDDX_MIN_VERSION})
        message(FATAL_ERROR "cuRANDDx version (${curanddx_VERSION}) is less than ${CURANDDX_MIN_VERSION}")
    endif()
endif()

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Enable testing only for selected architectures
foreach(CUDA_ARCH ${CURANDDX_CUDA_ARCHITECTURES})
    # Extract SM from SM-real/SM-virtual
    string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
    list(GET CUDA_ARCH_LIST 0 ARCH)
    # Remove "a"
    string(REPLACE "a" "" ARCH "${ARCH}")
    # Remove "f"
    string(REPLACE "f" "" ARCH "${ARCH}")
    add_compile_definitions(CURANDDX_EXAMPLE_ENABLE_SM_${ARCH})
endforeach()

if(CURANDDX_EXAMPLE_NVPL_RAND_AVAILABLE)
    # -Dnvpl_root=<NVPL install root path>
    find_package(nvpl REQUIRED COMPONENTS rand)
endif()

# ###############################################################
# add_curanddx_example_nvpl_rand
# ###############################################################
function(add_curanddx_example2_nvpl_rand GROUP_TARGET EXAMPLE_NAME ENABLE_RELAXED_CONSTEXPR EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})

    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            curanddx_compiler_options
            mathdx::curanddx
            CUDA::curand
    )
    if(CURANDDX_EXAMPLE_NVPL_RAND_AVAILABLE)
        target_link_libraries(${EXAMPLE_TARGET} PRIVATE nvpl::rand_mt)
        target_compile_definitions(${EXAMPLE_TARGET} PRIVATE -DCURANDDX_EXAMPLE_NVPL_RAND_AVAILABLE)
    endif()

    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
        CUDA_ARCHITECTURES "${CURANDDX_CUDA_ARCHITECTURES}"
    )
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
        # Required to support std::tuple in device code
        "$<$<BOOL:${ENABLE_RELAXED_CONSTEXPR}>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>>"
    )
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
        LABELS "EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

function(add_curanddx_example_nvpl_rand GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    add_curanddx_example2_nvpl_rand("${GROUP_TARGET}" "${EXAMPLE_NAME}" True "${EXAMPLE_SOURCES}")
endfunction()

# ###############################################################
# add_curanddx_example
# ###############################################################
function(add_curanddx_example2 GROUP_TARGET EXAMPLE_NAME ENABLE_RELAXED_CONSTEXPR EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})

    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
        curanddx_compiler_options
        mathdx::curanddx
        CUDA::curand
    )

    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
        CUDA_ARCHITECTURES "${CURANDDX_CUDA_ARCHITECTURES}"
    )
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
        # Required to support std::tuple in device code
        "$<$<BOOL:${ENABLE_RELAXED_CONSTEXPR}>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>>"
    )
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
        LABELS "EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

function(add_curanddx_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    add_curanddx_example2("${GROUP_TARGET}" "${EXAMPLE_NAME}" True "${EXAMPLE_SOURCES}")
endfunction()

# ###############################################################
# add_curanddx_nvrtc_example
# ###############################################################
function(add_curanddx_nvrtc_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})

    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            curanddx_compiler_options
            mathdx::curanddx
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::nvrtc
    )

    # For compatibility with CUDA 13+ & CMAKE 4+
    set(CURANDDX_CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
    list(LENGTH CURANDDX_CUDA_INCLUDE_DIRS CUDA_INCLUDE_LENGTH)

    if(${CUDA_INCLUDE_LENGTH} GREATER 1)
        list(GET CURANDDX_CUDA_INCLUDE_DIRS 0 CURANDDX_CUDA_INCLUDE_DIRS)
    else()
        set(CURANDDX_CUDA_INCLUDE_DIRS "${CURANDDX_CUDA_INCLUDE_DIRS}")
    endif()

    target_compile_definitions(${EXAMPLE_TARGET}
        PRIVATE
            CUDA_INCLUDE_DIR="${CURANDDX_CUDA_INCLUDE_DIRS}"
            CUDA_CCCL_INCLUDE_DIR="${CURANDDX_CUDA_INCLUDE_DIRS}/cccl/" # required for CTK 13+
            COMMONDX_INCLUDE_DIR="${curanddx_commondx_INCLUDE_DIR}"
            CURANDDX_INCLUDE_DIRS="${curanddx_INCLUDE_DIRS}"
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
        LABELS "EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()


# ###############################################################
# cuRANDDx Examples
# ###############################################################
add_custom_target(curanddx_examples)

# cuRANDDx examples using thread API
add_curanddx_example_nvpl_rand(curanddx_examples "cuRANDDx.example.introduction_simple_pcg_thread_api" 02_thread_api/pcg_thread_api.cu)

add_curanddx_example(curanddx_examples "cuRANDDx.example.philox_thread_api" 01_introduction/philox_thread_api.cu)
add_curanddx_example(curanddx_examples "cuRANDDx.example.philox_random_bits_thread_api" 02_thread_api/philox_random_bits_thread_api.cu)
add_curanddx_example(curanddx_examples "cuRANDDx.example.xorwow_init_and_generate_thread_api" 02_thread_api/xorwow_init_and_generate_thread_api.cu)
add_curanddx_example(curanddx_examples "cuRANDDx.example.sobol_thread_api" 02_thread_api/sobol_thread_api.cu)
add_curanddx_example(curanddx_examples "cuRANDDx.example.mrg_two_distributions_thread_api" 02_thread_api/mrg_two_distributions_thread_api.cu)

add_curanddx_nvrtc_example(curanddx_examples "cuRANDDx.example.nvrtc_pcg_thread_api" 03_nvrtc/nvrtc_pcg_thread_api.cpp)
