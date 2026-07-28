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

cmake_minimum_required(VERSION 3.18...4.0)

# cuBLASDx Examples project
project(cublasdx_examples VERSION 0.7.1 LANGUAGES CXX CUDA)

# PROJECT_IS_TOP_LEVEL is available since 3.21
if(CMAKE_VERSION VERSION_LESS 3.21)
    get_directory_property(project_has_parent PARENT_DIRECTORY)
    if(project_has_parent)
        set(PROJECT_IS_TOP_LEVEL FALSE)
    else()
        set(PROJECT_IS_TOP_LEVEL TRUE)
    endif()
endif()

if(PROJECT_IS_TOP_LEVEL)
    enable_testing()

    # Creates cublasdx_compiler_options INTERFACE target with compiler flags.
    # Link it PRIVATE to example targets: target_link_libraries(t PRIVATE cublasdx_compiler_options)
    add_library(cublasdx_compiler_options INTERFACE)

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
        target_compile_options(cublasdx_compiler_options INTERFACE
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
            target_compile_options(cublasdx_compiler_options INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:--display_error_number>
                $<$<COMPILE_LANGUAGE:CXX>:--diag_suppress1>
                $<$<COMPILE_LANGUAGE:CXX>:--diag_suppress111>
                $<$<COMPILE_LANGUAGE:CXX>:--diag_suppress177>
                $<$<COMPILE_LANGUAGE:CXX>:--diag_suppress185>
                $<$<COMPILE_LANGUAGE:CXX>:--diag_suppress941>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=--display_error_number>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=--diag_suppress1>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=--diag_suppress111>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=--diag_suppress177>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=--diag_suppress185>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=--diag_suppress941>
            )
        endif()
        # CUTLASS/CuTe workarounds
        target_compile_options(cublasdx_compiler_options INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-switch>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<AND:$<CXX_COMPILER_ID:Clang>,$<VERSION_LESS:${CMAKE_CXX_COMPILER_VERSION},13.0.1>>>:-Wno-unused-parameter>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CXX_COMPILER_ID:NVHPC>,$<STREQUAL:${CMAKE_CXX_COMPILER},nvc++>>>:-Wno-unused-parameter>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:NVHPC>>>>:-Wno-unused-but-set-parameter>
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-switch>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<AND:$<CXX_COMPILER_ID:Clang>,$<VERSION_LESS:${CMAKE_CXX_COMPILER_VERSION},13.0.1>>>:-Xcompiler=-Wno-unused-parameter>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CXX_COMPILER_ID:NVHPC>,$<STREQUAL:${CMAKE_CXX_COMPILER},nvc++>>>:-Xcompiler=-Wno-unused-parameter>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:NVHPC>>>>:-Xcompiler=-Wno-unused-but-set-parameter>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-sign-compare>
        )
    else()
        target_compile_definitions(cublasdx_compiler_options INTERFACE
            _CRT_SECURE_NO_WARNINGS
            _CRT_NONSTDC_NO_WARNINGS
            _SCL_SECURE_NO_WARNINGS
            NOMINMAX
        )
        target_compile_options(cublasdx_compiler_options INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:/W3>
            $<$<COMPILE_LANGUAGE:CXX>:/WX>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
        )
    endif()

    target_compile_options(cublasdx_compiler_options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --display_error_number>
    )

    # Targeted CUDA Architectures, see https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
    set(CUBLASDX_CUDA_ARCHITECTURES 80-real CACHE
        STRING "List of targeted cuBLASDx CUDA architectures, for example \"75-real;80-virtual;90-real\""
    )
    # Remove unsupported architectures below SM75 (minimum Turing).
    set(_cublasdx_archs_filtered)
    foreach(_arch IN LISTS CUBLASDX_CUDA_ARCHITECTURES)
        if(_arch MATCHES "^([0-9]+)")
            if(CMAKE_MATCH_1 GREATER_EQUAL 75)
                list(APPEND _cublasdx_archs_filtered "${_arch}")
            else()
                message(WARNING "cublasdx: removing unsupported CUDA architecture ${_arch} (minimum SM75)")
            endif()
        else()
            if(_arch MATCHES "^(all|all-major|native)$")
                list(APPEND _cublasdx_archs_filtered "${_arch}")
            else()
                message(FATAL_ERROR "cublasdx: unrecognized CUDA architecture '${_arch}'. "
                    "Expected SM75+ format (e.g. \"80-real\") or one of: all, all-major, native.")
            endif()
        endif()
    endforeach()
    set(CUBLASDX_CUDA_ARCHITECTURES "${_cublasdx_archs_filtered}" CACHE STRING
        "List of targeted cuBLASDx CUDA architectures (SM75+), for example \"75-real;80-real\""
        FORCE)
    unset(_cublasdx_archs_filtered)
    unset(_arch)
    message(STATUS "Targeted cuBLASDx CUDA Architectures: ${CUBLASDX_CUDA_ARCHITECTURES}")

    # Detect whether any accelerated architecture (e.g. 90a, 100a) is targeted.
    set(CUBLASDX_HAS_ACCELERATED_ARCH FALSE)
    foreach(_CUBLASDX_ARCH ${CUBLASDX_CUDA_ARCHITECTURES})
        if(_CUBLASDX_ARCH MATCHES "^[0-9]+[af](-|$)")
            set(CUBLASDX_HAS_ACCELERATED_ARCH TRUE)
            break()
        endif()
    endforeach()
    unset(_CUBLASDX_ARCH)

    # Fatbin compilation/linking is not supported when targeting accelerated
    # architectures with CUDA Toolkit < 13.2. In that case cuBLASDx operates in
    # header-only mode (GEMM still works; TRSM is unavailable).
    if(CUBLASDX_HAS_ACCELERATED_ARCH AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.2")
        set(CUBLASDX_SKIP_FATBIN TRUE)
        message(STATUS "cuBLASDx: accelerated arch with CTK < 13.2 - fatbin disabled (GEMM-only header mode, TRSM unavailable, NVBUG5504462)")
    else()
        set(CUBLASDX_SKIP_FATBIN FALSE)
    endif()
endif()

if(PROJECT_IS_TOP_LEVEL OR CUBLASDX_TEST_PACKAGE)
    find_package(mathdx REQUIRED CONFIG
        PATHS
            "${PROJECT_SOURCE_DIR}/../.." # example/cublasdx
            "/opt/nvidia/mathdx/*"
    )
    set(CUBLASDX_MIN_VERSION 0.7.1)
    if(cublasdx_VERSION VERSION_LESS ${CUBLASDX_MIN_VERSION})
        message(FATAL_ERROR "cuBLASDx version (${cublasdx_VERSION}) is less than ${CUBLASDX_MIN_VERSION}")
    endif()
endif()


# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Enable testing only for selected architectures
foreach(CUDA_ARCH ${CUBLASDX_CUDA_ARCHITECTURES})
    # Extract SM from SM-real/SM-virtual
    string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
    list(GET CUDA_ARCH_LIST 0 ARCH)
    # Remove "f"
    string(REPLACE "f" "" ARCH "${ARCH}")

    add_compile_definitions(CUBLASDX_EXAMPLE_ENABLE_SM_${ARCH})
endforeach()

###############################################################
# link_lto_library_workaround
# ###############################################################
# Workaround for LTO support in CMAKE <3.25.  INTERPROCEDURAL_OPTIMIZATION
# handles device LTO natively from CMake 3.25 onward; on older CMake the
# device-link flag and per-arch lto-code generation must be added manually.
# The example file declares cmake_minimum_required(3.18...4.0), so users on
# CMake 3.18-3.24 hit this path.
function(link_lto_library_workaround TARGET_NAME CUDA_ARCHITECTURES)
    target_link_options(${TARGET_NAME}
        PRIVATE
            $<DEVICE_LINK:-dlto>
    )
    target_compile_options(${TARGET_NAME}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-rdc=true>"
    )
    foreach(CUDA_ARCH ${CUDA_ARCHITECTURES})
        # Extract SM from SM-real/SM-virtual
        string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
        list(GET CUDA_ARCH_LIST 0 ARCH)
        target_compile_options(${TARGET_NAME}
            PRIVATE
                "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--generate-code arch=compute_${ARCH},code=lto_${ARCH}>"
        )
    endforeach()
endfunction()

###############################################################
# Conditionally link cublasdx
###############################################################
function(conditionally_link_cublasdx_fatbin EXAMPLE_TARGET)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.2)
        target_link_libraries(${EXAMPLE_TARGET}
            PRIVATE
            $<IF:$<TARGET_EXISTS:mathdx::cublasdx>,mathdx::cublasdx,cublasdx::cublasdx>
        )
        set_target_properties(${EXAMPLE_TARGET}
            PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            INTERPROCEDURAL_OPTIMIZATION ON # This is basically "$<DEVICE_LINK:-dlto>" but it requires CMake 3.25+
            CUDA_ARCHITECTURES "${CUBLASDX_CUDA_ARCHITECTURES}"
        )
        # Workaround for LTO support in CMAKE <3.25
        if(CMAKE_VERSION VERSION_LESS 3.25)
            link_lto_library_workaround(${EXAMPLE_TARGET} "${CUBLASDX_CUDA_ARCHITECTURES}")
        endif()
    elseif(CUBLASDX_SKIP_FATBIN)
        target_link_libraries(${EXAMPLE_TARGET}
            PRIVATE
            $<IF:$<TARGET_EXISTS:mathdx::cublasdx>,mathdx::cublasdx,cublasdx::cublasdx>
        )
    else()
        # Link cuBLASDx fatbin
        target_link_libraries(${EXAMPLE_TARGET}
            PRIVATE
                $<IF:$<TARGET_EXISTS:mathdx::cublasdx_fatbin>,mathdx::cublasdx_fatbin,cublasdx::cublasdx_fatbin>
        )
        set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            INTERPROCEDURAL_OPTIMIZATION ON # This is basically "$<DEVICE_LINK:-dlto>" but it requires CMake 3.25+
            CUDA_ARCHITECTURES "${CUBLASDX_CUDA_ARCHITECTURES}"
        )
        # Workaround for LTO support in CMAKE <3.25
        if(CMAKE_VERSION VERSION_LESS 3.25)
            link_lto_library_workaround(${EXAMPLE_TARGET} "${CUBLASDX_CUDA_ARCHITECTURES}")
        endif()
    endif()
endfunction()


###############################################################
# make_cublasdx_example_target_name
# ###############################################################
function(make_cublasdx_example_target_name EXAMPLE_MAIN_SOURCE EXAMPLE_TARGET_VAR)
    # Keep numeric directory prefixes, but avoid repeating the same example name
    # when the directory already uniquely identifies the target.
    set(EXAMPLE_TARGET "")

    if("${EXAMPLE_MAIN_SOURCE}" STREQUAL "04_gemm_blockdim/blockdim_gemm_fp16.cu")
        set(EXAMPLE_TARGET "04_gemm_blockdim_fp16")
    elseif("${EXAMPLE_MAIN_SOURCE}" STREQUAL "05_gemm_batched/batched_gemm_fp64.cu")
        set(EXAMPLE_TARGET "05_gemm_batched_fp64")
    elseif("${EXAMPLE_MAIN_SOURCE}" STREQUAL "11_gemm_device_performance/device_gemm_performance.cu")
        set(EXAMPLE_TARGET "11_gemm_device_performance")
    elseif("${EXAMPLE_MAIN_SOURCE}" STREQUAL "12_gemm_device_partial_sums/gemm_device_partial_sums.cu")
        set(EXAMPLE_TARGET "12_gemm_device_partial_sums")
    elseif("${EXAMPLE_MAIN_SOURCE}" STREQUAL "13_gemm_fft/gemm_fft.cu")
        set(EXAMPLE_TARGET "13_gemm_fft")
    elseif("${EXAMPLE_MAIN_SOURCE}" STREQUAL "13_gemm_fft/gemm_fft_fp16.cu")
        set(EXAMPLE_TARGET "13_gemm_fft_fp16")
    elseif("${EXAMPLE_MAIN_SOURCE}" STREQUAL "13_gemm_fft/gemm_fft_performance.cu")
        set(EXAMPLE_TARGET "13_gemm_fft_performance")
    elseif("${EXAMPLE_MAIN_SOURCE}" STREQUAL "16_dgemm_emulation/dgemm_emulation.cu")
        set(EXAMPLE_TARGET "16_dgemm_emulation")
    elseif("${EXAMPLE_MAIN_SOURCE}" STREQUAL "19_gemm_batched/batched_gemm.cu")
        set(EXAMPLE_TARGET "19_gemm_batched")
    elseif("${EXAMPLE_MAIN_SOURCE}" STREQUAL "19_gemm_batched/batched_gemm_pipeline.cu")
        set(EXAMPLE_TARGET "19_gemm_batched_pipeline")
    elseif("${EXAMPLE_MAIN_SOURCE}" STREQUAL "20_dgemm_emulation_pipeline/dgemm_emulation_pipeline.cu")
        set(EXAMPLE_TARGET "20_dgemm_emulation_pipeline")
    elseif("${EXAMPLE_MAIN_SOURCE}" STREQUAL "21_sgemm_emulation_pipeline/sgemm_emulation_pipeline.cu")
        set(EXAMPLE_TARGET "21_sgemm_emulation_pipeline")
    endif()

    if(NOT EXAMPLE_TARGET)
        get_filename_component(EXAMPLE_FILENAME "${EXAMPLE_MAIN_SOURCE}" NAME_WE)
        get_filename_component(EXAMPLE_DIR "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)
        set(EXAMPLE_TARGET "${EXAMPLE_DIR}_${EXAMPLE_FILENAME}")
    endif()

    if(TARGET "${EXAMPLE_TARGET}")
        message(FATAL_ERROR "cuBLASDx example target '${EXAMPLE_TARGET}' is already defined. Check example target-name overrides.")
    endif()

    set(${EXAMPLE_TARGET_VAR} "${EXAMPLE_TARGET}" PARENT_SCOPE)
endfunction()

###############################################################
# Common correctness checking objects
# ###############################################################
if (NOT TARGET common_correctness_objects)
    set(common_correctness_objects_sources reference/check_error.cu reference/naive_reference.cu)
    set_source_files_properties("${common_correctness_objects_sources}" PROPERTIES LANGUAGE CUDA)
    add_library(common_correctness_objects STATIC "${common_correctness_objects_sources}")
    set_target_properties(common_correctness_objects
        PROPERTIES
            CUDA_ARCHITECTURES "${CUBLASDX_CUDA_ARCHITECTURES}"
    )
    conditionally_link_cublasdx_fatbin(common_correctness_objects)
endif()

# ###############################################################
# add_cufftdx_cublasdx_example
# ###############################################################
function(add_cufftdx_cublasdx_example GROUP_TARGET EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)

    # Get the absolute path and filename
    get_filename_component(EXAMPLE_FILENAME "${EXAMPLE_MAIN_SOURCE}" NAME_WE)
    get_filename_component(EXAMPLE_DIR "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)
    make_cublasdx_example_target_name("${EXAMPLE_MAIN_SOURCE}" EXAMPLE_TARGET)

    set(EXAMPLE_NAME "cuBLASDx.example.${EXAMPLE_DIR}.${EXAMPLE_FILENAME}")

     # Compose output directory: <build_dir>/<subdir>
    set(EXAMPLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_DIR}")

    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})

    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            cublasdx_compiler_options
            CUDA::cublas
            cufftdx::cufftdx
            CUDA::cufft
            common_correctness_objects
    )

    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_ARCHITECTURES "${CUBLASDX_CUDA_ARCHITECTURES}"
    )

    conditionally_link_cublasdx_fatbin(${EXAMPLE_TARGET})
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})

    # Set the output directory and executable name
    set_target_properties(${EXAMPLE_TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_DIR}"
        OUTPUT_NAME "${EXAMPLE_FILENAME}"
    )

    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
            "--extended-lambda"
    )

    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# add_cublasdx_example
# ###############################################################
function(add_cublasdx_example2 GROUP_TARGET EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)

    # Get the absolute path and filename
    get_filename_component(EXAMPLE_FILENAME "${EXAMPLE_MAIN_SOURCE}" NAME_WE)
    get_filename_component(EXAMPLE_DIR "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)
    make_cublasdx_example_target_name("${EXAMPLE_MAIN_SOURCE}" EXAMPLE_TARGET)

    set(EXAMPLE_NAME "cuBLASDx.example.${EXAMPLE_DIR}.${EXAMPLE_FILENAME}")

     # Compose output directory: <build_dir>/<subdir>
    set(EXAMPLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_DIR}")

    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            cublasdx_compiler_options
            CUDA::cublas
            common_correctness_objects
    )

    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_ARCHITECTURES "${CUBLASDX_CUDA_ARCHITECTURES}"
    )

    conditionally_link_cublasdx_fatbin(${EXAMPLE_TARGET})

    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})

    # Set the output directory and executable name
    set_target_properties(${EXAMPLE_TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_DIR}"
        OUTPUT_NAME "${EXAMPLE_FILENAME}"
    )

    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    )

    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

function(add_cublasdx_example GROUP_TARGET EXAMPLE_SOURCES)
    add_cublasdx_example2("${GROUP_TARGET}" "${EXAMPLE_SOURCES}")
endfunction()

# ###############################################################
# add_cublasdx_nvrtc_example
# ###############################################################
function(add_cublasdx_nvrtc_example GROUP_TARGET EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)

    # Get the absolute path and filename
    get_filename_component(EXAMPLE_FILENAME "${EXAMPLE_MAIN_SOURCE}" NAME_WE)
    get_filename_component(EXAMPLE_DIR "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)
    make_cublasdx_example_target_name("${EXAMPLE_MAIN_SOURCE}" EXAMPLE_TARGET)

    set(EXAMPLE_NAME "cuBLASDx.example.${EXAMPLE_DIR}.${EXAMPLE_FILENAME}")
     # Compose output directory: <build_dir>/<subdir>
    set(EXAMPLE_DIR "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_DIR}")

    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            cublasdx_compiler_options
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::nvrtc
            CUDA::nvJitLink
            CUDA::cublas
            common_correctness_objects
    )

    conditionally_link_cublasdx_fatbin(${EXAMPLE_TARGET})

    set(CUBLASDX_CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
    list(LENGTH CUBLASDX_CUDA_INCLUDE_DIRS CUDA_INCLUDE_LENGTH)

    if(${CUDA_INCLUDE_LENGTH} GREATER 1)
        list(GET CUBLASDX_CUDA_INCLUDE_DIRS 0 CUBLASDX_CUDA_INCLUDE_DIR)
    else()
        set(CUBLASDX_CUDA_INCLUDE_DIR "${CUBLASDX_CUDA_INCLUDE_DIRS}")
    endif()

    target_compile_definitions(${EXAMPLE_TARGET}
        PRIVATE
            CUDA_INCLUDE_DIR="${CUBLASDX_CUDA_INCLUDE_DIR}"
            CUTLASS_INCLUDE_DIR="${cublasdx_cutlass_INCLUDE_DIR}"
            COMMONDX_INCLUDE_DIR="${cublasdx_commondx_INCLUDE_DIR}"
            CUBLASDX_INCLUDE_DIRS="${cublasdx_INCLUDE_DIRS}"
    )
    if(cublasdx_FATBIN)
        target_compile_definitions(${EXAMPLE_TARGET}
            PRIVATE
                CUBLASDX_FATBIN="${cublasdx_FATBIN}"
        )
    endif()
    # Disable NVRTC examples for NVC++ compiler, because of libcu++/NRTC/NVC++ bug
    if(NOT ((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR (CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*")))
        add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
        set_tests_properties(${EXAMPLE_NAME}
            PROPERTIES
                LABELS "EXAMPLE"
        )
    endif()

    # Set the output directory and executable name
    set_target_properties(${EXAMPLE_TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${EXAMPLE_DIR}"
        OUTPUT_NAME "${EXAMPLE_FILENAME}"
    )

    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_ARCHITECTURES "${CUBLASDX_CUDA_ARCHITECTURES}"
    )
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# cuBLASDx Examples
# ###############################################################

add_custom_target(cublasdx_examples)

# cuBLASDx NVRTC examples
add_cublasdx_nvrtc_example(cublasdx_examples 15_nvrtc/nvrtc_gemm.cpp)
if(NOT CUBLASDX_SKIP_FATBIN)
    add_cublasdx_nvrtc_example(cublasdx_examples 15_nvrtc/nvrtc_trsm.cpp)
else()
    message(STATUS "cuBLASDx: TRSM NVRTC example disabled (fatbin linking skipped)")
endif()

# cuBLASDx introduction examples
add_cublasdx_example(cublasdx_examples 01_gemm_introduction/introduction_example.cu)
add_cublasdx_example(cublasdx_examples 01_gemm_introduction/introduction_pipeline.cu)

# cuBLASDx simple examples
add_cublasdx_example(cublasdx_examples 02_gemm_precisions/simple_gemm_mixed_precision.cu)
add_cublasdx_example(cublasdx_examples 02_gemm_precisions/simple_gemm_fp32.cu)
add_cublasdx_example(cublasdx_examples 02_gemm_precisions/simple_gemm_int8_int8_int32.cu)
add_cublasdx_example(cublasdx_examples 08_gemm_decoupled_io_and_compute/simple_gemm_fp32_decoupled.cu)
add_cublasdx_example(cublasdx_examples 03_gemm_complex/simple_gemm_cfp16.cu)
add_cublasdx_example(cublasdx_examples 02_gemm_precisions/simple_gemm_fp8.cu)
add_cublasdx_example(cublasdx_examples 03_gemm_complex/simple_gemm_std_complex_fp32.cu)
add_cublasdx_example(cublasdx_examples 06_gemm_leading_dimension/simple_gemm_leading_dimensions.cu)
add_cublasdx_example(cublasdx_examples 09_gemm_custom_layout/simple_gemm_custom_layout.cu)
add_cublasdx_example(cublasdx_examples 07_gemm_transform/simple_gemm_transform.cu)
add_cublasdx_example(cublasdx_examples 09_gemm_custom_layout/simple_gemm_aat.cu)
add_cublasdx_example(cublasdx_examples 11_gemm_device_performance/device_gemm_performance.cu)
add_cublasdx_example(cublasdx_examples 12_gemm_device_partial_sums/gemm_device_partial_sums.cu)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.1)
    message("Enabling example 19, gemm_batched, CTK is compatible")
    add_cublasdx_example(cublasdx_examples 19_gemm_batched/batched_gemm_pipeline.cu)
    add_cublasdx_example(cublasdx_examples 19_gemm_batched/batched_gemm.cu)
else()
    message("Disabling example 19, gemm_batched, due to incompatible CTK (13.1 required)")
endif()

# cuBLASDx advanced examples
add_cublasdx_example(cublasdx_examples 14_gemm_fused/gemm_fusion.cu)
add_cublasdx_example(cublasdx_examples 04_gemm_blockdim/blockdim_gemm_fp16.cu)
add_cublasdx_example(cublasdx_examples 05_gemm_batched/batched_gemm_fp64.cu)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.1)
    message("Enabling examples 16, 20, and 21, emulation, CTK is compatible")
    add_cublasdx_example(cublasdx_examples 16_dgemm_emulation/dgemm_emulation.cu)
    add_cublasdx_example(cublasdx_examples 20_dgemm_emulation_pipeline/dgemm_emulation_pipeline.cu)
    add_cublasdx_example(cublasdx_examples 21_sgemm_emulation_pipeline/sgemm_emulation_pipeline.cu)
else()
    message("Disabling examples 16, 20, and 21, emulation, due to incompatible CTK (13.1 required)")
endif()

# cuBLASDx TRSM examples
if(NOT CUBLASDX_SKIP_FATBIN)
    add_cublasdx_example(cublasdx_examples 17_trsm/trsm_block.cu)
    add_cublasdx_example(cublasdx_examples 17_trsm/trsm_thread.cu)
    add_cublasdx_example(cublasdx_examples 18_tensor_transform/trsm_conj_transpose.cu)
else()
    message(STATUS "cuBLASDx: TRSM examples disabled (fatbin linking skipped)")
endif()

# cuBLASDx transform examples (GEMM, no fatbin required)
add_cublasdx_example(cublasdx_examples 18_tensor_transform/gemm_conj_transpose.cu)

# cuBLASDx performance examples
add_cublasdx_example(cublasdx_examples 10_gemm_block_performance/single_gemm_performance.cu)
add_cublasdx_example(cublasdx_examples 14_gemm_fused/fused_gemm_performance.cu)

# cuBLASDx/cuFFTDx examples
if(cufftdx_FOUND)
    add_cufftdx_cublasdx_example(cublasdx_examples 13_gemm_fft/gemm_fft.cu)
    add_cufftdx_cublasdx_example(cublasdx_examples 13_gemm_fft/gemm_fft_fp16.cu)
    add_cufftdx_cublasdx_example(cublasdx_examples 13_gemm_fft/gemm_fft_performance.cu)
endif()
