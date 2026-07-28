# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# cuSOLVERDx examples project
project(cusolverdx_examples VERSION 0.5.0 LANGUAGES CXX CUDA)

# PROJECT_IS_TOP_LEVEL is available since 3.21
if(CMAKE_VERSION VERSION_LESS 3.21)
    get_directory_property(project_has_parent PARENT_DIRECTORY)
    if(project_has_parent)
        set(PROJECT_IS_TOP_LEVEL FALSE)
    else()
        set(PROJECT_IS_TOP_LEVEL TRUE)
    endif()
endif()

set(CUSOLVERDX_EXAMPLES_DEBUG_BUILD OFF)
if((CMAKE_BUILD_TYPE STREQUAL "DEBUG") OR (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(CUSOLVERDX_EXAMPLES_DEBUG_BUILD ON)
endif()
message(STATUS "CUSOLVERDX_EXAMPLES_DEBUG_BUILD: ${CUSOLVERDX_EXAMPLES_DEBUG_BUILD}")
set(CUSOLVERDX_EXAMPLES_LINUX_X86_64 OFF)
if((CMAKE_SYSTEM_NAME STREQUAL "Linux") AND (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)"))
    set(CUSOLVERDX_EXAMPLES_LINUX_X86_64 ON)
endif()
message(STATUS "CUSOLVERDX_EXAMPLES_LINUX_X86_64: ${CUSOLVERDX_EXAMPLES_LINUX_X86_64}")

if(PROJECT_IS_TOP_LEVEL)
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to 'Release' as none was specified. Set CMAKE_BUILD_TYPE variable to change build type.")
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    endif()
    message(STATUS "CMAKE_BUILD_TYPE set to: ${CMAKE_BUILD_TYPE}")

    # Enable running examples with CTest
    enable_testing()

    # ******************************************
    # cuSOLVERDx Install Dirs
    # ******************************************
    include(GNUInstallDirs)
    set(CUSOLVERDX_EXAMPLES_BIN_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}/example/")

    # Creates cusolverdx_compiler_options INTERFACE target with compiler flags.
    # Link it PRIVATE to example targets: target_link_libraries(t PRIVATE cusolverdx_compiler_options)
    add_library(cusolverdx_compiler_options INTERFACE)

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
        target_compile_options(cusolverdx_compiler_options INTERFACE
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
            target_compile_options(cusolverdx_compiler_options INTERFACE
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
        if((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER MATCHES ".*clang.*"))
            target_compile_options(cusolverdx_compiler_options INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-function>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-function>
            )
        endif()
        # CUTLASS/CuTe workarounds
        target_compile_options(cusolverdx_compiler_options INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-switch>
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<AND:$<CXX_COMPILER_ID:Clang>,$<VERSION_LESS:${CMAKE_CXX_COMPILER_VERSION},13.0.1>>>:-Wno-unused-parameter>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CXX_COMPILER_ID:NVHPC>,$<STREQUAL:${CMAKE_CXX_COMPILER},nvc++>>>:-Wno-unused-parameter>
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:NVHPC>>>>:-Wno-unused-but-set-parameter>
            $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-switch>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unknown-pragmas>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<AND:$<CXX_COMPILER_ID:Clang>,$<VERSION_LESS:${CMAKE_CXX_COMPILER_VERSION},13.0.1>>>:-Xcompiler=-Wno-unused-parameter>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CXX_COMPILER_ID:NVHPC>,$<STREQUAL:${CMAKE_CXX_COMPILER},nvc++>>>:-Xcompiler=-Wno-unused-parameter>
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:NVHPC>>>>:-Xcompiler=-Wno-unused-but-set-parameter>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-sign-compare>
        )
    else()
        target_compile_definitions(cusolverdx_compiler_options INTERFACE
            _CRT_SECURE_NO_WARNINGS
            _CRT_NONSTDC_NO_WARNINGS
            _SCL_SECURE_NO_WARNINGS
            NOMINMAX
        )
        target_compile_options(cusolverdx_compiler_options INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:/W3>
            $<$<COMPILE_LANGUAGE:CXX>:/WX>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/WX>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>
        )
    endif()

    target_compile_options(cusolverdx_compiler_options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --display_error_number>
        # Enable lineinfo in Release
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:SHELL:-lineinfo>
        # Enable device debugging for static library only in Debug mode
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<AND:$<NOT:$<BOOL:$<TARGET_PROPERTY:INTERPROCEDURAL_OPTIMIZATION>>>,$<CONFIG:Debug>>>:-G>
    )

    # Targeted CUDA Architectures, see https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
    set(CUSOLVERDX_CUDA_ARCHITECTURES 80-real CACHE
        STRING "List of targeted cuSOLVERDx CUDA architectures (SM75+), for example \"75-real;80-real\""
    )

    # Remove unsupported architectures below SM75 (minimum Turing).
    set(_cusolverdx_archs_filtered)
    foreach(_arch IN LISTS CUSOLVERDX_CUDA_ARCHITECTURES)
        if(_arch MATCHES "^([0-9]+)")
            if(CMAKE_MATCH_1 GREATER_EQUAL 75)
                list(APPEND _cusolverdx_archs_filtered "${_arch}")
            else()
                message(WARNING "cusolverdx: removing unsupported CUDA architecture ${_arch} (minimum SM75)")
            endif()
        else()
            if(_arch MATCHES "^(all|all-major|native)$")
                list(APPEND _cusolverdx_archs_filtered "${_arch}")
            else()
                message(FATAL_ERROR "cusolverdx: unrecognized CUDA architecture '${_arch}'. "
                    "Expected SM75+ format (e.g. \"80-real\") or one of: all, all-major, native.")
            endif()
        endif()
    endforeach()
    set(CUSOLVERDX_CUDA_ARCHITECTURES "${_cusolverdx_archs_filtered}" CACHE STRING
        "List of targeted cuSOLVERDx CUDA architectures (SM75+), for example \"75-real;80-real\""
        FORCE)
    unset(_cusolverdx_archs_filtered)
    unset(_arch)
    message(STATUS "Targeted cuSOLVERDx CUDA Architectures: ${CUSOLVERDX_CUDA_ARCHITECTURES}")
endif()

# ******************************************
# Tests as standalone project to enable testing release package
# ******************************************
if(PROJECT_IS_TOP_LEVEL)
    file(GLOB INSTALLED_MATHDX_DIRECTORIES LIST_DIRECTORIES TRUE "/opt/nvidia/mathdx/*")
    find_package(mathdx REQUIRED COMPONENTS cusolverdx cublasdx CONFIG
        PATHS
            "${PROJECT_SOURCE_DIR}/../.." # example/cusolverdx
            "${INSTALLED_MATHDX_DIRECTORIES}"
    )
    set(CUSOLVERDX_MIN_VERSION 0.5.0)
    if(cusolverdx_VERSION VERSION_LESS ${CUSOLVERDX_MIN_VERSION})
        message(FATAL_ERROR "cuSOLVERDx version (${cusolverdx_VERSION}) is less than ${CUSOLVERDX_MIN_VERSION}")
    endif()
endif()

if(TARGET mathdx::cublasdx AND NOT DEFINED cusolverdx_cublasdx_INCLUDE_DIRS)
    get_property(cusolverdx_cublasdx_INCLUDE_DIRS TARGET mathdx::cublasdx PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
endif()

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Enable testing only for selected architectures
foreach(CUDA_ARCH ${CUSOLVERDX_CUDA_ARCHITECTURES})
    # Extract SM from SM-real/SM-virtual
    string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
    list(GET CUDA_ARCH_LIST 0 ARCH)
    # Remove "a"
    string(REPLACE "a" "" ARCH "${ARCH}")
    # Remove "f"
    string(REPLACE "f" "" ARCH "${ARCH}")

    add_compile_definitions(CUSOLVERDX_EXAMPLE_ENABLE_SM_${ARCH})
endforeach()

# ###############################################################
# install_example
# ###############################################################
function(install_example EXAMPLE_TARGET)
    install(TARGETS ${EXAMPLE_TARGET}
        RUNTIME DESTINATION ${CUSOLVERDX_EXAMPLES_BIN_INSTALL_DIR}/
        COMPONENT Examples
    )
endfunction()


# ###############################################################
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


# ##############################################################
# Common example objects
# #############################################################
set(common_example_objects_sources common/error_checking.cpp)
set_source_files_properties("${common_example_objects_sources}" PROPERTIES LANGUAGE CUDA)
add_library(common_example_objects STATIC "${common_example_objects_sources}")
target_link_libraries(common_example_objects PRIVATE cusolverdx_compiler_options mathdx::cusolverdx_fatbin)
target_compile_definitions(common_example_objects PRIVATE CUSOLVERDX_TEST_NO_GTEST)
set_target_properties(common_example_objects
    PROPERTIES
        CUDA_ARCHITECTURES "${CUSOLVERDX_CUDA_ARCHITECTURES}"
)

# ###############################################################
# add_cublasdx_cusolverdx_example
# ###############################################################
function(add_cublasdx_cusolverdx_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CUSOLVERDX_CUDA_ARCHITECTURES}"
    )
    # Debugging only supported with static library, so it's limited to Linux x86-64 platforms
    if(CUSOLVERDX_EXAMPLES_DEBUG_BUILD AND CUSOLVERDX_EXAMPLES_LINUX_X86_64)
        target_link_libraries(${EXAMPLE_TARGET} PRIVATE mathdx::cusolverdx)
    else()
        # Enables LTO, this is basically "$<DEVICE_LINK:-dlto>" but it requires CMake 3.25+
        set_target_properties(${EXAMPLE_TARGET} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        # Workaround for LTO support in CMAKE <3.25
        if(CMAKE_VERSION VERSION_LESS 3.25)
            link_lto_library_workaround(${EXAMPLE_TARGET} "${CUSOLVERDX_CUDA_ARCHITECTURES}")
        endif()
        target_link_libraries(${EXAMPLE_TARGET} PRIVATE mathdx::cusolverdx_fatbin)
    endif()
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            cusolverdx_compiler_options
            mathdx::cublasdx
            CUDA::cusolver
            common_example_objects
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
            # Required to support std::tuple in device code
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
    )
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# add_cusolverdx_example_link_to_lto_lib
# ###############################################################
function(add_cusolverdx_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CUSOLVERDX_CUDA_ARCHITECTURES}"
    )
    # Debugging only supported with static library, so it's limited to Linux x86-64 platforms
    if(CUSOLVERDX_EXAMPLES_DEBUG_BUILD AND CUSOLVERDX_EXAMPLES_LINUX_X86_64)
        target_link_libraries(${EXAMPLE_TARGET} PRIVATE mathdx::cusolverdx)
    else()
        # Enables LTO, this is basically "$<DEVICE_LINK:-dlto>" but it requires CMake 3.25+
        set_target_properties(${EXAMPLE_TARGET} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        # Workaround for LTO support in CMAKE <3.25
        if(CMAKE_VERSION VERSION_LESS 3.25)
            link_lto_library_workaround(${EXAMPLE_TARGET} "${CUSOLVERDX_CUDA_ARCHITECTURES}")
        endif()
        target_link_libraries(${EXAMPLE_TARGET} PRIVATE mathdx::cusolverdx_fatbin)
    endif()
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            cusolverdx_compiler_options
            CUDA::cusolver
            CUDA::cusparse
            common_example_objects
    )
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    )
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "EXAMPLE"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
    install_example(${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# add_cusolverdx_nvrtc_example
# ###############################################################
function(add_cusolverdx_nvrtc_example GROUP_TARGET EXAMPLE_NAME EXAMPLE_SOURCES)
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_TARGET ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})
    target_link_libraries(${EXAMPLE_TARGET}
        PRIVATE
            cusolverdx_compiler_options
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::nvrtc
            $<IF:$<TARGET_EXISTS:CUDA::nvJitLink>,CUDA::nvJitLink,${CUDAToolkit_LIBRARY_DIR}/libnvJitLink.so>
    )
    # For compatibility with CUDA 13+ & CMAKE 4+
    set(CUSOLVERDX_CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
    list(LENGTH CUSOLVERDX_CUDA_INCLUDE_DIRS CUDA_INCLUDE_LENGTH)

    if(${CUDA_INCLUDE_LENGTH} GREATER 1)
        list(GET CUSOLVERDX_CUDA_INCLUDE_DIRS 0 CUSOLVERDX_CUDA_INCLUDE_DIRS)
    else()
        set(CUSOLVERDX_CUDA_INCLUDE_DIRS "${CUSOLVERDX_CUDA_INCLUDE_DIRS}")
    endif()
    target_compile_definitions(${EXAMPLE_TARGET}
        PRIVATE
            CUDA_INCLUDE_DIR="${CUSOLVERDX_CUDA_INCLUDE_DIRS}"
            CUDA_CCCL_INCLUDE_DIR="${CUSOLVERDX_CUDA_INCLUDE_DIRS}/cccl/" # for CTK 13.0 support
            CUTLASS_INCLUDE_DIR="${cusolverdx_cutlass_INCLUDE_DIR}"
            COMMONDX_INCLUDE_DIR="${cusolverdx_commondx_INCLUDE_DIR}"
            CUSOLVERDX_INCLUDE_DIRS="${cusolverdx_INCLUDE_DIRS}"
            CUSOLVERDX_LIBRARY="$<TARGET_FILE:mathdx::cusolverdx>"
    )
    # Use fatbin for arm/aarch64 (.a file is x86-64 library)
    if((NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)"))
        target_compile_definitions(${EXAMPLE_TARGET}
            PRIVATE
                CUSOLVERDX_FATBIN="${cusolverdx_FATBIN}"
        )
        if(TARGET cusolverdx_fatbin)
            add_dependencies(${EXAMPLE_TARGET} cusolverdx_fatbin)
        endif()
    elseif(TARGET cusolverdx)
        add_dependencies(${EXAMPLE_TARGET} cusolverdx)
    endif()
    add_test(NAME ${EXAMPLE_NAME} COMMAND ${EXAMPLE_TARGET})
    set_tests_properties(${EXAMPLE_NAME}
        PROPERTIES
            LABELS "EXAMPLE"
    )
    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
    )
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})
    install_example(${EXAMPLE_TARGET})
endfunction()

# ###############################################################
# cuSOLVERDx Examples
# ###############################################################
add_custom_target(cusolverdx_examples)

# cuSOLVERDx NVRTC examples
add_cusolverdx_nvrtc_example(cusolverdx_examples "cusolverdx.example.nvrtc_potrs" 08_NVRTC/nvrtc_potrs.cpp)

# cuSOLVERDx introduction examples
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.posv_batched_block" 00_Introduction/posv_batched_block.cu)

# thread execution examples
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.potrf_batched_thread" 01_Linear_Solve/potrf_batched_thread.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.posv_batched_thread" 01_Linear_Solve/posv_batched_thread.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.gesv_batched_wo_pivot_thread" 01_Linear_Solve/gesv_batched_wo_pivot_thread.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.gesv_batched_partial_pivot_thread" 01_Linear_Solve/gesv_batched_partial_pivot_thread.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.gtsv_batched_wo_pivot_thread" 01_Linear_Solve/gtsv_batched_wo_pivot_thread.cu)

# block execution examples
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.potrf_block" 01_Linear_Solve/potrf_block.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.potrf_runtime_ld_block" 01_Linear_Solve/potrf_runtime_ld_block.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.getrf_wo_pivot_block" 01_Linear_Solve/getrf_wo_pivot_block.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.getrf_partial_pivot_block" 01_Linear_Solve/getrf_partial_pivot_block.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.gesv_batched_wo_pivot_block" 01_Linear_Solve/gesv_batched_wo_pivot_block.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.gesv_batched_partial_pivot_block" 01_Linear_Solve/gesv_batched_partial_pivot_block.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.gtsv_batched_wo_pivot_block" 01_Linear_Solve/gtsv_batched_wo_pivot_block.cu)

add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.gels_batched_thread" 02_Least_Squares/gels_batched_thread.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.gels_batched_block" 02_Least_Squares/gels_batched_block.cu)

add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.geqrf_batched_thread" 03_Orthogonal_Factors/geqrf_batched_thread.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.unmqr_batched_thread" 03_Orthogonal_Factors/unmqr_batched_thread.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.ungqr_batched_thread" 03_Orthogonal_Factors/ungqr_batched_thread.cu)

add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.geqrf_batched_block" 03_Orthogonal_Factors/geqrf_batched_block.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.unmqr_batched_block" 03_Orthogonal_Factors/unmqr_batched_block.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.ungqr_batched_block" 03_Orthogonal_Factors/ungqr_batched_block.cu)

add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.htev_batched_thread" 04_Symmetric_Eigenvalues/htev_batched_thread.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.heev_batched_thread" 04_Symmetric_Eigenvalues/heev_batched_thread.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.hegst_batched_thread" 04_Symmetric_Eigenvalues/hegst_batched_thread.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.htev_batched_block" 04_Symmetric_Eigenvalues/htev_batched_block.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.heev_batched_block" 04_Symmetric_Eigenvalues/heev_batched_block.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.hegst_batched_block" 04_Symmetric_Eigenvalues/hegst_batched_block.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.hegv_batched_block" 04_Symmetric_Eigenvalues/hegv_batched_block.cu)

add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.bdsvd_batched_thread" 05_Singular_Value_Decomposition/bdsvd_batched_thread.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.gesvd_batched_thread" 05_Singular_Value_Decomposition/gesvd_batched_thread.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.bdsvd_batched_block" 05_Singular_Value_Decomposition/bdsvd_batched_block.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.gesvd_batched_block" 05_Singular_Value_Decomposition/gesvd_batched_block.cu)

add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.trsm_batched_block" 07_BLAS/trsm_batched_block.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.trsm_batched_thread" 07_BLAS/trsm_batched_thread.cu)
add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.trsm_batched_thread_advanced" 07_BLAS/trsm_batched_thread_advanced.cu)

add_cusolverdx_example(cusolverdx_examples "cusolverdx.example.geqrf_batched_performance" 09_Performance/geqrf_batched_performance.cu)

# Cluster execution examples
add_cublasdx_cusolverdx_example(cusolverdx_examples "cusolverdx.example.potrf_batched_cluster" 01_Linear_Solve/potrf_batched_cluster.cu)

# cuBLASDx/cuSolverDx examples
message(STATUS "cuBLASDx is required, examples requiring cuBLASDx enabled")
# Example for handling larger matrices directly out of global memory
add_cublasdx_cusolverdx_example(cusolverdx_examples "cusolverdx.example.blocked_potrf" 10_Advanced/blocked_potrf.cu)
add_cublasdx_cusolverdx_example(cusolverdx_examples "cusolverdx.example.reg_least_squares" 10_Advanced/reg_least_squares.cu)
add_cublasdx_cusolverdx_example(cusolverdx_examples "cusolverdx.example.cga_blocked_potrf_right_looking" 10_Advanced/cga_blocked_potrf_right_looking.cu)
add_cublasdx_cusolverdx_example(cusolverdx_examples "cusolverdx.example.cga_tsqrhr" 10_Advanced/cga_tsqrhr.cu)
