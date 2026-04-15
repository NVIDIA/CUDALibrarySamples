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

cmake_minimum_required(VERSION 3.26.4)
project(nvcompdx_examples LANGUAGES CXX CUDA)

if(PROJECT_IS_TOP_LEVEL)
    # Enable running examples with CTest
    enable_testing()

    set(DX_PROJECT_CMAKE_VAR_PREFIX "NVCOMPDX")
    set(DX_PROJECT_FULL_NAME "nvCOMPDx")

    # ******************************************
    # nvCOMPDx Install Dirs
    # ******************************************
    include(GNUInstallDirs)
    set(NVCOMPDX_EXAMPLES_BIN_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}/example/")

    # Include custom CMake modules/scripts
    list(APPEND CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/
    )

    if(NOT MSVC)
        set(NVCOMPDX_CUDA_CXX_FLAGS "${NVCOMPDX_CUDA_CXX_FLAGS} -Wall -Wextra")
        set(NVCOMPDX_CUDA_CXX_FLAGS "${NVCOMPDX_CUDA_CXX_FLAGS} -fno-strict-aliasing")
        set(NVCOMPDX_CUDA_CXX_FLAGS "${NVCOMPDX_CUDA_CXX_FLAGS} -Wno-deprecated-declarations")
        if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch64")
            set(NVCOMPDX_CUDA_CXX_FLAGS "${NVCOMPDX_CUDA_CXX_FLAGS} -m64")
        endif ()

        if((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR (CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*"))
            # Print error/warnings numbers
            set(NVCOMPDX_CUDA_CXX_FLAGS "${NVCOMPDX_CUDA_CXX_FLAGS} --display_error_number")
        endif()

        if((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR (CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*"))
            # if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 23.1.0)
                set(NVCOMPDX_CUDA_CXX_FLAGS "${NVCOMPDX_CUDA_CXX_FLAGS} --diag_suppress1")
                set(NVCOMPDX_CUDA_CXX_FLAGS "${NVCOMPDX_CUDA_CXX_FLAGS} --diag_suppress111")
                set(NVCOMPDX_CUDA_CXX_FLAGS "${NVCOMPDX_CUDA_CXX_FLAGS} --diag_suppress177")
                set(NVCOMPDX_CUDA_CXX_FLAGS "${NVCOMPDX_CUDA_CXX_FLAGS} --diag_suppress941")
            # endif()
        endif()

        # because -Wall is passed, all diagnostic ignores of -Wunkown-pragmas
        # are ignored, which leads to unlegible CuTe compilation output
        # fixed in GCC13 https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53431
        # but still not widely adopted
        set(NVCOMPDX_CUDA_CXX_FLAGS "${NVCOMPDX_CUDA_CXX_FLAGS} -Wno-unknown-pragmas")
    else()
        add_definitions(-D_CRT_SECURE_NO_WARNINGS)
        add_definitions(-D_CRT_NONSTDC_NO_WARNINGS)
        add_definitions(-D_SCL_SECURE_NO_WARNINGS)
        add_definitions(-DNOMINMAX)
        set(NVCOMPDX_CUDA_CXX_FLAGS "${NVCOMPDX_CUDA_CXX_FLAGS} /W3") # Warning level
        set(NVCOMPDX_CUDA_CXX_FLAGS "${NVCOMPDX_CUDA_CXX_FLAGS} /WX") # All warnings are errors
        set(NVCOMPDX_CUDA_CXX_FLAGS "${NVCOMPDX_CUDA_CXX_FLAGS} /Zc:__cplusplus") # Enable __cplusplus macro
    endif()

    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0.0)
        message(FATAL_ERROR "nvCOMPDx examples require CUDA 13.0.0 or newer, but the compiler version is ${CMAKE_CUDA_COMPILER_VERSION}")
    endif()

    # Global CXX flags/options
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NVCOMPDX_CUDA_CXX_FLAGS}")

    # Global CUDA CXX flags/options
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all") # Compress all fatbins
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number") # Show error/warning numbers
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${NVCOMPDX_CUDA_CXX_FLAGS}\"")

    # Targeted CUDA Architectures, see https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES
    set(NVCOMPDX_CUDA_ARCHITECTURES 75-real;80-real CACHE
        STRING "List of targeted nvCOMPDx CUDA architectures, for example \"75-real;80-real\""
    )

    # Remove unsupported architectures
    list(REMOVE_ITEM NVCOMPDX_CUDA_ARCHITECTURES 30;32;35;37;50;52;53;60;61;62;70;72)
    list(REMOVE_ITEM NVCOMPDX_CUDA_ARCHITECTURES 30-real;32-real;35-real;37-real;50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real)
    list(REMOVE_ITEM NVCOMPDX_CUDA_ARCHITECTURES 30-virtual;32-virtual;35-virtual;37-virtual;50-virtual;52-virtual;53-virtual;60-virtual;61-virtual;62-virtual;70-virtual;72-virtual)
    message(STATUS "Targeted nvCOMPDx CUDA Architectures: ${NVCOMPDX_CUDA_ARCHITECTURES}")

    file(GLOB INSTALLED_MATHDX_DIRECTORIES LIST_DIRECTORIES TRUE "/opt/nvidia/mathdx/*")

    find_package(mathdx REQUIRED COMPONENTS nvcompdx CONFIG
        PATHS
            "${PROJECT_SOURCE_DIR}/../.." # example/nvcompdx
            "${INSTALLED_MATHDX_DIRECTORIES}"
    )
    set(NVCOMPDX_MIN_VERSION 0.1.0)
    if(nvcompdx_VERSION VERSION_LESS ${NVCOMPDX_MIN_VERSION})
        message(FATAL_ERROR "nvCOMPDx version (${nvcompdx_VERSION}) is less than ${NVCOMPDX_MIN_VERSION}")
    endif()
endif()

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Enable testing only for selected architectures
foreach(CUDA_ARCH ${NVCOMPDX_CUDA_ARCHITECTURES})
    # Extract SM from SM-real/SM-virtual
    string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
    list(GET CUDA_ARCH_LIST 0 ARCH)
    # Remove "a"
    string(REPLACE "a" "" ARCH "${ARCH}")
    # Remove "f"
    string(REPLACE "f" "" ARCH "${ARCH}")
    add_compile_definitions(NVCOMPDX_EXAMPLE_ENABLE_SM_${ARCH})
endforeach()

# ###############################################################
# Test file generation
# ###############################################################
add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/random_file_64KiB.bin"
    COMMAND dd if=/dev/urandom of="${CMAKE_BINARY_DIR}/random_file_64KiB.bin" bs=64KiB count=1
    COMMENT "Generating a 64 KiB random file"
)

add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/random_file_100MiB.bin"
    COMMAND dd if=/dev/urandom of="${CMAKE_BINARY_DIR}/random_file_100MiB.bin" bs=100MiB count=1
    COMMENT "Generating a 100 MiB random file"
)

add_custom_target(generate_test_files
    DEPENDS
        ${CMAKE_BINARY_DIR}/random_file_64KiB.bin
        ${CMAKE_BINARY_DIR}/random_file_100MiB.bin
)

# ###############################################################
# add_nvcompdx_example
# ###############################################################
function(add_nvcompdx_example_impl GROUP_TARGET EXAMPLE_SOURCES CTEST_ARGUMENTS)
    # General ---------------------------------------------
    list(GET EXAMPLE_SOURCES 0 EXAMPLE_MAIN_SOURCE)
    get_filename_component(EXAMPLE_FILENAME ${EXAMPLE_MAIN_SOURCE} NAME_WE)
    get_filename_component(EXAMPLE_DIRECTORY "${EXAMPLE_MAIN_SOURCE}" DIRECTORY)

    # Compose target name: dir_filename
    set(EXAMPLE_TARGET "${EXAMPLE_DIRECTORY}_${EXAMPLE_FILENAME}")

    # Compose user-readable example name
    set(EXAMPLE_NAME "nvCOMPDx.example.${EXAMPLE_DIRECTORY}.${EXAMPLE_FILENAME}")

    set_source_files_properties(${EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)
    add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCES})

    set_target_properties(${EXAMPLE_TARGET}
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${NVCOMPDX_CUDA_ARCHITECTURES}"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_DIRECTORY}"
            OUTPUT_NAME "${EXAMPLE_FILENAME}"
    )

    if(NOT(CMAKE_BUILD_TYPE STREQUAL "DEBUG") AND NOT(CMAKE_BUILD_TYPE STREQUAL "Debug"))
        set_property(TARGET ${EXAMPLE_TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()

    target_compile_options(${EXAMPLE_TARGET}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
            # Required to support std::tuple in device code
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
    )

    target_link_libraries(${EXAMPLE_TARGET} PRIVATE CUDA::cudart_static)

    if (NOT MSVC)
        add_dependencies(${EXAMPLE_TARGET} generate_test_files)
    endif()
    add_dependencies(${GROUP_TARGET} ${EXAMPLE_TARGET})

    # LZ4 specialization ----------------------------------
    if (LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
        target_include_directories(${EXAMPLE_TARGET} PRIVATE ${LZ4_INCLUDE_DIR})
        target_link_libraries(${EXAMPLE_TARGET} PRIVATE ${LZ4_LIBRARY})
    endif()

    # NVRTC specialization --------------------------------
    string(FIND ${EXAMPLE_TARGET} "_nvrtc_" _NVRTC_POS)
    set(NVRTC_USED OFF)
    if (NOT (_NVRTC_POS EQUAL -1))
      set(NVRTC_USED ON)
    endif()
    if (${NVRTC_USED})
        target_include_directories(${EXAMPLE_TARGET}
            PRIVATE
                ${nvcompdx_INCLUDE_DIRS}
                ${nvcompdx_commondx_INCLUDE_DIR}
        )
        target_link_libraries(${EXAMPLE_TARGET}
            PRIVATE
                CUDA::cuda_driver
                CUDA::nvrtc
                CUDA::nvJitLink
        )
        target_compile_definitions(${EXAMPLE_TARGET}
            PRIVATE
                CUDAToolkit_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}"
                COMMONDX_INCLUDE_DIR="${nvcompdx_commondx_INCLUDE_DIR}"
                NVCOMPDX_INCLUDE_DIRS="${nvcompdx_INCLUDE_DIRS}"
                NVCOMPDX_LIBRARY="$<TARGET_FILE:mathdx::nvcompdx>"
        )
        # Use fatbin for arm/aarch64 (given the .a file is an x86-64 library)
        if((NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)"))
            target_compile_definitions(${EXAMPLE_TARGET}
                PRIVATE
                    NVCOMPDX_FATBIN="${nvcompdx_FATBIN}"
            )
        endif()
    else()
        # Use the .a library for Debug mode
        if((CMAKE_BUILD_TYPE STREQUAL "DEBUG") OR (CMAKE_BUILD_TYPE STREQUAL "Debug"))
            target_link_libraries(${EXAMPLE_TARGET} PRIVATE mathdx::nvcompdx)
        else()
            target_link_libraries(${EXAMPLE_TARGET} PRIVATE mathdx::nvcompdx_fatbin)
        endif()
    endif()

    # Workaround for LTO support in CMAKE <3.25 -----------
    if(CMAKE_VERSION VERSION_LESS 3.25)
        if (NOT ${NVRTC_USED})
            target_link_options(${EXAMPLE_TARGET}
                PRIVATE
                    $<DEVICE_LINK:-dlto>
            )
        endif()
        target_compile_options(${EXAMPLE_TARGET}
            PRIVATE
                "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-rdc=true>"
        )
        foreach(CUDA_ARCH ${NVCOMPDX_CUDA_ARCHITECTURES})
            # Extract SM from SM-real/SM-virtual
            string(REPLACE "-" ";" CUDA_ARCH_LIST ${CUDA_ARCH})
            list(GET CUDA_ARCH_LIST 0 ARCH)
            # Remove "a"
            string(REPLACE "a" "" ARCH "${ARCH}")
            # Remove "f"
            string(REPLACE "f" "" ARCH "${ARCH}")
            target_compile_options(${EXAMPLE_TARGET}
                PRIVATE
                    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--generate-code arch=compute_${ARCH},code=lto_${ARCH}>"
            )
        endforeach()
    endif()

    # Testing ---------------------------------------------
    # Disable NVRTC example testing for NVC++ compiler, because of libcu++/NRTC/NVC++ bug
    if((NOT NVRTC_USED) OR
       (NVRTC_USED AND (NOT ((CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") OR (CMAKE_CXX_COMPILER MATCHES ".*nvc\\+\\+.*")))))
        add_test(NAME ${EXAMPLE_NAME} COMMAND $<PATH:RELATIVE_PATH,$<TARGET_FILE:${EXAMPLE_TARGET}>,${CMAKE_CURRENT_BINARY_DIR}> ${CTEST_ARGUMENTS})
        set_tests_properties(${EXAMPLE_NAME}
            PROPERTIES
                LABELS "EXAMPLE"
        )
    endif()

    # Installation ----------------------------------------
    install(TARGETS ${EXAMPLE_TARGET}
        RUNTIME DESTINATION ${NVCOMPDX_EXAMPLES_BIN_INSTALL_DIR}/${EXAMPLE_DIRECTORY}
        COMPONENT Examples
    )
endfunction()

function(add_nvcompdx_example GROUP_TARGET EXAMPLE_SOURCES CTEST_ARGUMENTS)
    add_nvcompdx_example_impl("${GROUP_TARGET}" "${EXAMPLE_SOURCES}" "${CTEST_ARGUMENTS}")
endfunction()

# ###############################################################
# nvCOMPDx Examples
# ###############################################################

add_custom_target(nvcompdx_examples)
find_path(LZ4_INCLUDE_DIR NAMES lz4.h)
find_library(LZ4_LIBRARY NAMES lz4)
set(EXAMPLE_FILE_FOLDER $<PATH:RELATIVE_PATH,${CMAKE_BINARY_DIR},${CMAKE_CURRENT_BINARY_DIR}>)

# LZ4 nvRTC example
# Needs the LZ4 library, NVRTC, and nvJitLink
if (LZ4_INCLUDE_DIR AND LZ4_LIBRARY AND TARGET CUDA::nvrtc AND TARGET CUDA::nvJitLink)
    add_nvcompdx_example(nvcompdx_examples 05_lz4_cpu_and_nvrtc/lz4_cpu_compression_nvrtc_decompression.cu "-f;${EXAMPLE_FILE_FOLDER}/random_file_100MiB.bin")
endif()

# LZ4 introductory example
add_nvcompdx_example(nvcompdx_examples 01_introduction/lz4_gpu_compression_introduction.cu "-f;${EXAMPLE_FILE_FOLDER}/random_file_64KiB.bin;-o;${EXAMPLE_FILE_FOLDER}/random_file_64KiB.lz4")
# LZ4 GPU compression and decompression example
add_nvcompdx_example(nvcompdx_examples 02_lz4_gpu/lz4_gpu_compression_decompression.cu "-f;${EXAMPLE_FILE_FOLDER}/random_file_64KiB.bin")

# LZ4 CPU example(s)
if (LZ4_INCLUDE_DIR AND LZ4_LIBRARY)
    add_nvcompdx_example(nvcompdx_examples 03_lz4_gpu_and_cpu/lz4_cpu_compression_gpu_decompression.cu "-f;${EXAMPLE_FILE_FOLDER}/random_file_100MiB.bin")
    add_nvcompdx_example(nvcompdx_examples 03_lz4_gpu_and_cpu/lz4_gpu_compression_cpu_decompression.cu "-f;${EXAMPLE_FILE_FOLDER}/random_file_100MiB.bin")
else()
    message(WARNING "Skipping building LZ4 CPU examples, as no LZ4 library was found.")
endif()

# ANS example(s)
add_nvcompdx_example(nvcompdx_examples 04_ans_gpu/ans_gpu_compression_decompression.cu "-t;uint8;-f;${EXAMPLE_FILE_FOLDER}/random_file_100MiB.bin")
add_nvcompdx_example(nvcompdx_examples 04_ans_gpu/ans_gpu_decompression_reduction.cu "")
