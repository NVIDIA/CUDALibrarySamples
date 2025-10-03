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

# Check all needed variables are defined
if(NOT DEFINED CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "CMAKE_BUILD_TYPE is not defined")
endif()

function(build_cufft_device_api_lto_helper SRC_DIR BUILD_DIR)
    message(STATUS "Configuring and building 'cufft_device_api_lto_helper' immediately...")

    set(INTRODUCTION_LTO_COMMAND_ARGS
        -S ${SRC_DIR}
        -B ${BUILD_DIR}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    )

    # Set Toolchain file or CXX compiler for introduction_lto_helper project
    if(DEFINED CUFFTDX_LTO_TOOLCHAIN_FILE)
        list(APPEND INTRODUCTION_LTO_COMMAND_ARGS -DCMAKE_TOOLCHAIN_FILE=${CUFFTDX_LTO_TOOLCHAIN_FILE})
    else()
        if(NOT CMAKE_CROSSCOMPILING)
            if(DEFINED CMAKE_TOOLCHAIN_FILE AND NOT CMAKE_TOOLCHAIN_FILE STREQUAL "")
                list(APPEND INTRODUCTION_LTO_COMMAND_ARGS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
            else()
                list(APPEND INTRODUCTION_LTO_COMMAND_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
            endif()
        endif()
    endif()

    # Add cufft_ROOT only if it's set
    if (DEFINED cufft_ROOT)
        list(APPEND INTRODUCTION_LTO_COMMAND_ARGS -Dcufft_ROOT=${cufft_ROOT})
    endif()

    # Execute cmake command
    execute_process(COMMAND
        ${CMAKE_COMMAND} ${INTRODUCTION_LTO_COMMAND_ARGS}
        RESULT_VARIABLE CONFIG_RESULT
    )
    if(NOT CONFIG_RESULT EQUAL 0)
        message(FATAL_ERROR "Configuration of 'cufft_device_api_lto_helper' failed!")
    endif()

    # Execute build command
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${BUILD_DIR}
        RESULT_VARIABLE BUILD_RESULT
    )
    if(NOT BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Build of 'cufft_device_api_lto_helper' failed!")
    endif()
    message(STATUS "'cufft_device_api_lto_helper' has been built successfully.")

    # Register the files in cufft_device_api_lto_helper as part of the standard clean
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "${BUILD_DIR}")
endfunction()

function(run_cufft_device_api_lto_helper SRC_DIR BUILD_DIR OUTPUT_NAME CUDA_ARCHITECTURES)
    # Define the artifacts directory
    set(ARTIFACTS_DIR ${BUILD_DIR}/${OUTPUT_NAME}_artifacts)
    # Ensure artifacts directory exists
    file(MAKE_DIRECTORY ${ARTIFACTS_DIR})

    # Variable to track the success or failure of this function
    set(CUFFT_DEVICE_API_LTO_HELPER_RESULT TRUE)

    # Check if the artifacts are already generated
    if (NOT EXISTS "${ARTIFACTS_DIR}/lto_database.hpp.inc")
        # Remove files under artifacts directory
        file(GLOB FILES_TO_REMOVE "${ARTIFACTS_DIR}/*")
        file(REMOVE ${FILES_TO_REMOVE})

        # Check that executable exists
        if(NOT EXISTS ${BUILD_DIR}/cufft_device_api_lto_helper)
            build_cufft_device_api_lto_helper(${SRC_DIR} ${BUILD_DIR})
        endif()

        # Run helper executable with directory and arch arguments to generate artifacts
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E echo "Running cufft_device_api_lto_helper for OUTPUT_NAME=${OUTPUT_NAME}..."
            COMMAND ${CMAKE_COMMAND} -E env ${BUILD_DIR}/cufft_device_api_lto_helper ${ARTIFACTS_DIR} "--CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Execution of cufft_cufftdx_helper failed")
        endif()
    endif()

    # Add logic to dynamically create the object library
    file(GLOB LTO_FILES CONFIGURE_DEPENDS "${ARTIFACTS_DIR}/*.fatbin" "${ARTIFACTS_DIR}/*.ltoir")
    if(NOT LTO_FILES)
        message(WARNING "No .fatbin or .ltoir files were found for ${OUTPUT_NAME}.")
        set(CUFFT_DEVICE_API_LTO_HELPER_RESULT FALSE)
    else()
        message(STATUS "Generated files for ${OUTPUT_NAME}")

        # Link the generated files into an object library
        add_library(${OUTPUT_NAME}_lto_lib OBJECT IMPORTED GLOBAL)
        set_property(TARGET ${OUTPUT_NAME}_lto_lib PROPERTY IMPORTED_OBJECTS ${LTO_FILES})

        message(STATUS "Generated lib ${OUTPUT_NAME}_lto_lib")

        # Register the generated files as part of the standard clean
        file(GLOB GENERATED_FILES CONFIGURE_DEPENDS "${ARTIFACTS_DIR}/*")
        set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${GENERATED_FILES})
    endif()

    # Propagate the result to the caller
    set(CUFFT_DEVICE_API_LTO_HELPER_RESULT ${CUFFT_DEVICE_API_LTO_HELPER_RESULT} PARENT_SCOPE)
endfunction()