# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Every FFT example validates its result against CUDA::cufft. The convolution
# examples keep the pointwise product on device through a cuFFT LTO store
# callback, built here. Use nvcc and bin2c from the toolkit that supplies
# cuFFT: CMAKE_CUDA_COMPILER is the Tile compiler, whose LTO IR is not
# necessarily readable by that cuFFT.
if(CUDAToolkit_ROOT)
    set(CUFFTDX_TILE_EXAMPLES_CUDA_BIN_DIR "${CUDAToolkit_ROOT}/bin")
else()
    set(CUFFTDX_TILE_EXAMPLES_CUDA_BIN_DIR "${CUDAToolkit_BIN_DIR}")
endif()
set(CUFFTDX_TILE_EXAMPLES_NVCC
    "${CUFFTDX_TILE_EXAMPLES_CUDA_BIN_DIR}/nvcc${CMAKE_EXECUTABLE_SUFFIX}")
set(CUFFTDX_TILE_EXAMPLES_BIN2C
    "${CUFFTDX_TILE_EXAMPLES_CUDA_BIN_DIR}/bin2c${CMAKE_EXECUTABLE_SUFFIX}")
if(NOT EXISTS "${CUFFTDX_TILE_EXAMPLES_NVCC}")
    message(FATAL_ERROR "Could not find the cuFFT toolkit compiler at ${CUFFTDX_TILE_EXAMPLES_NVCC}")
endif()
if(NOT EXISTS "${CUFFTDX_TILE_EXAMPLES_BIN2C}")
    message(FATAL_ERROR "Could not find bin2c at ${CUFFTDX_TILE_EXAMPLES_BIN2C}")
endif()
set(CUFFTDX_TILE_EXAMPLES_LTO_CALLBACK_FATBIN
    "${CMAKE_CURRENT_BINARY_DIR}/cufft_lto_store_multiply.fatbin")
set(CUFFTDX_TILE_EXAMPLES_LTO_CALLBACK_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/cufft_lto_store_multiply.h")
set(CUFFTDX_TILE_EXAMPLES_LTO_ARCHITECTURES)
foreach(architecture IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX MATCH "^[0-9]+" architecture_number "${architecture}")
    if(architecture_number)
        list(APPEND CUFFTDX_TILE_EXAMPLES_LTO_ARCHITECTURES
            "-gencode=arch=compute_${architecture_number},code=lto_${architecture_number}")
    elseif(architecture MATCHES "^(all|all-major|native)$")
        list(LENGTH CMAKE_CUDA_ARCHITECTURES architecture_count)
        if(NOT architecture_count EQUAL 1)
            message(FATAL_ERROR "CUDA architecture '${architecture}' cannot be combined with other architectures")
        endif()
        list(APPEND CUFFTDX_TILE_EXAMPLES_LTO_ARCHITECTURES "-arch=${architecture}" "-dlto")
    else()
        message(FATAL_ERROR "Unsupported CUDA architecture '${architecture}' for the cuFFT LTO callback")
    endif()
endforeach()
if(NOT CUFFTDX_TILE_EXAMPLES_LTO_ARCHITECTURES)
    message(FATAL_ERROR "A CUDA architecture is required for the cuFFT LTO callback")
endif()
add_custom_command(
    OUTPUT "${CUFFTDX_TILE_EXAMPLES_LTO_CALLBACK_HEADER}"
    BYPRODUCTS "${CUFFTDX_TILE_EXAMPLES_LTO_CALLBACK_FATBIN}"
    COMMAND "${CUFFTDX_TILE_EXAMPLES_NVCC}" ${CUFFTDX_TILE_EXAMPLES_LTO_ARCHITECTURES} -dc -fatbin
            "${CMAKE_CURRENT_LIST_DIR}/../common/cufft_lto_store_multiply.cu"
            -o "${CUFFTDX_TILE_EXAMPLES_LTO_CALLBACK_FATBIN}"
    # bin2c writes to stdout; the script captures it without relying on
    # platform-specific shell redirection.
    COMMAND "${CMAKE_COMMAND}"
            -DBIN2C=${CUFFTDX_TILE_EXAMPLES_BIN2C}
            -DINPUT=${CUFFTDX_TILE_EXAMPLES_LTO_CALLBACK_FATBIN}
            -DOUTPUT=${CUFFTDX_TILE_EXAMPLES_LTO_CALLBACK_HEADER}
            -P "${CMAKE_CURRENT_LIST_DIR}/../common/bin2c.cmake"
    DEPENDS
        "${CMAKE_CURRENT_LIST_DIR}/../common/cufft_lto_store_multiply.cu"
        "${CMAKE_CURRENT_LIST_DIR}/../common/bin2c.cmake"
    VERBATIM)
add_custom_target(cufft_lto_callback
    DEPENDS "${CUFFTDX_TILE_EXAMPLES_LTO_CALLBACK_HEADER}")

function(cufftdx_tile_enable_cufft_reference target)
    # The mixed-precision example is the only one without the LTO store callback.
    if(NOT target STREQUAL "mixed_precision_fft_1d")
        target_include_directories("${target}" PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        add_dependencies("${target}" cufft_lto_callback)
    endif()
    target_link_libraries("${target}" PRIVATE CUDA::cufft)
endfunction()
