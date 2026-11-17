# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.27)

# Native is a convenient default for the standalone examples.  Set this before
# project() so CMake detects the GPU while identifying the CUDA compiler.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(mathdx_tile_examples LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   /W3 /WX")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /W3 /WX")
else()
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -Wall -Werror -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Werror -Wextra")
endif()

include(CTest)

# The reference checks link cuBLAS, cuSolver, and cuFFT. When the Tile
# compiler's own CUDA package does not ship them, point CUDAToolkit_ROOT at a
# full toolkit.
find_package(CUDAToolkit REQUIRED)
find_package(mathdx_tile CONFIG REQUIRED COMPONENTS cublasdx cufftdx cusolverdx nvcompdx)

function(mathdx_add_tile_example target source)
    add_executable(${target} "${source}")
    # Helpers shared by the cuFFTDx and cuSolverDx examples live at this
    # directory's root.
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
    install(TARGETS ${target} RUNTIME DESTINATION bin)
    add_test(NAME ${target} COMMAND ${target})
endfunction()

add_subdirectory(cusolverdx)
add_subdirectory(cufftdx)
add_subdirectory(nvcompdx)
