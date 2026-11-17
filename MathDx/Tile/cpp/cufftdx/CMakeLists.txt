# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cufft_reference.cmake")

function(add_cufftdx_tile_example source)
    cmake_path(GET source STEM target)

    mathdx_add_tile_example("${target}" "${source}")
    target_include_directories("${target}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/common")
    target_link_libraries("${target}" PRIVATE mathdx_tile::cufftdx CUDA::cudart)
    cufftdx_tile_enable_cufft_reference("${target}")
endfunction()

file(GLOB example_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
foreach(example_source IN LISTS example_sources)
    add_cufftdx_tile_example("${example_source}")
endforeach()
