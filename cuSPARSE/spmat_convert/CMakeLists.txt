# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.9)

set(ROUTINE spmat_convert)

project("${ROUTINE}_example"
        DESCRIPTION  "GPU-Accelerated Sparse Linear Algebra"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cusparse/index.html"
        LANGUAGES    C CUDA)

add_executable(${ROUTINE}_example)

target_sources(${ROUTINE}_example
    PUBLIC ${PROJECT_SOURCE_DIR}/${ROUTINE}_example.c
)

target_include_directories(${ROUTINE}_example
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(${ROUTINE}_example
    PUBLIC cudart cusparse
)
