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

cmake_minimum_required(VERSION 3.9)

set(ROUTINE spmv_symmetric)

project("${ROUTINE}_example"
        DESCRIPTION  "GPU-Accelerated Sparse Linear Algebra"
        LANGUAGES    C CUDA)

add_executable(${ROUTINE}_example)

set(CUSPARSE_INCLUDE_DIR "" CACHE PATH
    "Optional path to cuSPARSE headers when testing a source build")
set(CUSPARSE_LIBRARY_DIR "" CACHE PATH
    "Optional path to cuSPARSE libraries when testing a source build")

target_sources(${ROUTINE}_example
    PUBLIC ${PROJECT_SOURCE_DIR}/${ROUTINE}_example.c
)

target_include_directories(${ROUTINE}_example
    PUBLIC
        ${CUSPARSE_INCLUDE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

if(CUSPARSE_LIBRARY_DIR)
    find_library(CUSPARSE_LIBRARY
        NAMES cusparse
        PATHS ${CUSPARSE_LIBRARY_DIR}
        NO_DEFAULT_PATH
    )
    if(NOT CUSPARSE_LIBRARY)
        message(FATAL_ERROR "cuSPARSE library not found in ${CUSPARSE_LIBRARY_DIR}")
    endif()
    set_target_properties(${ROUTINE}_example PROPERTIES
        BUILD_RPATH "${CUSPARSE_LIBRARY_DIR}"
    )
else()
    set(CUSPARSE_LIBRARY cusparse)
endif()

target_link_libraries(${ROUTINE}_example
    PUBLIC cudart ${CUSPARSE_LIBRARY}
)
