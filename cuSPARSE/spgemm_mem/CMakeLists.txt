# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(ROUTINE spgemm_mem)

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