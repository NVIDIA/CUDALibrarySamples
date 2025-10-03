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

cmake_minimum_required(VERSION 3.18)

# Define project
project(lto_helper LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Possible variables that can be set externally
# cufft_ROOT: points to the cuFFT library root directory

find_package(cufft 11.5.0 EXACT REQUIRED CONFIG
    PATHS
        "${PROJECT_SOURCE_DIR}/../../../../cufft"
        "/opt/cufft"
)

# Define the helper executable
add_executable(cufftdx_cufft_lto_helper cufftdx_cufft_lto_helper.cpp)
target_link_libraries(cufftdx_cufft_lto_helper
    PRIVATE
        cufft::cufft_static
)