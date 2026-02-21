# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(NPPCannySimple LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Find packages
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})

# Simple example
add_executable(nppCannySimple npp_canny_simple.cpp)

target_link_libraries(nppCannySimple
    ${OpenCV_LIBS}
    CUDA::nppif
    CUDA::nppc
    CUDA::cudart
)

# Set CUDA architectures (Volta, Ampere, Hopper)
set_target_properties(nppCannySimple PROPERTIES
    CUDA_ARCHITECTURES "70;80;86;90"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "NPP Canny Simple - Build configured")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "CUDA: ${CUDAToolkit_VERSION}")
