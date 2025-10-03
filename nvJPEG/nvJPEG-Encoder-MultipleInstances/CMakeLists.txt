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

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)


option(CROSS_COMPILE_AARCH64 "Cross compile for ARM64" OFF)

if(CROSS_COMPILE_AARCH64)
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    set(CMAKE_CUDA_HOST_COMPILER aarch64-linux-gnu-g++)
endif()

project(nvJPEGEncMultipleInstances LANGUAGES CXX CUDA)

find_package(Threads REQUIRED)
find_package(CUDAToolkit 12.9 REQUIRED)

add_executable(nvJPEGEncMultipleInstances "nvJPEGEncMultipleInstances.cpp")

target_compile_features(nvJPEGEncMultipleInstances PRIVATE
    cxx_std_17
    cuda_std_17)

target_link_libraries(nvJPEGEncMultipleInstances PUBLIC
    CUDA::nvjpeg
    CUDA::cudart_static
    Threads::Threads)