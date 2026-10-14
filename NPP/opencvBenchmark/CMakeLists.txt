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

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(opencvBenchmark LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(opencvBenchmark opencvBenchmark.cpp)
set_source_files_properties(opencvBenchmark.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(opencvBenchmark PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencvBenchmark PRIVATE ${OpenCV_LIBS} CUDA::cudart CUDA::nppc CUDA::nppif CUDA::nppig CUDA::nppicc)

install(TARGETS opencvBenchmark DESTINATION bin)
