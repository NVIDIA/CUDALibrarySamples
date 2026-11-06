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

# Builds every sample in this directory against an nvJPEG2000 package, and installs them with
# their images and a script that runs them:
#
#   cmake -S . -B build -DNVJPEG2K_PATH=<package root>
#   cmake --build build --config Release
#   cmake --install build --config Release --prefix <dir>
#   <dir>/run_examples.sh <package root> <CUDA major>    (run_examples.cmd on Windows)

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(nvjpeg2k_examples LANGUAGES CXX CUDA)

set(NVJPEG2K_PATH "" CACHE PATH "nvJPEG2000 package root, holding include/ and lib/")

find_package(CUDAToolkit REQUIRED)

# Look only inside NVJPEG2K_PATH, so that another installed nvJPEG2000 is never picked up; the
# samples reuse this cache entry.
find_library(NVJPEG2K_LIB
    NAMES nvjpeg2k
    HINTS ${NVJPEG2K_PATH}/lib64/${CUDAToolkit_VERSION_MAJOR}
          ${NVJPEG2K_PATH}/lib/${CUDAToolkit_VERSION_MAJOR}
          ${NVJPEG2K_PATH}/lib
    NO_DEFAULT_PATH)
if(NOT NVJPEG2K_LIB)
    message(FATAL_ERROR "nvJPEG2000 library not found under NVJPEG2K_PATH=${NVJPEG2K_PATH}")
endif()
message(STATUS "nvJPEG2000 library: ${NVJPEG2K_LIB}")

add_subdirectory(nvJPEG2000-Decoder)
add_subdirectory(nvJPEG2000-Decoder-Pipelined)
add_subdirectory(nvJPEG2000-Decoder-Tile-Partial)
add_subdirectory(nvJPEG2000-Encoder)

install(TARGETS nvjpeg2k_decode nvjpeg2k_dec_pipelined nvjpeg2k_decode_tile_partial nvjpeg2k_encode
    RUNTIME DESTINATION bin)
install(DIRECTORY nvJPEG2000-Decoder/images/ DESTINATION images/decoder)
install(DIRECTORY nvJPEG2000-Encoder/images/ DESTINATION images/encoder)
if(WIN32)
    install(PROGRAMS run_examples.cmd DESTINATION .)
else()
    install(PROGRAMS run_examples.sh DESTINATION .)
endif()
