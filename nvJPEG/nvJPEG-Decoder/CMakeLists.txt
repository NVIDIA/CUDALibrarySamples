# SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.17)
project(nvjpegDecoder LANGUAGES CXX)

find_package(CUDAToolkit 13.0 REQUIRED)
add_executable(nvjpegDecoder nvjpegDecoder.cpp)
target_compile_features(nvjpegDecoder PRIVATE cxx_std_17)
target_link_libraries(nvjpegDecoder PRIVATE CUDA::nvjpeg CUDA::cudart)
if(MSVC)
    target_compile_definitions(nvjpegDecoder PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
install(TARGETS nvjpegDecoder DESTINATION bin)

option(NVJPEG_DECODER_BUILD_TESTS "Enable GPU integration tests (requires Python and Pillow)" OFF)
if(NVJPEG_DECODER_BUILD_TESTS)
    enable_testing()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME decoder COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_decoder.py $<TARGET_FILE:nvjpegDecoder>)
endif()
