/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#pragma once

#include <nvMatmulHeuristics/nvMatmulHeuristics.h>

#include <cassert>
#include <iostream>
#include <string>


/**
 * Converts a kernel config into a string
 * @param config 
 * @return 
 */
static std::string to_string(const nvmmhKernelConfiguration_t& config) {
    char buf[1024];
    const int count = nvMatmulHeuristicsKernelConfigurationGetString(&config, buf, sizeof(buf));
    assert(count <= sizeof(buf));
    if (count == 0) { return ""; }
    return buf;
}

/**
 * Converts a matmul problem into a string
 * @param config 
 * @return 
 */
static std::string to_string(const nvmmhMatmulProblem_t& config) {
    char buf[1024];
    const int count = nvMatmulHeuristicsMatmulProblemGetString(&config, buf, sizeof(buf));
    assert(count <= sizeof(buf));
    if (count == 0) { return ""; }
    return buf;
}

/**
 * Printing nvmmh version at app init.
 */
static inline bool print_version_on_init = [] {
    char buf[1024];
    const int count = nvMatmulHeuristicsGetVersionString(buf, sizeof(buf));
    assert(count <= sizeof(buf));
    if (count == 0) { return false; }
    std::cout << "nvMatmulHeuristics version " << buf << '\n' << std::endl;
    return true;
}();

template<typename T> static constexpr T divUp(const T& a, const T& b) { return (a + b - 1) / b; }