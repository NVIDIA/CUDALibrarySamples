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


#include <nvMatmulHeuristics/nvMatmulHeuristics.h>

#include "sampleHelpers.h"

/**
 * Small example that shows nvMatmulHeuristics "discovery" aka. tuning flow.
 */

constexpr int MAX_DISCOVERY_PROBLEMS = 1024;
constexpr nvmmhTarget_t TARGET = NVMMH_TARGET_TRITON;
constexpr auto PRECISION = "TST";
constexpr nvmmhMatmulLayout_t LAYOUT = NVMMH_MATMUL_LAYOUT_TN_ROW_MAJOR;

int main() {
    nvmmhHardwareDescriptor_t hwDescr = nullptr;

    nvmmhDependencyConfiguration_t hasCuda{};
    NVMMH_CHECK(nvMatmulHeuristicsGetDependencyConfiguration(&hasCuda));

    nvmmhHandle_t handle{};
    NVMMH_CHECK(nvMatmulHeuristicsCreate(&handle));


    if (!hasCuda) {
        NVMMH_CHECK(nvMatmulHeuristicsHardwareDescriptorCreate(&hwDescr));
        NVMMH_CHECK(nvMatmulHeuristicsHardwareDescriptorSetPredefinedGpu(hwDescr, NVMMH_NVGPU_A100_SXM_80GB));
    }

    nvmmhMatmulProblem_t problems[MAX_DISCOVERY_PROBLEMS] = {};
    nvmmhKernelConfiguration_t configs[MAX_DISCOVERY_PROBLEMS] = {};
    float runtimes[MAX_DISCOVERY_PROBLEMS] = {};

    const unsigned count = nvMatmulHeuristicsGetDiscoverySet(handle, PRECISION, TARGET, LAYOUT, problems, configs, MAX_DISCOVERY_PROBLEMS, hwDescr);


    for (int i = 0; i < count; ++i) {
        std::cout << "[" << i + 1 << "] Problem: " << to_string(problems[i]) << ", Config: " << to_string(configs[i]) << std::endl;

        /* Place here a benchmark of your GEMM kernel implementation. As a placeholder in the sample we will just use the internal timing model  */
        runtimes[i] = nvMatmulHeuristicsEstimateRuntime(handle, PRECISION, TARGET, problems + i, configs + i, hwDescr);
    }


    // Will update the handle to map the tuning profile to the selected backend, layout, precision and current GPU ID or context ID if called within a green context.
    NVMMH_CHECK(nvMatmulHeuristicsCommitDiscoveryResults(handle, PRECISION, TARGET, LAYOUT, problems, configs, runtimes, count, hwDescr));

    /* All the calls that are done after this API will use the tuning data if the handle, layout, precision, backend and GPU context match. */

    if (hwDescr) { NVMMH_CHECK(nvMatmulHeuristicsHardwareDescriptorDestroy(&hwDescr)); }
    NVMMH_CHECK(nvMatmulHeuristicsDestroy(&handle));
}