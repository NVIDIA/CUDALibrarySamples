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



#include <cassert>
#include <iostream>
#include <nvMatmulHeuristics/nvMatmulHeuristics.h>

#include "sampleHelpers.h"

/**
 * This sample shows the best way to use nvMatmulHeuristics to get a GEMM kernel configuration.
 */

int main() {
    nvmmhHandle_t handle = nullptr;
    // In case the user does not want to manage handles, a nullptr can be used as a handle which will use an internal global handle.
    // This is only recommended when it is known that no other library is using nvMatmulHeuristics
    NVMMH_CHECK(nvMatmulHeuristicsCreate(&handle));


    // We can create a hardware descriptor to specify what hardware nvMatmulHeuristics will target.
    // The hardware descriptor is optional, the user can pass a nullptr which will cause nvMatmulHeuristics to target the current GPU, if there's one.
    nvmmhHardwareDescriptor_t descr = nullptr;
    NVMMH_CHECK(nvMatmulHeuristicsHardwareDescriptorCreate(&descr));
    // Here we set A100 SXM properties.
    NVMMH_CHECK(nvMatmulHeuristicsHardwareDescriptorSetPredefinedGpu(descr, NVMMH_NVGPU_A100_SXM_80GB));

    // See header for precision string convention. HSH means FP16 A/B, FP32 computation and FP16 C/D.
    const char* precision = "HSH";
    constexpr int kernelCount = 8;
    constexpr auto layout = NVMMH_MATMUL_LAYOUT_TN_ROW_MAJOR;
    constexpr auto target = NVMMH_TARGET_CUTLASS;
    // Some matmul problem
    constexpr nvmmhMatmulProblem_t p = {
            .M = 128,
            .N = 128,
            .K = 400000,
            .batchSize = 1,
            .matmulLayout = static_cast<uint8_t>(layout),
    };

    // Loads the internal discovery data (silicon performance scans) to tune nvMatmulHeuristics to the actual kernel implementation.
    // This allows for a quick/cold start. If you are using customized kernels, or kernels not included in nvMatmulHeuristics, you need to run the discovery manually.
    if (nvMatmulHeuristicsLoadInternalDiscoverySet(handle, precision, target, layout, descr) != NVMMH_STATUS_SUCCESS) {
        std::cout << "Please check sample #2 to see how to pass the data to nvMatmulHeuristics manually." << std::endl;
        std::cout << "We can continue without the tuning data." << std::endl;
    }

    nvmmhKernelConfiguration_t configs[kernelCount];
    // NVMMH_FLAG_PERF_MODEL_BASED_AUTO_TUNING is the recommended flag.
    const int count = nvMatmulHeuristicsGetGemmConfig(handle, precision, NVMMH_FLAG_PERF_MODEL_BASED_AUTO_TUNING, target, &p, configs, kernelCount, descr);

    // nvMatmulHeuristics might return fewer kernels than requested.
    assert(count <= kernelCount);

    // Printing the kernels
    for (int i = 0; i < count; ++i) {
        // nvMatmulHeuristicsEstimateRuntime might use a different path that the heuristic internal ordering.
        const double runtime = nvMatmulHeuristicsEstimateRuntime(handle, precision, target, &p, &configs[i], descr) * 1000.;
        std::cout << '[' << i + 1 << "] " << to_string(configs[i]) << ", estimated GEMM runtime: " << runtime << " ms" << std::endl;
    }

    // Freeing memory. We pass a pointer, so nvMatmulHeuristics can set it to nullptr, to avoid use-after-free/double-free,
    NVMMH_CHECK(nvMatmulHeuristicsHardwareDescriptorDestroy(&descr));
    NVMMH_CHECK(nvMatmulHeuristicsDestroy(&handle));
}