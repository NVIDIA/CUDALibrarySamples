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

#include <iomanip>
#include <iostream>

int main() {
    nvmmhHandle_t handle = nullptr;
    NVMMH_CHECK(nvMatmulHeuristicsCreate(&handle));


    // Create hardware descriptor for H100 SXM5
    nvmmhHardwareDescriptor_t hwDescr{};
    NVMMH_CHECK(nvMatmulHeuristicsHardwareDescriptorCreate(&hwDescr));
    NVMMH_CHECK(nvMatmulHeuristicsHardwareDescriptorSetPredefinedGpu(hwDescr, NVMMH_NVGPU_H100_SXM));

    // Create a matmul problem
    constexpr nvmmhMatmulProblem_t problem = {
            .M = 1024,                                         // Output matrix height
            .N = 2048,                                         // Output matrix width
            .K = 4096,                                         // Reduced dimension
            .batchSize = 1,                                    // Single batch
            .matmulLayout = NVMMH_MATMUL_LAYOUT_NN_ROW_MAJOR   // No transpose, row major
    };

    // Get kernel configuration using heuristics
    nvmmhKernelConfiguration_t kernelConfig;
    const unsigned numConfigs = nvMatmulHeuristicsGetGemmConfig(handle, "HSH",          // mixed precision
                                                                NVMMH_FLAG_NONE,        // No special flags
                                                                NVMMH_TARGET_CUTLASS,   // Targets CUTLASS2-style kernels
                                                                &problem, &kernelConfig,
                                                                1,        // Request one configuration
                                                                hwDescr   // Use H100 SXM5 hardware descriptor
    );

    if (numConfigs == 0) {
        std::cerr << "Failed to get kernel configuration" << std::endl;
        NVMMH_CHECK(nvMatmulHeuristicsHardwareDescriptorDestroy(&hwDescr));
        return 1;
    }

    // Estimate runtime for this configuration
    const double estimatedRuntime = nvMatmulHeuristicsEstimateRuntime(handle, "HSH",          // mixed precision
                                                                      NVMMH_TARGET_CUTLASS,   // Target CUTLASS2
                                                                      &problem, &kernelConfig,
                                                                      hwDescr   // Use H100 SXM5 hardware descriptor
    );

    // Print problem details
    std::cout << "Problem: M=" << problem.M << ", N=" << problem.N << ", K=" << problem.K
              << ", Layout=" << (problem.matmulLayout == NVMMH_MATMUL_LAYOUT_NN_ROW_MAJOR ? "NN" : "Other") << std::endl;

    // Print kernel configuration
    std::cout << "Kernel Configuration:" << std::endl;
    std::cout << "  CTA: " << kernelConfig.cta[0] << "x" << kernelConfig.cta[1] << "x" << kernelConfig.cta[2] << std::endl;
    std::cout << "  Warp: " << kernelConfig.warp[0] << "x" << kernelConfig.warp[1] << "x" << kernelConfig.warp[2] << std::endl;
    std::cout << "  Instr: " << kernelConfig.instr[0] << "x" << kernelConfig.instr[1] << "x" << kernelConfig.instr[2] << std::endl;
    std::cout << "  SplitK: " << kernelConfig.splitK << std::endl;
    std::cout << "  Load Stages: " << static_cast<int>(kernelConfig.loadStages) << std::endl;
    std::cout << "  Grid Swizzle: " << static_cast<int>(kernelConfig.gridSwizzle) << std::endl;
    std::cout << "  CTA Order: " << static_cast<int>(kernelConfig.ctaOrder) << std::endl;
    std::cout << "  Cluster Config: " << static_cast<int>(kernelConfig.cluster[0]) << "x" << static_cast<int>(kernelConfig.cluster[1]) << std::endl;

    // Print estimated runtime
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Estimated Runtime: " << estimatedRuntime << " seconds" << std::endl;

    // Clean up hardware descriptor
    NVMMH_CHECK(nvMatmulHeuristicsHardwareDescriptorDestroy(&hwDescr));
    NVMMH_CHECK(nvMatmulHeuristicsDestroy(&handle));
    return 0;
}