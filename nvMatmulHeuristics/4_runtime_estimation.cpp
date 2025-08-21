/*
* Copyright 2025 NVIDIA Corporation. All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users.  These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
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
