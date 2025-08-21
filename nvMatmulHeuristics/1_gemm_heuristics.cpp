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
