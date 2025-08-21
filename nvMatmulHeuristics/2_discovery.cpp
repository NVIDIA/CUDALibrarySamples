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
