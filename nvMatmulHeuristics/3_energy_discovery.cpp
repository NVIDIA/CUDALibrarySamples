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


constexpr nvmmhNvidiaGpu_t GPU = NVMMH_NVGPU_A100_SXM_80GB;
constexpr nvmmhTarget_t TARGET = NVMMH_TARGET_CUTLASS;
constexpr const char* PRECISION = "HSS";
constexpr nvmmhMatmulLayout_t LAYOUT = NVMMH_MATMUL_LAYOUT_TN_ROW_MAJOR;

constexpr nvmmhMatmulProblem_t someProblem = {.M = 16384, .N = 16384, .K = 16384, .batchSize = 1, .matmulLayout = LAYOUT};


int main() {
    nvmmhHandle_t handle = nullptr;
    NVMMH_CHECK(nvMatmulHeuristicsCreate(&handle));


    nvmmhBackend_t backend = nullptr;
    NVMMH_CHECK(nvMatmulHeuristicsBackendCreate(&backend, TARGET));

    nvmmhHardwareDescriptor_t hwDescr{};
    NVMMH_CHECK(nvMatmulHeuristicsHardwareDescriptorCreate(&hwDescr));
    NVMMH_CHECK(nvMatmulHeuristicsHardwareDescriptorSetPredefinedGpu(hwDescr, GPU));

    // Energy awareness tuning requires the regular heuristic tuning.
    // For the demo we will load an embedded profile, but the user can also perform the online tuning before.
    NVMMH_CHECK(nvMatmulHeuristicsLoadInternalDiscoverySet(handle, PRECISION, TARGET, LAYOUT, hwDescr));
    nvmmhKernelConfiguration_t kernel;
    if (nvMatmulHeuristicsGetGemmConfigEx(handle, PRECISION, NVMMH_FLAG_NONE, backend, &someProblem, &kernel, 1, hwDescr) != 1) abort();

    {
        const double energy_before = nvMatmulHeuristicsEstimateSiliconMetricEx(handle, PRECISION, backend, &someProblem, &kernel, NVMMH_METRIC_ENERGY_JOULES, hwDescr);
        std::cout << "Estimated Energy Before: " << energy_before << std::endl;
    }

    {
        nvmmhMatmulProblem_t problems[1024] = {};
        nvmmhKernelConfiguration_t configs[1024] = {};
        float energies[1024] = {};
        const unsigned count = nvMatmulHeuristicsGetEnergyDiscoverySet(handle, PRECISION, TARGET, LAYOUT, problems, configs, 1024, hwDescr);
        for (int i = 0; i < count; ++i) {
            std::cout << "[" << i + 1 << "] Problem: " << to_string(problems[i]) << ", Config: " << to_string(configs[i]) << std::endl;
            /* Run your benchmark */
            double energy = 0;
            // Dummy formula assuming an arbitrary power of 100 Watts for compute and memory subsystems.
            energy += 100 * nvMatmulHeuristicsEstimateSiliconMetricEx(handle, PRECISION, backend, problems + i, configs + i, NVMMH_METRIC_COMPUTE_S, hwDescr);
            energy += 100 * nvMatmulHeuristicsEstimateSiliconMetricEx(handle, PRECISION, backend, problems + i, configs + i, NVMMH_METRIC_LOAD_S, hwDescr);
            energy += 100 * nvMatmulHeuristicsEstimateSiliconMetricEx(handle, PRECISION, backend, problems + i, configs + i, NVMMH_METRIC_STORE_S, hwDescr);
            energies[i] = energy;
        }
        NVMMH_CHECK(nvMatmulHeuristicsCommitEnergyDiscoveryResults(handle, PRECISION, TARGET, LAYOUT, problems, configs, energies, count, hwDescr));
    }

    /* Same semantics as in sample #2 and for profile matching. */

    {
        const double energy_after = nvMatmulHeuristicsEstimateSiliconMetricEx(handle, PRECISION, backend, &someProblem, &kernel, NVMMH_METRIC_ENERGY_JOULES, hwDescr);
        std::cout << "Estimated Energy After: " << energy_after << std::endl;
    }
    NVMMH_CHECK(nvMatmulHeuristicsDestroy(&handle));
}
