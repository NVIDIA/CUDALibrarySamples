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