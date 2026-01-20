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

#ifndef CUSOLVERDX_EXAMPLE_COMMON_MEASURE_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_MEASURE_HPP

#include "numeric.hpp"
#include "cusolverdx.hpp"

namespace common {

    struct measure {
        // Returns execution time in ms.
        template<typename Kernel>
        static float execution(Kernel&& kernel, const unsigned int warm_up_runs, const unsigned int runs, cudaStream_t stream) {
            cudaEvent_t startEvent, stopEvent;
            CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvent));
            CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvent));
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            for (unsigned int i = 0; i < warm_up_runs; i++) {
                kernel(stream);
            }

            CUDA_CHECK_AND_EXIT(cudaGetLastError());
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            CUDA_CHECK_AND_EXIT(cudaEventRecord(startEvent, stream));
            for (unsigned int i = 0; i < runs; i++) {
                kernel(stream);
            }
            CUDA_CHECK_AND_EXIT(cudaEventRecord(stopEvent, stream));
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            float time;
            CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&time, startEvent, stopEvent));
            CUDA_CHECK_AND_EXIT(cudaEventDestroy(startEvent));
            CUDA_CHECK_AND_EXIT(cudaEventDestroy(stopEvent));
            return time;
        }

        // Returns execution time in ms.
        // Takes a reset function to allow testing in-place functions
        template<typename Kernel, typename Reset>
        static float execution(Kernel&& kernel, Reset&& reset, const unsigned int warm_up_runs, const unsigned int runs, cudaStream_t stream) {
            std::vector<cudaEvent_t> startEvents (runs), stopEvents(runs);
            for (int i = 0; i < runs; ++i) {
                CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvents[i]));
                CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvents[i]));
            }
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            for (unsigned int i = 0; i < warm_up_runs; i++) {
                kernel(stream);
                reset(stream);
            }

            CUDA_CHECK_AND_EXIT(cudaGetLastError());
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            for (unsigned int i = 0; i < runs; i++) {
                CUDA_CHECK_AND_EXIT(cudaEventRecord(startEvents[i], stream));
                kernel(stream);
                CUDA_CHECK_AND_EXIT(cudaEventRecord(stopEvents[i], stream));
                if (i+1 != runs) {
                    reset(stream);
                }
            }
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            float total_time = 0;
            for (int i = 0; i < runs; ++i) {
                float time;
                CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&time, startEvents[i], stopEvents[i]));
                total_time += time;

                CUDA_CHECK_AND_EXIT(cudaEventDestroy(startEvents[i]));
                CUDA_CHECK_AND_EXIT(cudaEventDestroy(stopEvents[i]));
            }
            return total_time;
        }
    };

    //==================================
    // GFLOPS calculation
    //==================================

    template<typename DataType>
    constexpr double get_flops_gemm(const unsigned int m, const unsigned int n, const unsigned int k) {
        auto fmuls = m * n * k;
        auto fadds = m * n * k;

        if constexpr (common::is_complex<DataType>()) {
            return 6. * fmuls + 2. * fadds;
        } else {
            return fmuls + fadds;
        }
    }

    template<typename DataType>
    constexpr double get_flops_trsm(const unsigned int n, const unsigned int nrhs) {
        auto fmuls = nrhs * n * (n + 1) / 2.;
        auto fadds = nrhs * n * (n - 1) / 2.;

        if constexpr (common::is_complex<DataType>()) {
            return 6. * fmuls + 2. * fadds;
        } else {
            return fmuls + fadds;
        }
    }

    // https://github.com/icl-utk-edu/magma/blob/master/testing/flops.h
    template<typename DataType>
    constexpr double get_flops_potrf(const unsigned int n) {
        auto fmuls = n * (((1. / 6.) * n + 0.5) * n + (1. / 3.));
        auto fadds = n * (((1. / 6.) * n) * n - (1. / 6.));

        if constexpr (common::is_complex<DataType>()) {
            return 6. * fmuls + 2. * fadds;
        } else {
            return fmuls + fadds;
        }
    }

    template<typename DataType>
    constexpr double get_flops_potrs(const unsigned int n, const unsigned int nrhs) {
        return 2 * get_flops_trsm<DataType>(n, nrhs);
    }

    template<typename DataType>
    constexpr double get_flops_getrf(const unsigned int m, const unsigned int n) {
        // FLOP calculation depends on whether m or n is smaller
        auto mn = min(m, n);
        auto mx = max(m, n);

        auto fmuls = 0.5 * mn * (mn * (mx - mn / 3. - 1) + mx) + 2. * mn / 3.;
        auto fadds = 0.5 * mn * (mn * (mx - mn / 3.)     - mx) +      mn / 6.;

        if constexpr (common::is_complex<DataType>()) {
            return 6. * fmuls + 2. * fadds;
        } else {
            return fmuls + fadds;
        }
    }

    template<typename DataType>
    constexpr double get_flops_geqrf(const unsigned int m, const unsigned int n) {

        double fmuls_geqrf = (m > n) ? (n * (n * (0.5 - (1./3.) * n + m) + m + 23. / 6.)) : (m * (m * (-0.5 - (1./3.) * m + n) + 2.*(n) + 23. / 6.));
        double fadds_geqrf = (m > n) ? (n * (n * (0.5 - (1./3.) * n + m) + 5. / 6.)) : (m * (m * (-0.5 - (1./3.) * m + n) + n + 5. / 6.));

        if constexpr (common::is_complex<DataType>()) {
            return 6. * fmuls_geqrf + 2. * fadds_geqrf;
        } else {
            return fmuls_geqrf + fadds_geqrf;
        }
    }

    template<typename DataType>
    constexpr double get_flops_unmqr(const cusolverdx::side side, const unsigned int m, const unsigned int n, const unsigned int k) {

        double fmuls_unmqr = (side == cusolverdx::side::left) ? k * (2 * m - k) * n : k * (2 * n - k) * m;
        double fadds_unmqr = (side == cusolverdx::side::left) ? k * (2 * m - k) * n : k * (2 * n - k) * m;

        if constexpr (common::is_complex<DataType>()) {
            return 6. * fmuls_unmqr + 2. * fadds_unmqr;
        } else {
            return fmuls_unmqr + fadds_unmqr;
        }
    }

    template<typename DataType>
    constexpr double get_flops_ungqr(const unsigned int m, const unsigned int n, const unsigned int k) {
        auto fmuls = k * (2 * m * n - (m + n) * k + 2./3. * k * k  + 2 * n - k - 5. / 3.);
        auto fadds = k * (2 * m * n - (m + n) * k + 2./3. * k * k  + n - m + 1. / 3.);

        if constexpr (common::is_complex<DataType>()) {
            return 6. * fmuls + 2. * fadds;
        } else {
            return fmuls + fadds;
        }
    }

    inline void print_perf(const std::string msg, const unsigned int batches, const unsigned int M, const unsigned int N, const unsigned int nrhs, const double gflops, const double gb_s, const double ms, const unsigned int blockDim) {
        printf("%-30s %10u %5u %5u %5u  %8.2f GFLOP/s, %7.2f GB/s, %7.4f ms, %d blockDim\n", msg.c_str(), batches, M, N, nrhs, gflops, gb_s, ms, blockDim);
    }
    inline void print_perf(const std::string msg, const unsigned int batches, const unsigned int M, const unsigned int N, const unsigned int nrhs, const double gflops, const double gb_s, const double ms, const unsigned int blockDim, const unsigned int bpb) {
        printf("%-30s %10u %5u %5u %5u  %8.2f GFLOP/s, %7.2f GB/s, %7.4f ms, %d blockDim, %d bpb\n", msg.c_str(), batches, M, N, nrhs, gflops, gb_s, ms, blockDim, bpb);
    }
} // namespace common

#endif
