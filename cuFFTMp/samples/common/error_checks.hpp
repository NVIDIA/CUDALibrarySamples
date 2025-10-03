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

#include <cufftMp.h>
#include <complex>
#include "../iterators/box_iterator.hpp"

#define CUDA_CHECK(ans) { gpu_checkAssert((ans), __FILE__, __LINE__); }
inline void gpu_checkAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"CUDA_CHECK: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CUFFT_CHECK(ans) { cufft_check((ans), __FILE__, __LINE__); }
inline void cufft_check(int code, const char *file, int line, bool abort=true)
{
    if (code != CUFFT_SUCCESS) 
    {
        fprintf(stderr,"CUFFT_CHECK: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}

template<typename T>
double compute_error(const T& ref, const T& test, Box3D box, MPI_Comm comm) {
    auto[begin, end] = BoxIterators(box, ref.data());
    double diff_sq = 0;
    double norm_sq = 0;
    for(auto it = begin; it != end; it++) {
        diff_sq += std::norm(ref[it.i()] - test[it.i()]);
        norm_sq += std::norm(ref[it.i()]);
    }

    double diff_global = 0;
    double norm_global = 0;

    MPI_Allreduce(&diff_sq, &diff_global, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&norm_sq, &norm_global, 1, MPI_DOUBLE, MPI_SUM, comm);

    return std::sqrt(diff_global / norm_global);
}

int assess_error(double error, int rank, double tolerance = 1e-6) {
    if(error > tolerance) {
        if (rank == 0) {
            printf("FAILED with L2 error %e > %e\n", error, tolerance);
        }
        return 1;
    } else {
        if (rank == 0) {
            printf("PASSED with L2 error %e < %e\n", error, tolerance);
        }
        return 0;
    }
}