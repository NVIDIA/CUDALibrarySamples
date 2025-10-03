/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#ifndef _COMMON__H_
#define _COMMON__H_

#include <cuda.h>
#include <cufft.h>
#include <complex>

// Some helper definitions
#define ERROR_VALUE -1
#define PASS_VALUE   0
#define PI 3.1415926535897932

// Check CUDA API error
inline int checkErrors(cudaError_t error, int line_number) {
	if (error != cudaSuccess) {
		printf("Example failed in CUDA API on line %d with error %d\n", line_number, error);
		return ERROR_VALUE;
	}
	return PASS_VALUE;
}

// Check cuFFT API error
inline int checkErrors(cufftResult error, int line_number) {
	if (error != CUFFT_SUCCESS) {
		printf("Example failed in cuFFT API on line %d with error %d\n", line_number, error);
		return ERROR_VALUE;
	}
	return PASS_VALUE;
}

#define CHECK_ERROR(error) checkErrors(error, __LINE__)

template<typename T>
double compute_error(T* ref, T* out, unsigned batches, unsigned signal_size){
    double squared_diff = 0;
    double squared_norm = 0;
	const unsigned batch_offset = 2 * (signal_size / 2 + 1);
    for (int b = 0; b < batches; b++) {
        for (int i = 0; i < signal_size; i++) {
            unsigned  ref_idx = b * batch_offset + i;
            squared_diff += std::norm(ref[ref_idx] - out[ref_idx]); // Note that std::norm(z) = z * conj(z), not the usual sqrt(z * conj(z))
            squared_norm += std::norm(ref[ref_idx]);
        }
    }
    return std::sqrt(squared_diff / squared_norm);
}

// Initialize input signals
void init_input_signals(unsigned batches, unsigned signal_size, float* signals);
#endif // _COMMON__H_