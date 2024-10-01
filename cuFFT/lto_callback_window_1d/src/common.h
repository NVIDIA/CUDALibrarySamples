/* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

void init_input_signals(unsigned batches, unsigned signal_size, float* signals);

#endif // _COMMON__H_
