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

/*
 * Reference for the example of LTO callbacks,
 * run the same plans but perform the windowing with
 * a separate kernel.
 *
*/

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <stdio.h>
#include <vector>
#include "r2c_c2r_reference.h"

__global__ void windowing(unsigned nbatches, unsigned complex_signal_size, unsigned window_size, float2* buffer)
{
	const size_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= nbatches * complex_signal_size) return;

	const unsigned sample = idx % complex_signal_size;

	buffer[idx].x = (sample < window_size) ? buffer[idx].x : 0.f;
	buffer[idx].y = (sample < window_size) ? buffer[idx].y : 0.f;
}

int reference_r2c_window_c2r(unsigned batches, unsigned signal_size, unsigned window_size, float* input_signals, float* output_signals) {
	const unsigned complex_signal_size = signal_size / 2 + 1;
	const size_t complex_size_bytes    = batches * complex_signal_size * 2 * sizeof(float);

	// Allocate and copy input from host to GPU
	float *device_signals;
	CHECK_ERROR(cudaMalloc((void **)&device_signals, complex_size_bytes));
	CHECK_ERROR(cudaMemcpy(device_signals, input_signals, complex_size_bytes, cudaMemcpyHostToDevice));

	// Create a CUFFT plan for the forward transform, and a cuFFT plan for the inverse transform
	cufftHandle forward_plan, inverse_plan;
	size_t work_size;

	CHECK_ERROR(cufftCreate(&forward_plan));
	CHECK_ERROR(cufftCreate(&inverse_plan));

	CHECK_ERROR(cufftMakePlan1d(forward_plan, signal_size, CUFFT_R2C, batches, &work_size));
	CHECK_ERROR(cufftMakePlan1d(inverse_plan, signal_size, CUFFT_C2R, batches, &work_size));

	// Transform signal forward
	printf("Transforming reference cufftExecR2C\n");
	CHECK_ERROR(cufftExecR2C(forward_plan, (cufftReal *)device_signals, (cufftComplex *)device_signals));

	// Apply window via separate kernel
	windowing<<<(batches * complex_signal_size + 255) / 256, 256>>>(batches, complex_signal_size, window_size, (float2*) device_signals);
	cudaDeviceSynchronize();
	CHECK_ERROR(cudaGetLastError());

	// Inverse-transform the signal
	printf("Transforming reference cufftExecC2R\n");
	CHECK_ERROR(cufftExecC2R(inverse_plan, (cufftComplex *)device_signals, (cufftReal *)device_signals));

	// Copy device memory to host
	CHECK_ERROR(cudaMemcpy(output_signals, device_signals, complex_size_bytes, cudaMemcpyDeviceToHost));

	// Destroy CUFFT context
	CHECK_ERROR(cufftDestroy(forward_plan));
	CHECK_ERROR(cufftDestroy(inverse_plan));

	// Cleanup memory
	CHECK_ERROR(cudaFree(device_signals));

	return PASS_VALUE;
}