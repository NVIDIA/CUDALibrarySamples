// /* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//  *
//  * Redistribution and use in source and binary forms, with or without
//  * modification, are permitted provided that the following conditions
//  * are met:
//  *  * Redistributions of source code must retain the above copyright
//  *    notice, this list of conditions and the following disclaimer.
//  *  * Redistributions in binary form must reproduce the above copyright
//  *    notice, this list of conditions and the following disclaimer in the
//  *    documentation and/or other materials provided with the distribution.
//  *  * Neither the name of NVIDIA CORPORATION nor the names of its
//  *    contributors may be used to endorse or promote products derived
//  *    from this software without specific prior written permission.
//  *
//  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
//  * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
//  * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
//  * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//  */


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
