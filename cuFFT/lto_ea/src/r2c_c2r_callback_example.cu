/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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


/* 
 * Example showing the use of LTO callbacks with CUFFT to perform 
 * normalization and truncation with zero padding.
 * 
*/

#include <cuda_runtime_api.h>
#include <cufftXt.h>
#include "common.h"
#include "r2c_c2r_reference.h"

// Struct to pass data to callback
struct cb_params {
  unsigned window_N;
  unsigned signal_size;
};

// This is the store callback routine. It filters high frequencies
// based on a truncation window specified by the user
// NOTE: unlike the LTO version, the callback function can have
// any name
__device__ cufftComplex windowing_callback(void *input,
                                           size_t index,
                                           void *info,
                                           void *sharedmem) {
 	const cb_params* params = static_cast<const cb_params*>(info);
	cufftComplex* cb_output = static_cast<cufftComplex*>(input);
	const unsigned sample   = index % params->signal_size;

	return (sample < params->window_N) ? cb_output[index] : cufftComplex{0.f, 0.f};
}

__device__ cufftCallbackLoadC device_callback_ptr = windowing_callback;

// Problem input parameters
constexpr unsigned batches              = 830;
constexpr unsigned signal_size          = 328;
constexpr unsigned window_size          =  32;
constexpr unsigned complex_signal_size  = signal_size / 2 + 1;

// Precision threshold
constexpr float threshold = 1e-6;

static_assert(window_size < (signal_size/2 + 1), "The window size must be smaller than the signal size in complex space");

int test_r2c_window_c2r() {

	// Padded array for in-place transforms
	float  input_signals[batches][2 * complex_signal_size] = {};
	float output_signals[batches][2 * complex_signal_size];
	float      reference[batches][2 * complex_signal_size];

	init_input_signals(batches, signal_size, &input_signals[0][0]);

	const size_t complex_size_bytes = batches * complex_signal_size * 2 * sizeof(float);

	// Allocate and copy input from host to GPU
	float *device_signals;
	CHECK_ERROR(cudaMalloc((void **)&device_signals, complex_size_bytes));
	CHECK_ERROR(cudaMemcpy(device_signals, input_signals, complex_size_bytes, cudaMemcpyHostToDevice));

	// Define a structure used to pass in the window size
	cb_params host_params;
	host_params.window_N    = window_size;
	host_params.signal_size = complex_signal_size;

	// Allocate and copy callback parameters from host to GPU
	cb_params *device_params;
	CHECK_ERROR(cudaMalloc((void **)&device_params, sizeof(cb_params)));
	CHECK_ERROR(cudaMemcpy(device_params, &host_params, sizeof(cb_params), cudaMemcpyHostToDevice));

	// Create a CUFFT plan for the forward transform, and a cuFFT plan for the inverse transform with load callback
	cufftHandle forward_plan, inverse_plan_cb;
	size_t work_size;

	CHECK_ERROR(cufftCreate(&forward_plan));
	CHECK_ERROR(cufftCreate(&inverse_plan_cb));

	CHECK_ERROR(cufftMakePlan1d(forward_plan, signal_size, CUFFT_R2C, batches, &work_size));
	CHECK_ERROR(cufftMakePlan1d(inverse_plan_cb, signal_size, CUFFT_C2R, batches, &work_size));

	// NOTE: The host needs to get a copy of the device pointer to the callback. Not required for LTO callback
	cufftCallbackLoadC host_callback_ptr;
	CHECK_ERROR(cudaMemcpyFromSymbol(&host_callback_ptr, device_callback_ptr, sizeof(host_callback_ptr)));

	// Now associate the load callback with the plan.
	CHECK_ERROR(cufftXtSetCallback(inverse_plan_cb, (void **)&host_callback_ptr, CUFFT_CB_LD_COMPLEX, (void **)&device_params));

	// Transform signal forward
	printf("Transforming signal cufftExecR2C\n");
	CHECK_ERROR(cufftExecR2C(forward_plan,    (cufftReal *)device_signals, (cufftComplex *)device_signals));

	// Apply window via load callback and inverse-transform the signal
	printf("Transforming signal cufftExecC2R\n");
	CHECK_ERROR(cufftExecC2R(inverse_plan_cb, (cufftComplex *)device_signals, (cufftReal *)device_signals));

	// Copy device memory to host
	CHECK_ERROR(cudaMemcpy(output_signals, device_signals, complex_size_bytes, cudaMemcpyDeviceToHost));

	// Destroy CUFFT context
	CHECK_ERROR(cufftDestroy(forward_plan));
	CHECK_ERROR(cufftDestroy(inverse_plan_cb));

	// Cleanup memory
	CHECK_ERROR(cudaFree(device_signals));
	CHECK_ERROR(cudaFree(device_params));

	// Compute reference
	if(reference_r2c_window_c2r(batches, signal_size, window_size, &input_signals[0][0], &reference[0][0]) != PASS_VALUE) {
		printf("Failed to compute the reference");
		return ERROR_VALUE;
	};

	double l2_error = compute_error<float>(&reference[0][0], &output_signals[0][0], batches, signal_size);
	printf("L2 error: %e\n", l2_error);

	return (l2_error < threshold) ? PASS_VALUE : ERROR_VALUE;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  struct cudaDeviceProp properties;
  int device;
  CHECK_ERROR(cudaGetDevice(&device));
  CHECK_ERROR(cudaGetDeviceProperties(&properties, device));
  if (!(properties.major >= 5)) {
    printf("cuFFT with LTO requires CUDA architecture SM5.0 or higher\n");
    return ERROR_VALUE;
  }

  return test_r2c_window_c2r();
}

