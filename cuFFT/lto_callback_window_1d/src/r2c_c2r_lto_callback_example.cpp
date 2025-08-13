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


/*
 * Example showing the use of LTO callbacks with CUFFT to perform
 * truncation with zero padding.
 *
*/

#include <cuda_runtime_api.h>
#include <cufftXt.h>
#include "r2c_c2r_reference.h"
#include "common.h"
#include "callback_params.h"

// NOTE: Header containing the compiled LTO callback device function in a C array, generated with bin2c
#include "r2c_c2r_lto_callback_device_fatbin.h"

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

	// Create a CUFFT plan for the forward transform, and a cuFFT plan for the inverse transform with load callback
	cufftHandle forward_plan, inverse_plan_cb;
	size_t work_size;

	CHECK_ERROR(cufftCreate(&forward_plan));
	CHECK_ERROR(cufftCreate(&inverse_plan_cb));

	// NOTE: LTO callbacks must be set before plan creation and cannot be unset (yet)
#ifdef CB_USE_CONSTANT_MEMORY
	cb_params *device_params = nullptr;
	std::string callback_name = "windowing_constant_memory_callback";
#else
	// Define a structure used to pass in the window size
	cb_params host_params;
	host_params.window_size = window_size;
	host_params.signal_size = complex_signal_size;

	// Allocate and copy callback parameters from host to GPU
	cb_params *device_params;
	CHECK_ERROR(cudaMalloc((void **)&device_params, sizeof(cb_params)));
	CHECK_ERROR(cudaMemcpy(device_params, &host_params, sizeof(cb_params), cudaMemcpyHostToDevice));

	std::string callback_name = "windowing_callback";
#endif
	size_t lto_callback_fatbin_size = sizeof(window_callback);
	CHECK_ERROR(cufftXtSetJITCallback(inverse_plan_cb,
                                      callback_name.c_str(),
                                      (void*)window_callback,
                                      lto_callback_fatbin_size,
                                      CUFFT_CB_LD_COMPLEX,
                                      (void **)&device_params));

	CHECK_ERROR(cufftMakePlan1d(forward_plan, signal_size, CUFFT_R2C, batches, &work_size));
	CHECK_ERROR(cufftMakePlan1d(inverse_plan_cb, signal_size, CUFFT_C2R, batches, &work_size));

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
