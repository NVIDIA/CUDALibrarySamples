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
 * Example showing the use of legacy callbacks with CUFFT to perform
 * zero-padding on a single 1D signal.
 *
 */

#include <cuda_runtime_api.h>
#include <cufftXt.h>
#include "common.h"
#include "r2c_c2r_padding_reference.h"
#include "callback_params.h"

// Load callback for R2C: returns 0 for indices beyond signal_size
__device__ cufftReal padding_load_callback_r2c(void* input,
                                             size_t index,
                                             void* info,
                                             void* sharedMem) {
    PaddingCallbackParams* params = (PaddingCallbackParams*)info;
    cufftReal* in = (cufftReal*)input;
    
    // If index is beyond the original data, return zero
    if (index >= params->signal_size) {
        return 0.0f;
    }
    
    return in[index];
}

// Store callback for C2R: only stores first signal_size elements, zeros the rest
__device__ void padding_store_callback_c2r(void* output,
                                          size_t index,
                                          cufftReal in,
                                          void* info,
                                          void* sharedMem) {
    PaddingCallbackParams* params = (PaddingCallbackParams*)info;
    cufftReal* out = (cufftReal*)output;
    
    // Only store if within original size, zero and normalize otherwise
    if (index < params->signal_size) {
        out[index] = in / static_cast<float>(params->padded_signal_size);
    } else {
        out[index] = 0.0f;
    }
}

// Device pointers to callback functions (required for legacy callbacks)
__device__ cufftCallbackLoadR d_padding_load_callback_ptr = padding_load_callback_r2c;
__device__ cufftCallbackStoreR d_padding_store_callback_ptr = padding_store_callback_c2r;

static_assert(padded_signal_size > signal_size, "The padded size must be larger than the signal size");

int test_r2c_padding_c2r() {

    // Allocate and initialize host input (real data) - only signal_size elements
    float *input_signals = new float[signal_size];
    float *output_signals = new float[signal_size];
    float *reference = new float[signal_size];

    // Initialize with simple linear ramp for debugging
    for (unsigned i = 0; i < signal_size; i++) {
        input_signals[i] = static_cast<float>(i);
    }

    const size_t padded_real_size_bytes = padded_signal_size * sizeof(float);
    const size_t complex_size_bytes = padding_complex_signal_size * 2 * sizeof(float);

    // Allocate device memory for padded real data
    float *device_signals;
    CHECK_ERROR(cudaMalloc((void **)&device_signals, padded_real_size_bytes));
    CHECK_ERROR(cudaMemcpy(device_signals, input_signals, signal_size * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate device memory for complex data
    cufftComplex *device_complex;
    CHECK_ERROR(cudaMalloc((void **)&device_complex, complex_size_bytes));

    // Create callback parameters on host
    PaddingCallbackParams h_params;
    h_params.signal_size = signal_size;
    h_params.padded_signal_size = padded_signal_size;

    // Allocate and copy callback parameters to device
    PaddingCallbackParams *device_params;
    CHECK_ERROR(cudaMalloc((void **)&device_params, sizeof(PaddingCallbackParams)));
    CHECK_ERROR(cudaMemcpy(device_params, &h_params, sizeof(PaddingCallbackParams), cudaMemcpyHostToDevice));

    // Get device pointers to callback functions
    cufftCallbackLoadR h_load_callback_ptr;
    cufftCallbackStoreR h_store_callback_ptr;
    CHECK_ERROR(cudaMemcpyFromSymbol(&h_load_callback_ptr, d_padding_load_callback_ptr, sizeof(h_load_callback_ptr)));
    CHECK_ERROR(cudaMemcpyFromSymbol(&h_store_callback_ptr, d_padding_store_callback_ptr, sizeof(h_store_callback_ptr)));

    // Create cuFFT plans
    cufftHandle forward_plan, inverse_plan;
    size_t work_size;

    CHECK_ERROR(cufftCreate(&forward_plan));
    CHECK_ERROR(cufftCreate(&inverse_plan));

    // Create forward plan (R2C) with load callback
    CHECK_ERROR(cufftMakePlan1d(forward_plan, padded_signal_size, CUFFT_R2C, 1, &work_size));
    CHECK_ERROR(cufftXtSetCallback(forward_plan, (void **)&h_load_callback_ptr,
                                   CUFFT_CB_LD_REAL, (void **)&device_params));

    // Create inverse plan (C2R) with store callback
    CHECK_ERROR(cufftMakePlan1d(inverse_plan, padded_signal_size, CUFFT_C2R, 1, &work_size));
    CHECK_ERROR(cufftXtSetCallback(inverse_plan, (void **)&h_store_callback_ptr,
                                   CUFFT_CB_ST_REAL, (void **)&device_params));

    // Execute forward FFT (R2C) with load callback
    printf("Transforming signal cufftExecR2C with padding load callback\n");
    CHECK_ERROR(cufftExecR2C(forward_plan, (cufftReal *)device_signals, (cufftComplex *)device_complex));

    // Execute inverse FFT (C2R) with store callback
    printf("Transforming signal cufftExecC2R with padding store callback\n");
    CHECK_ERROR(cufftExecC2R(inverse_plan, (cufftComplex *)device_complex, (cufftReal *)device_signals));

    // Copy result back to host (only first signal_size real values)
    CHECK_ERROR(cudaMemcpy(output_signals, device_signals, signal_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute reference
    if (reference_r2c_padding_c2r(1, signal_size, padded_signal_size, input_signals, reference) != PASS_VALUE) {
        printf("Failed to compute the reference");
        delete[] input_signals;
        delete[] output_signals;
        delete[] reference;
        cufftDestroy(forward_plan);
        cufftDestroy(inverse_plan);
        cudaFree(device_signals);
        cudaFree(device_complex);
        cudaFree(device_params);
        return ERROR_VALUE;
    }

    double l2_error = compute_error_padding<float>(reference, output_signals, 1, signal_size);
    printf("L2 error: %e\n", l2_error);

    // Cleanup host memory
    delete[] input_signals;
    delete[] output_signals;
    delete[] reference;

    // Destroy CUFFT context
    CHECK_ERROR(cufftDestroy(forward_plan));
    CHECK_ERROR(cufftDestroy(inverse_plan));

    // Cleanup memory
    CHECK_ERROR(cudaFree(device_signals));
    CHECK_ERROR(cudaFree(device_complex));
    CHECK_ERROR(cudaFree(device_params));

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
        printf("cuFFT with callbacks requires CUDA architecture SM5.0 or higher\n");
        return ERROR_VALUE;
    }

    return test_r2c_padding_c2r();
}
