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
 * Example showing the use of LTO callbacks with CUFFT to perform
 * zero-padding on a single 1D signal.
 *
 */

#include <cuda_runtime_api.h>
#include <cufftXt.h>
#include "r2c_c2r_padding_reference.h"
#include "common.h"
#include "callback_params.h"

// NOTE: Headers containing the compiled LTO callback device functions in C arrays, generated with bin2c
#include "r2c_c2r_padding_load_callback_fatbin.h"
#include "r2c_c2r_padding_store_callback_fatbin.h"

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

    // Create cuFFT plans
    cufftHandle forward_plan, inverse_plan;
    size_t work_size;

    CHECK_ERROR(cufftCreate(&forward_plan));
    CHECK_ERROR(cufftCreate(&inverse_plan));

    // NOTE: LTO callbacks must be set before plan creation and cannot be unset (yet)
    // Set load callback for forward plan (R2C)
    size_t lto_load_callback_fatbin_size = sizeof(padding_load_callback);
    CHECK_ERROR(cufftXtSetJITCallback(forward_plan,
                                       "padding_load_callback",
                                       (void*)padding_load_callback,
                                       lto_load_callback_fatbin_size,
                                       CUFFT_CB_LD_REAL,
                                       (void **)&device_params));

    // Set store callback for inverse plan (C2R)
    size_t lto_store_callback_fatbin_size = sizeof(padding_store_callback);
    CHECK_ERROR(cufftXtSetJITCallback(inverse_plan,
                                       "padding_store_callback",
                                       (void*)padding_store_callback,
                                       lto_store_callback_fatbin_size,
                                       CUFFT_CB_ST_REAL,
                                       (void **)&device_params));

    // Create plans AFTER setting callbacks
    CHECK_ERROR(cufftMakePlan1d(forward_plan, padded_signal_size, CUFFT_R2C, 1, &work_size));
    CHECK_ERROR(cufftMakePlan1d(inverse_plan, padded_signal_size, CUFFT_C2R, 1, &work_size));

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
        printf("cuFFT with LTO requires CUDA architecture SM5.0 or higher\n");
        return ERROR_VALUE;
    }

    return test_r2c_padding_c2r();
}
