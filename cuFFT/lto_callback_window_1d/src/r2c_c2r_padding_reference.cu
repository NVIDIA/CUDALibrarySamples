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
 * Reference for the padding example,
 * performs zero-padding manually without callbacks.
 *
 */

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <stdio.h>
#include "r2c_c2r_padding_reference.h"

int reference_r2c_padding_c2r(unsigned batches, unsigned signal_size, unsigned padded_signal_size, float* input_signals, float* output_signals) {
    const size_t padded_real_size_bytes = batches * padded_signal_size * sizeof(float);
    const size_t complex_size_bytes = batches * (padded_signal_size / 2 + 1) * 2 * sizeof(float);

    // Allocate device memory for padded real input
    float *device_padded_input;
    CHECK_ERROR(cudaMalloc((void **)&device_padded_input, padded_real_size_bytes));

    // Allocate device memory for complex data
    float *device_complex;
    CHECK_ERROR(cudaMalloc((void **)&device_complex, complex_size_bytes));

    // Allocate device memory for padded real output
    float *device_padded_output;
    CHECK_ERROR(cudaMalloc((void **)&device_padded_output, padded_real_size_bytes));

    // Zero-fill the padded input buffer
    CHECK_ERROR(cudaMemset(device_padded_input, 0, padded_real_size_bytes));

    // Copy input to device (only first signal_size elements per batch)
    for (unsigned b = 0; b < batches; b++) {
        CHECK_ERROR(cudaMemcpy(device_padded_input + b * padded_signal_size, 
                              input_signals + b * signal_size, 
                              signal_size * sizeof(float), 
                              cudaMemcpyHostToDevice));
    }

    // Create cuFFT plans
    cufftHandle forward_plan, inverse_plan;
    size_t work_size;

    CHECK_ERROR(cufftCreate(&forward_plan));
    CHECK_ERROR(cufftCreate(&inverse_plan));

    CHECK_ERROR(cufftMakePlan1d(forward_plan, padded_signal_size, CUFFT_R2C, batches, &work_size));
    CHECK_ERROR(cufftMakePlan1d(inverse_plan, padded_signal_size, CUFFT_C2R, batches, &work_size));

    // Transform signal forward (R2C) on padded data
    printf("Transforming reference cufftExecR2C\n");
    CHECK_ERROR(cufftExecR2C(forward_plan, (cufftReal *)device_padded_input, (cufftComplex *)device_complex));

    // Transform signal inverse (C2R)
    printf("Transforming reference cufftExecC2R\n");
    CHECK_ERROR(cufftExecC2R(inverse_plan, (cufftComplex *)device_complex, (cufftReal *)device_padded_output));

    // Copy padded output to host for normalization
    float *host_padded_output = new float[batches * padded_signal_size];
    CHECK_ERROR(cudaMemcpy(host_padded_output, device_padded_output, batches * padded_signal_size * sizeof(float), cudaMemcpyDeviceToHost));

    // Normalize and copy only first signal_size elements to output
    for (unsigned b = 0; b < batches; b++) {
        for (unsigned i = 0; i < signal_size; i++) {
            output_signals[b * signal_size + i] = host_padded_output[b * padded_signal_size + i] / static_cast<float>(padded_signal_size);
        }
    }
    
    delete[] host_padded_output;

    // Destroy CUFFT context
    CHECK_ERROR(cufftDestroy(forward_plan));
    CHECK_ERROR(cufftDestroy(inverse_plan));

    // Cleanup memory
    CHECK_ERROR(cudaFree(device_padded_input));
    CHECK_ERROR(cudaFree(device_complex));
    CHECK_ERROR(cudaFree(device_padded_output));

    return PASS_VALUE;
}
