/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <complex>
#include <iostream>
#include <vector>
#include <cufft.h>
#include "cufft_utils.h"

int main(int argc, char *argv[]) {
    cufftHandle plan;
    cudaStream_t stream = NULL;

    int fft_size = 8;
    int batch_size = 2;
    int element_count = batch_size * fft_size;

    using scalar_type = float;
    using data_type = std::complex<scalar_type>;

    std::vector<data_type> data(element_count, 0);

    for (int i = 0; i < element_count; i++) {
        data[i] = data_type(i, -i);
    }

    std::printf("Input array:\n");
    for (auto &i : data) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");

    cufftComplex *d_data = nullptr;

    CUFFT_CALL(cufftPlan1d(&plan, fft_size, CUFFT_C2C, batch_size));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan, stream));

    // Create device data arrays
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), sizeof(data_type) * data.size()));
    CUDA_RT_CALL(cudaMemcpyAsync(d_data, data.data(), sizeof(data_type) * data.size(),
                                 cudaMemcpyHostToDevice, stream));

    /*
     * Note:
     *  Identical pointers to data and output arrays implies in-place transformation
     */
    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    // Normalize the data
    scaling_kernel<<<1, 128, 0, stream>>>(d_data, element_count, 1.f/fft_size);

    // The original data should be recovered after Forward FFT, normalization and inverse FFT
    CUFFT_CALL(cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE));

    CUDA_RT_CALL(cudaMemcpyAsync(data.data(), d_data, sizeof(data_type) * data.size(),
                                 cudaMemcpyDeviceToHost, stream));

    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    std::printf("Output array after Forward FFT, Normalization, and Inverse FFT :\n");
    for (auto &i : data) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");

    /* free resources */
    CUDA_RT_CALL(cudaFree(d_data))

    CUFFT_CALL(cufftDestroy(plan));

    CUDA_RT_CALL(cudaStreamDestroy(stream));

    CUDA_RT_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}