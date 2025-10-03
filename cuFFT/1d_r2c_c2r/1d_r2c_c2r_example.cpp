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
    cufftHandle planr2c, planc2r;
    cudaStream_t stream = NULL;

    int fft_size = 8;
    int batch_size = 2;
    int element_count = batch_size * fft_size;

    using scalar_type = float;
    using input_type = scalar_type;
    using output_type = std::complex<scalar_type>;

    std::vector<input_type> input(element_count, 0);
    std::vector<output_type> output((fft_size / 2 + 1) * batch_size);

    for (auto i = 0; i < element_count; i++) {
        input[i] = static_cast<input_type>(i);
    }

    std::printf("Input array:\n");
    for (auto &i : input) {
        std::printf("%f\n", i);
    }
    std::printf("=====\n");

    input_type *d_input = nullptr;
    cufftComplex *d_output = nullptr;

    CUFFT_CALL(cufftPlan1d(&planr2c, fft_size, CUFFT_R2C, batch_size));
    CUFFT_CALL(cufftPlan1d(&planc2r, fft_size, CUFFT_C2R, batch_size));

    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(planr2c, stream));
    CUFFT_CALL(cufftSetStream(planc2r, stream));

    // Create device arrays
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(input_type) * input.size()));
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(output_type) * output.size()));
    CUDA_RT_CALL(cudaMemcpyAsync(d_input, input.data(), sizeof(input_type) * input.size(),
                                 cudaMemcpyHostToDevice, stream));

    // out-of-place Forward transform
    CUFFT_CALL(cufftExecR2C(planr2c, d_input, d_output));

    CUDA_RT_CALL(cudaMemcpyAsync(output.data(), d_output, sizeof(output_type) * output.size(),
                                 cudaMemcpyDeviceToHost, stream));

    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    std::printf("Output array after Forward FFT:\n");
    for (auto &i : output) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");

    // Normalize the data
    scaling_kernel<<<1, 128, 0, stream>>>(d_output, element_count, 1./fft_size);

    // out-of-place Inverse transform
    CUFFT_CALL(cufftExecC2R(planc2r, d_output, d_input));

    CUDA_RT_CALL(cudaMemcpyAsync(output.data(), d_input, sizeof(input_type) * input.size(),
                                 cudaMemcpyDeviceToHost, stream));

    std::printf("Output array after Forward FFT, Normalization, and Inverse FFT:\n");
    for (auto i = 0; i < input.size()/2; i++) {
        std::printf("%f + %fj\n", output[i].real(), output[i].imag());
    }
    std::printf("=====\n");

    /* free resources */
    CUDA_RT_CALL(cudaFree(d_input));
    CUDA_RT_CALL(cudaFree(d_output));

    CUFFT_CALL(cufftDestroy(planr2c));
    CUFFT_CALL(cufftDestroy(planc2r));

    CUDA_RT_CALL(cudaStreamDestroy(stream));

    CUDA_RT_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}