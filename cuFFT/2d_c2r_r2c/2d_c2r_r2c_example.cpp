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


#include <array>
#include <complex>
#include <iostream>
#include <vector>
#include <cufft.h>
#include "cufft_utils.h"

using dim_t = std::array<int, 2>;

int main(int argc, char *argv[]) {
    cufftHandle planc2r, planr2c;
    cudaStream_t stream = NULL;

    int nx = 2;
    int ny = 4;
    dim_t fft_size = {nx, ny};
    int batch_size = 2;

    using scalar_type = float;
    using input_type = std::complex<scalar_type>;
    using output_type = scalar_type;

    std::vector<input_type> input_complex(batch_size * nx * (ny/2 + 1));
    std::vector<output_type> output_real(batch_size * nx * ny, 0);

    for (int i = 0; i < input_complex.size(); i++) {
        input_complex[i] = input_type(i, 0);
    }

    std::printf("Input array:\n");
    for (auto &i : input_complex) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");

    cufftComplex *d_data = nullptr;

    // inembed/onembed being nullptr indicates contiguous data for each batch, then the stride and dist settings are ignored
    CUFFT_CALL(cufftPlanMany(&planc2r, fft_size.size(), fft_size.data(),
                             nullptr, 1, 0, // *inembed, istride, idist
                             nullptr, 1, 0, // *onembed, ostride, odist
                             CUFFT_C2R, batch_size));
    CUFFT_CALL(cufftPlanMany(&planr2c, fft_size.size(), fft_size.data(),
                             nullptr, 1, 0, // *inembed, istride, idist
                             nullptr, 1, 0, // *onembed, ostride, odist
                             CUFFT_R2C, batch_size));

    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(planc2r, stream));
    CUFFT_CALL(cufftSetStream(planr2c, stream));

    // Create device arrays
    // For in-place r2c/c2r transforms, make sure the device array is always allocated to the size of complex array
    CUDA_RT_CALL(
        cudaMalloc(reinterpret_cast<void **>(&d_data), sizeof(input_type) * input_complex.size()));

    CUDA_RT_CALL(cudaMemcpyAsync(d_data, (input_complex.data()), sizeof(input_type) * input_complex.size(),
                                 cudaMemcpyHostToDevice, stream));

    // C2R
    CUFFT_CALL(cufftExecC2R(planc2r, d_data, reinterpret_cast<scalar_type*>(d_data)));

    CUDA_RT_CALL(cudaMemcpyAsync(output_real.data(), reinterpret_cast<scalar_type*>(d_data), sizeof(output_type) * output_real.size(),
                                 cudaMemcpyDeviceToHost, stream));

    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    std::printf("Output array after C2R:\n");
    for (auto i = 0; i < output_real.size(); i++) {
        std::printf("%f\n", output_real[i]);
    }
    std::printf("=====\n");

    // Normalize the data
    scaling_kernel<<<1, 128, 0, stream>>>(d_data, input_complex.size(), 1.f/(nx * ny));

    // R2C
    CUFFT_CALL(cufftExecR2C(planr2c, reinterpret_cast<scalar_type*>(d_data), d_data));

    CUDA_RT_CALL(cudaMemcpyAsync(input_complex.data(), d_data, sizeof(input_type) * input_complex.size(),
                                 cudaMemcpyDeviceToHost, stream));

    std::printf("Output array after C2R, Normalization, and R2C:\n");
    for (auto i = 0; i < input_complex.size(); i++) {
        std::printf("%f + %fj\n", input_complex[i].real(), input_complex[i].imag());
    }
    std::printf("=====\n");


    /* free resources */
    CUDA_RT_CALL(cudaFree(d_data));

    CUFFT_CALL(cufftDestroy(planc2r));
    CUFFT_CALL(cufftDestroy(planr2c));

    CUDA_RT_CALL(cudaStreamDestroy(stream));

    CUDA_RT_CALL(cudaDeviceReset());

    return EXIT_SUCCESS;
}