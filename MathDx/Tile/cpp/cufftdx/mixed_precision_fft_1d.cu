// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdio>
#include <vector>

#include "cufftdx_tile.hpp"
#include "device_buffer.hpp"
#include "error_checks.hpp"
#include "error_metrics.hpp"

namespace ct = cuda::tiles;

constexpr size_t N = 16;

__tile_global__ void mixed_precision_fft_kernel(const float* input, float* output) {
    using complex_shape = ct::shape<N, 2>;
    auto input_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(input), complex_shape {} }, complex_shape {} };
    auto output_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(output), complex_shape {} }, complex_shape {} };

    // Global-memory I/O is single precision, but the Tile FFT itself runs in
    // double precision. Convert the resulting spectrum back to float before
    // storing, matching the mixed-I/O pattern in the cuFFTDx sample.
    const auto double_precision_input = ct::element_cast<double>(input_view.load(0, 0));
    const auto double_precision_output = cufftdx::tiles::fft(double_precision_input, cufftdx::tiles::axis<0>);
    output_view.store(ct::element_cast<float>(double_precision_output), 0, 0);
}

void cufft_reference(const std::vector<float>& input,
                     const std::vector<float>& tile_output,
                     double& relative_l2_error) {
    std::vector<cufftDoubleComplex> cufft_input(N);
    std::vector<cufftDoubleComplex> cufft_output(N);
    for (size_t index = 0; index < N; ++index) {
        cufft_input[index] = { static_cast<double>(input[2 * index]), static_cast<double>(input[2 * index + 1]) };
    }

    cufftHandle plan = 0;
    cufftDoubleComplex* device_input = nullptr;
    cufftDoubleComplex* device_output = nullptr;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_Z2Z, 1));
    CUDA_CHECK(cudaMalloc(&device_input, N * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&device_output, N * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMemcpy(device_input, cufft_input.data(), N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));
    CUFFT_CHECK(cufftExecZ2Z(plan, device_input, device_output, CUFFT_FORWARD));
    CUDA_CHECK(cudaMemcpy(cufft_output.data(), device_output, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));

    std::vector<double> reference_output(2 * N);
    for (size_t index = 0; index < N; ++index) {
        reference_output[2 * index] = cufft_output[index].x;
        reference_output[2 * index + 1] = cufft_output[index].y;
    }
    relative_l2_error =
        mathdx_examples::relative_l2_error(tile_output.begin(), tile_output.end(), reference_output.begin());
}

int main() {
    std::vector<float> input(2 * N, 0.0f);
    std::vector<float> output(2 * N, 0.0f);
    for (size_t index = 0; index < N; ++index) {
        input[2 * index] = 1.0f + static_cast<float>(index) / 8.0f;
        input[2 * index + 1] = 0.125f * static_cast<float>(static_cast<int>(index % 5) - 2);
    }

    mathdx_examples::device_buffer<float> device_input(input.size());
    mathdx_examples::device_buffer<float> device_output(output.size());
    device_input.copy_from_host(input.data(), input.size());

    mixed_precision_fft_kernel<<<1, 1>>>(device_input.data(), device_output.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    device_output.copy_to_host(output.data(), output.size());

    double cufft_relative_l2_error = 0.0;
    cufft_reference(input, output, cufft_relative_l2_error);
    std::printf("float I/O, double FFT; relative L2 error vs. double cuFFT: %.6g\n", cufft_relative_l2_error);
    return cufft_relative_l2_error < 1e-5 ? EXIT_SUCCESS : EXIT_FAILURE;
}
