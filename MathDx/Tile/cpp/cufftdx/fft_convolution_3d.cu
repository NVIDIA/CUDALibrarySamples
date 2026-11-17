// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdio>
#include <vector>

#include "cufft_convolution.hpp"
#include "cufftdx_tile.hpp"
#include "device_buffer.hpp"
#include "error_checks.hpp"
#include "error_metrics.hpp"

namespace ct = cuda::tiles;

// Zero-pad a 4^3 signal and a 5^3 filter to an 8^3 FFT. Their 8^3 linear
// convolution then fits exactly, with no circular wraparound.
constexpr size_t N = 8;
constexpr size_t signal_extent = 4;
constexpr size_t filter_extent = 5;
constexpr size_t output_extent = signal_extent + filter_extent - 1;
constexpr size_t element_count = N * N * N;
static_assert(output_extent == N, "the padded 3D convolution must fit the FFT extent");

constexpr size_t offset(size_t i, size_t j, size_t k) {
    return (i * N + j) * N + k;
}

std::vector<float> make_signal() {
    std::vector<float> signal(2 * element_count, 0.0f);
    for (size_t i = 0; i < signal_extent; ++i) {
        for (size_t j = 0; j < signal_extent; ++j) {
            for (size_t k = 0; k < signal_extent; ++k) {
                signal[2 * offset(i, j, k)] = static_cast<float>((i + 1) * (j + 2)) - 0.5f * static_cast<float>(k);
            }
        }
    }
    return signal;
}

std::vector<float> make_filter() {
    std::vector<float> filter(2 * element_count, 0.0f);
    for (size_t i = 0; i < filter_extent; ++i) {
        for (size_t j = 0; j < filter_extent; ++j) {
            for (size_t k = 0; k < filter_extent; ++k) {
                filter[2 * offset(i, j, k)] = 0.25f * static_cast<float>(i + 1) - 0.125f * static_cast<float>(j) +
                                              0.0625f * static_cast<float>(k);
            }
        }
    }
    return filter;
}

__tile_global__ void fft_convolution_3d_kernel(const float* signal, const float* filter, float* output) {
    using complex_shape = ct::shape<N, N, N, 2>;
    auto signal_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(signal), complex_shape {} }, complex_shape {} };
    auto filter_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(filter), complex_shape {} }, complex_shape {} };
    auto output_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(output), complex_shape {} }, complex_shape {} };

    // A 3D FFT is a one-dimensional FFT along each logical axis.
    auto signal_spectrum = cufftdx::tiles::fft(signal_view.load(0, 0, 0, 0), cufftdx::tiles::axis<0>);
    signal_spectrum = cufftdx::tiles::fft(signal_spectrum, cufftdx::tiles::axis<1>);
    signal_spectrum = cufftdx::tiles::fft(signal_spectrum, cufftdx::tiles::axis<2>);
    auto filter_spectrum = cufftdx::tiles::fft(filter_view.load(0, 0, 0, 0), cufftdx::tiles::axis<0>);
    filter_spectrum = cufftdx::tiles::fft(filter_spectrum, cufftdx::tiles::axis<1>);
    filter_spectrum = cufftdx::tiles::fft(filter_spectrum, cufftdx::tiles::axis<2>);

    // Complex values have trailing {real, imaginary} storage. Multiply the
    // spectra as complex values, rather than elementwise across that pair.
    using scalar_shape = ct::shape<N, N, N, 1>;
    const auto signal_real = ct::extract(signal_spectrum, scalar_shape {}, 0, 0, 0, 0);
    const auto signal_imag = ct::extract(signal_spectrum, scalar_shape {}, 0, 0, 0, 1);
    const auto filter_real = ct::extract(filter_spectrum, scalar_shape {}, 0, 0, 0, 0);
    const auto filter_imag = ct::extract(filter_spectrum, scalar_shape {}, 0, 0, 0, 1);
    const auto product_real = signal_real * filter_real - signal_imag * filter_imag;
    const auto product_imag = signal_real * filter_imag + signal_imag * filter_real;
    auto result = ct::cat<3>(product_real, product_imag);

    result = cufftdx::tiles::ifft(result, cufftdx::tiles::axis<0>);
    result = cufftdx::tiles::ifft(result, cufftdx::tiles::axis<1>);
    result = cufftdx::tiles::ifft(result, cufftdx::tiles::axis<2>);
    output_view.store(result * (1.0f / static_cast<float>(element_count)), 0, 0, 0, 0);
}

void cufft_reference(const std::vector<float>& signal,
                     const std::vector<float>& filter,
                     const std::vector<float>& tile_output,
                     double& relative_l2_error) {
    const auto cufft_output = mathdx_examples::cufft::convolve(mathdx_examples::cufft::complex_from_interleaved(signal),
                                                               mathdx_examples::cufft::complex_from_interleaved(filter),
                                                               { N, N, N });
    relative_l2_error = mathdx_examples::relative_l2_error(
        tile_output, mathdx_examples::cufft::interleaved(cufft_output, 1.0f / static_cast<float>(element_count)));
}

int main() {
    std::vector<float> signal = make_signal();
    std::vector<float> filter = make_filter();
    std::vector<float> output(2 * element_count, 0.0f);

    mathdx_examples::device_buffer<float> device_signal(signal.size());
    mathdx_examples::device_buffer<float> device_filter(filter.size());
    mathdx_examples::device_buffer<float> device_output(output.size());
    device_signal.copy_from_host(signal.data(), signal.size());
    device_filter.copy_from_host(filter.data(), filter.size());

    fft_convolution_3d_kernel<<<1, 1>>>(device_signal.data(), device_filter.data(), device_output.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    device_output.copy_to_host(output.data(), output.size());

    double cufft_relative_l2_error = 0.0;
    cufft_reference(signal, filter, output, cufft_relative_l2_error);
    std::printf("relative L2 error vs. cuFFT: %.6g\n", cufft_relative_l2_error);
    return cufft_relative_l2_error < 1e-5 ? EXIT_SUCCESS : EXIT_FAILURE;
}
