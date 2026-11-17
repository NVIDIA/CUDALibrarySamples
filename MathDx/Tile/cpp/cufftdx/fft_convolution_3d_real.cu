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

// A 4^3 signal convolved with a 5^3 filter has an 8^3 linear result.
constexpr size_t N = 8;
constexpr size_t signal_extent = 4;
constexpr size_t filter_extent = 5;
constexpr size_t element_count = N * N * N;

constexpr size_t offset(size_t i, size_t j, size_t k) {
    return (i * N + j) * N + k;
}

__tile_global__ void fft_convolution_3d_real_kernel(const float* signal, const float* filter, float* output) {
    using real_shape = ct::shape<N, N, N>;
    auto signal_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(signal), real_shape {} }, real_shape {} };
    auto filter_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(filter), real_shape {} }, real_shape {} };
    auto output_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(output), real_shape {} }, real_shape {} };

    // rfft/irfft expose a full complex spectrum but pack/unpack the tilelib's
    // folded real layout internally. The remaining axes use C2C transforms.
    auto signal_spectrum =
        cufftdx::tiles::rfft(signal_view.load(0, 0, 0), cufftdx::tiles::axis<0>, cufftdx::tiles::full);
    signal_spectrum = cufftdx::tiles::fft(signal_spectrum, cufftdx::tiles::axis<1>);
    signal_spectrum = cufftdx::tiles::fft(signal_spectrum, cufftdx::tiles::axis<2>);
    auto filter_spectrum =
        cufftdx::tiles::rfft(filter_view.load(0, 0, 0), cufftdx::tiles::axis<0>, cufftdx::tiles::full);
    filter_spectrum = cufftdx::tiles::fft(filter_spectrum, cufftdx::tiles::axis<1>);
    filter_spectrum = cufftdx::tiles::fft(filter_spectrum, cufftdx::tiles::axis<2>);

    using scalar_shape = ct::shape<N, N, N, 1>;
    const auto signal_real = ct::extract(signal_spectrum, scalar_shape {}, 0, 0, 0, 0);
    const auto signal_imag = ct::extract(signal_spectrum, scalar_shape {}, 0, 0, 0, 1);
    const auto filter_real = ct::extract(filter_spectrum, scalar_shape {}, 0, 0, 0, 0);
    const auto filter_imag = ct::extract(filter_spectrum, scalar_shape {}, 0, 0, 0, 1);
    auto result = ct::cat<3>(signal_real * filter_real - signal_imag * filter_imag,
                             signal_real * filter_imag + signal_imag * filter_real);

    result = cufftdx::tiles::ifft(result, cufftdx::tiles::axis<2>);
    result = cufftdx::tiles::ifft(result, cufftdx::tiles::axis<1>);
    output_view.store(cufftdx::tiles::irfft(result, cufftdx::tiles::axis<0>, cufftdx::tiles::full) *
                          (1.0f / static_cast<float>(element_count)),
                      0,
                      0,
                      0);
}

void cufft_reference(const std::vector<float>& signal,
                     const std::vector<float>& filter,
                     const std::vector<float>& tile_output,
                     double& relative_l2_error) {
    const auto cufft_output = mathdx_examples::cufft::convolve(mathdx_examples::cufft::complex_from_real(signal),
                                                               mathdx_examples::cufft::complex_from_real(filter),
                                                               { N, N, N });
    relative_l2_error = mathdx_examples::relative_l2_error(
        tile_output, mathdx_examples::cufft::real_parts(cufft_output, 1.0f / static_cast<float>(element_count)));
}

int main() {
    std::vector<float> signal(element_count, 0.0f);
    std::vector<float> filter(element_count, 0.0f);
    std::vector<float> output(element_count, 0.0f);

    for (size_t i = 0; i < signal_extent; ++i) {
        for (size_t j = 0; j < signal_extent; ++j) {
            for (size_t k = 0; k < signal_extent; ++k) {
                signal[offset(i, j, k)] = static_cast<float>((i + 1) * (j + 2)) - 0.5f * static_cast<float>(k);
            }
        }
    }
    for (size_t i = 0; i < filter_extent; ++i) {
        for (size_t j = 0; j < filter_extent; ++j) {
            for (size_t k = 0; k < filter_extent; ++k) {
                filter[offset(i, j, k)] = 0.25f * static_cast<float>(i + 1) - 0.125f * static_cast<float>(j) +
                                          0.0625f * static_cast<float>(k);
            }
        }
    }

    mathdx_examples::device_buffer<float> device_signal(signal.size());
    mathdx_examples::device_buffer<float> device_filter(filter.size());
    mathdx_examples::device_buffer<float> device_output(output.size());
    device_signal.copy_from_host(signal.data(), signal.size());
    device_filter.copy_from_host(filter.data(), filter.size());

    fft_convolution_3d_real_kernel<<<1, 1>>>(device_signal.data(), device_filter.data(), device_output.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    device_output.copy_to_host(output.data(), output.size());

    double cufft_relative_l2_error = 0.0;
    cufft_reference(signal, filter, output, cufft_relative_l2_error);
    std::printf("8x8x8 real 3D convolution; relative L2 error vs. cuFFT: %.6g\n", cufft_relative_l2_error);
    return cufft_relative_l2_error < 1e-5 ? EXIT_SUCCESS : EXIT_FAILURE;
}
