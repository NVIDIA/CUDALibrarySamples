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

// A 64^3 signal convolved with a 65^3 filter has a 128^3 linear result.
// Splitting the transform lets each Tile kernel hold only a 128x128 plane or
// a 128-element line, instead of a complete 128^3 tensor.
constexpr size_t N = 128;
constexpr size_t signal_extent = 64;
constexpr size_t filter_extent = 65;
constexpr size_t element_count = N * N * N;
static_assert(signal_extent + filter_extent - 1 == N);

constexpr size_t offset(size_t i, size_t j, size_t k) {
    return (i * N + j) * N + k;
}

// One block per axis-2 coordinate. The plane's strided mapping preserves the
// global [axis0, axis1, axis2, complex] layout while presenting axis0/axis1
// as the two logical FFT dimensions.
__tile_global__ void forward_axes_01_kernel(const float* signal,
                                            const float* filter,
                                            float* signal_stage,
                                            float* filter_stage) {
    const auto k = ct::bid().x;
    using plane_shape = ct::shape<N, N, 2>;
    using plane_strides = ct::shape<2 * N * N, 2 * N, 1>;
    const auto mapping = ct::layout_strided_mapping { plane_shape {}, plane_strides {} };

    auto signal_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<8>(signal + 2 * k), mapping }, plane_shape {} };
    auto filter_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<8>(filter + 2 * k), mapping }, plane_shape {} };
    auto signal_stage_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<8>(signal_stage + 2 * k), mapping }, plane_shape {} };
    auto filter_stage_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<8>(filter_stage + 2 * k), mapping }, plane_shape {} };

    auto signal_spectrum = cufftdx::tiles::fft(signal_view.load(0, 0, 0), cufftdx::tiles::axis<0>);
    signal_spectrum = cufftdx::tiles::fft(signal_spectrum, cufftdx::tiles::axis<1>);
    auto filter_spectrum = cufftdx::tiles::fft(filter_view.load(0, 0, 0), cufftdx::tiles::axis<0>);
    filter_spectrum = cufftdx::tiles::fft(filter_spectrum, cufftdx::tiles::axis<1>);

    signal_stage_view.store(signal_spectrum, 0, 0, 0);
    filter_stage_view.store(filter_spectrum, 0, 0, 0);
}

// One block per (axis0, axis1) coordinate. axis2 is contiguous in the global
// layout, so this stage transforms it, multiplies the spectra, and immediately
// applies its inverse transform.
__tile_global__ void convolve_axis_2_kernel(const float* signal_stage, const float* filter_stage, float* result_stage) {
    const auto i = ct::bid().x;
    const auto j = ct::bid().y;
    using line_shape = ct::shape<N, 2>;
    const auto base = 2 * (i * N * N + j * N);
    auto signal_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(signal_stage + base), line_shape {} },
                             line_shape {} };
    auto filter_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(filter_stage + base), line_shape {} },
                             line_shape {} };
    auto result_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(result_stage + base), line_shape {} },
                             line_shape {} };

    const auto signal_spectrum = cufftdx::tiles::fft(signal_view.load(0, 0), cufftdx::tiles::axis<0>);
    const auto filter_spectrum = cufftdx::tiles::fft(filter_view.load(0, 0), cufftdx::tiles::axis<0>);
    const auto signal_real = ct::extract(signal_spectrum, ct::shape<N, 1> {}, 0, 0);
    const auto signal_imag = ct::extract(signal_spectrum, ct::shape<N, 1> {}, 0, 1);
    const auto filter_real = ct::extract(filter_spectrum, ct::shape<N, 1> {}, 0, 0);
    const auto filter_imag = ct::extract(filter_spectrum, ct::shape<N, 1> {}, 0, 1);
    const auto product = ct::cat<1>(signal_real * filter_real - signal_imag * filter_imag,
                                    signal_real * filter_imag + signal_imag * filter_real);
    result_view.store(cufftdx::tiles::ifft(product, cufftdx::tiles::axis<0>), 0, 0);
}

// One block per axis-2 coordinate. Finish the inverse transforms of the two
// axes fused by the first kernel, and apply the sole N^3 normalization.
__tile_global__ void inverse_axes_10_kernel(const float* result_stage, float* output) {
    const auto k = ct::bid().x;
    using plane_shape = ct::shape<N, N, 2>;
    using plane_strides = ct::shape<2 * N * N, 2 * N, 1>;
    const auto mapping = ct::layout_strided_mapping { plane_shape {}, plane_strides {} };
    auto result_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<8>(result_stage + 2 * k), mapping }, plane_shape {} };
    auto output_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<8>(output + 2 * k), mapping }, plane_shape {} };

    auto result = cufftdx::tiles::ifft(result_view.load(0, 0, 0), cufftdx::tiles::axis<1>);
    result = cufftdx::tiles::ifft(result, cufftdx::tiles::axis<0>);
    output_view.store(result * (1.0f / static_cast<float>(element_count)), 0, 0, 0);
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
    std::vector<float> signal(2 * element_count, 0.0f);
    std::vector<float> filter(2 * element_count, 0.0f);
    std::vector<float> output(2 * element_count, 0.0f);
    for (size_t i = 0; i < signal_extent; ++i) {
        for (size_t j = 0; j < signal_extent; ++j) {
            for (size_t k = 0; k < signal_extent; ++k) {
                signal[2 * offset(i, j, k)] =
                    0.03125f * static_cast<float>((i + 1) * (j + 2)) - 0.015625f * static_cast<float>(k);
            }
        }
    }
    for (size_t i = 0; i < filter_extent; ++i) {
        for (size_t j = 0; j < filter_extent; ++j) {
            for (size_t k = 0; k < filter_extent; ++k) {
                filter[2 * offset(i, j, k)] = 0.0078125f * static_cast<float>(i + 1) -
                                              0.00390625f * static_cast<float>(j) +
                                              0.001953125f * static_cast<float>(k);
            }
        }
    }

    mathdx_examples::device_buffer<float> device_signal(signal.size());
    mathdx_examples::device_buffer<float> device_filter(filter.size());
    mathdx_examples::device_buffer<float> device_signal_stage(signal.size());
    mathdx_examples::device_buffer<float> device_filter_stage(filter.size());
    mathdx_examples::device_buffer<float> device_result_stage(output.size());
    mathdx_examples::device_buffer<float> device_output(output.size());
    device_signal.copy_from_host(signal.data(), signal.size());
    device_filter.copy_from_host(filter.data(), filter.size());

    forward_axes_01_kernel<<<N, 1>>>(
        device_signal.data(), device_filter.data(), device_signal_stage.data(), device_filter_stage.data());
    convolve_axis_2_kernel<<<dim3(N, N), 1>>>(
        device_signal_stage.data(), device_filter_stage.data(), device_result_stage.data());
    inverse_axes_10_kernel<<<N, 1>>>(device_result_stage.data(), device_output.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    device_output.copy_to_host(output.data(), output.size());

    double cufft_relative_l2_error = 0.0;
    cufft_reference(signal, filter, output, cufft_relative_l2_error);
    std::printf("128x128x128 split 3D convolution; relative L2 error vs. cuFFT: %.6g\n", cufft_relative_l2_error);
    return cufft_relative_l2_error < 1e-5 ? EXIT_SUCCESS : EXIT_FAILURE;
}
