// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdio>
#include <vector>

#include "cufft_convolution.hpp"
#include "cufftdx_tile.hpp"
#include "cufftdx_tile_config.hpp"
#include "device_buffer.hpp"
#include "error_checks.hpp"
#include "error_metrics.hpp"
#include "sm_arch.hpp"

namespace ct = cuda::tiles;

// N is larger than signal_size + filter_size - 1, so the FFT implements a
// linear (rather than circular) convolution of the zero-padded inputs.
constexpr size_t N = 16;
constexpr size_t signal_size = 8;
constexpr size_t filter_size = 5;
constexpr size_t output_size = signal_size + filter_size - 1;
// Rows of signals and filters; a multiple of every suggested batch per CTA.
constexpr size_t batch = 1024;

// Each device pass compiles the configuration its architecture selects. The
// three transforms share one (rows, N) tile, so one entry configures the
// kernel: the inverse transform's, measured closest to the fused kernel's
// optimum. An architecture without an entry gets the one-row, one-warp fallback.
constexpr cufftdx::tiles::tile_config_t compiled_suggested =
    cufftdx::tiles::suggested_tile_config<float>(cufftdx::tiles::transform_t::irfft,
                                                 1,
                                                 0,
                                                 N,
                                                 mathdx_examples::compiled_sm_arch());

[[cutile::hint(0, num_worker_warps_per_cta = compiled_suggested.num_worker_warps_per_cta)]]
__tile_global__ void fft_convolution_kernel(const float* signals, const float* filters, float* outputs) {
    // One (batches_per_block, N) tile of the (batch, N) arrays per block.
    constexpr size_t batches_per_block = compiled_suggested.batches_per_block;
    using array_shape = ct::shape<batch, N>;
    using tile_shape = ct::shape<batches_per_block, N>;
    const auto block = ct::bid().x;
    auto signal_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(signals), array_shape {} }, tile_shape {} };
    auto filter_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(filters), array_shape {} }, tile_shape {} };
    auto output_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(outputs), array_shape {} }, tile_shape {} };

    // The tilelib packs the full public spectrum into its folded R2C layout.
    const auto signal_spectrum =
        cufftdx::tiles::rfft(signal_view.load(block, 0), cufftdx::tiles::axis<1>, cufftdx::tiles::full);
    const auto filter_spectrum =
        cufftdx::tiles::rfft(filter_view.load(block, 0), cufftdx::tiles::axis<1>, cufftdx::tiles::full);

    // Complex values are stored as [..., {real, imaginary}]. Form the complex
    // product with Tile elementwise operators rather than treating the pairs
    // as independent real values.
    using scalar_shape = ct::shape<batches_per_block, N, 1>;
    const auto signal_real = ct::extract(signal_spectrum, scalar_shape {}, 0, 0, 0);
    const auto signal_imag = ct::extract(signal_spectrum, scalar_shape {}, 0, 0, 1);
    const auto filter_real = ct::extract(filter_spectrum, scalar_shape {}, 0, 0, 0);
    const auto filter_imag = ct::extract(filter_spectrum, scalar_shape {}, 0, 0, 1);
    const auto product_real = signal_real * filter_real - signal_imag * filter_imag;
    const auto product_imag = signal_real * filter_imag + signal_imag * filter_real;
    const auto product = ct::cat<2>(product_real, product_imag);

    output_view.store(
        cufftdx::tiles::irfft(product, cufftdx::tiles::axis<1>, cufftdx::tiles::full) * (1.0f / N), block, 0);
}

void cufft_reference(const std::vector<float>& signals,
                     const std::vector<float>& filters,
                     const std::vector<float>& tile_outputs,
                     double& relative_l2_error) {
    const auto cufft_output = mathdx_examples::cufft::convolve(mathdx_examples::cufft::complex_from_real(signals),
                                                               mathdx_examples::cufft::complex_from_real(filters),
                                                               { N },
                                                               static_cast<int>(batch));
    relative_l2_error = mathdx_examples::relative_l2_error(
        tile_outputs, mathdx_examples::cufft::real_parts(cufft_output, 1.0f / static_cast<float>(N)));
}

int main() {
    // Real inputs are padded to N so their linear convolution does not wrap.
    // Row r holds the signal scaled by r + 1 and the same filter.
    std::vector<float> signals(batch * N, 0.0f);
    std::vector<float> filters(batch * N, 0.0f);
    std::vector<float> outputs(batch * N, 0.0f);
    constexpr float signal_values[signal_size] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    constexpr float filter_values[filter_size] = { 0.5f, -1.0f, 0.25f, 2.0f, -0.5f };

    for (size_t row = 0; row < batch; ++row) {
        for (size_t i = 0; i < signal_size; ++i) {
            signals[row * N + i] = static_cast<float>(row + 1) * signal_values[i];
        }
        for (size_t i = 0; i < filter_size; ++i) {
            filters[row * N + i] = filter_values[i];
        }
    }

    // The launch needs the batches per CTA the device code was compiled with: the
    // entry of the loaded kernel image's architecture, not the device's.
    const int sm_arch = mathdx_examples::loaded_sm_arch(fft_convolution_kernel);
    const cufftdx::tiles::tile_config_t suggested =
        cufftdx::tiles::suggested_tile_config<float>(cufftdx::tiles::transform_t::irfft, 1, 0, N, sm_arch);
    const mathdx_examples::cc_t cc(sm_arch);
    std::printf("sm_%d%d: %d batches per CTA, %d warp(s)%s\n",
                cc.major,
                cc.minor,
                suggested.batches_per_block,
                suggested.num_worker_warps_per_cta,
                suggested.found ? "" : " (no suggested tile config for this size)");

    mathdx_examples::device_buffer<float> device_signals(signals.size());
    mathdx_examples::device_buffer<float> device_filters(filters.size());
    mathdx_examples::device_buffer<float> device_outputs(outputs.size());
    device_signals.copy_from_host(signals.data(), signals.size());
    device_filters.copy_from_host(filters.data(), filters.size());

    fft_convolution_kernel<<<batch / suggested.batches_per_block, 1>>>(
        device_signals.data(), device_filters.data(), device_outputs.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    device_outputs.copy_to_host(outputs.data(), outputs.size());

    std::printf("1D linear convolution, row 0 (first %zu values):\n", output_size);
    for (size_t i = 0; i < output_size; ++i) {
        std::printf("y[%zu] = %8.5f\n", i, outputs[i]);
    }
    double cufft_relative_l2_error = 0.0;
    cufft_reference(signals, filters, outputs, cufft_relative_l2_error);
    std::printf("relative L2 error vs. cuFFT: %.6g\n", cufft_relative_l2_error);
    return cufft_relative_l2_error < 1e-5 ? EXIT_SUCCESS : EXIT_FAILURE;
}
