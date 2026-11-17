// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include <cstdio>
#include <vector>

#include "cufft_convolution.hpp"
#include "cufftdx_tile.hpp"
#include "device_buffer.hpp"
#include "error_checks.hpp"
#include "error_metrics.hpp"

namespace ct = cuda::tiles;

// A 4^5 signal and a 5^5 filter fit exactly in an 8^5 padded linear
// convolution. The logical tensor is [i,j,k,l,m,{real,imaginary}], stored in
// row-major order: m is the fastest spatial dimension and each complex value
// occupies two adjacent floats.
//
// Tile FFT calls accept at most three axes, so the 5D DFT is factored into
// 64 three-dimensional transforms over (i,j,k), followed by 512
// two-dimensional transforms over (l,m). The inverse performs those groups
// in reverse order. The reshapes below only change each tile's logical view;
// they do not copy global-memory data.
constexpr size_t N = 8;
constexpr size_t signal_extent = 4;
constexpr size_t filter_extent = 5;
constexpr size_t output_extent = signal_extent + filter_extent - 1;
constexpr size_t element_count = N * N * N * N * N;
static_assert(output_extent == N, "the padded 5D convolution must fit the FFT extent");

constexpr size_t offset(size_t i, size_t j, size_t k, size_t l, size_t m) {
    return ((((i * N + j) * N + k) * N + l) * N + m);
}

std::vector<float> make_signal() {
    std::vector<float> signal(2 * element_count, 0.0f);
    for (size_t i = 0; i < signal_extent; ++i) {
        for (size_t j = 0; j < signal_extent; ++j) {
            for (size_t k = 0; k < signal_extent; ++k) {
                for (size_t l = 0; l < signal_extent; ++l) {
                    for (size_t m = 0; m < signal_extent; ++m) {
                        signal[2 * offset(i, j, k, l, m)] =
                            static_cast<float>(i + 1) + 0.5f * static_cast<float>(j) - 0.25f * static_cast<float>(k) +
                            0.125f * static_cast<float>(l) - 0.0625f * static_cast<float>(m);
                    }
                }
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
                for (size_t l = 0; l < filter_extent; ++l) {
                    for (size_t m = 0; m < filter_extent; ++m) {
                        filter[2 * offset(i, j, k, l, m)] =
                            0.25f * static_cast<float>(i + 1) - 0.125f * static_cast<float>(j) +
                            0.0625f * static_cast<float>(k) - 0.03125f * static_cast<float>(l) +
                            0.015625f * static_cast<float>(m);
                    }
                }
            }
        }
    }
    return filter;
}

__tile_global__ void forward_axes_012_kernel(const float* signal,
                                             const float* filter,
                                             float* signal_stage,
                                             float* filter_stage) {
    const size_t l = ct::bid().x;
    const size_t m = ct::bid().y;

    // This block owns one fixed (l,m) pair. It takes a strided 3D slice:
    // (i,j,k,l,m) for all i,j,k. The explicit strides skip to the next i, j,
    // or k while retaining the fixed l and m coordinates.
    using logical_slice_shape = ct::shape<N, N, N, 1, 1, 2>;
    using fft_tile_shape = ct::shape<N, N, N, 2>;
    using logical_slice_strides = ct::shape<N * N * N * N * 2, N * N * N * 2, N * N * 2, N * 2, 2, 1>;
    const auto mapping = ct::layout_strided_mapping { logical_slice_shape {}, logical_slice_strides {} };
    const size_t slice_offset = 2 * (l * N + m);

    auto signal_view = ct::partition_view { ct::tensor_span { ct::assume_aligned<8>(signal + slice_offset), mapping },
                                            logical_slice_shape {} };
    auto filter_view = ct::partition_view { ct::tensor_span { ct::assume_aligned<8>(filter + slice_offset), mapping },
                                            logical_slice_shape {} };
    auto signal_stage_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<8>(signal_stage + slice_offset), mapping },
                             logical_slice_shape {} };
    auto filter_stage_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<8>(filter_stage + slice_offset), mapping },
                             logical_slice_shape {} };

    auto signal_tile = ct::reshape(signal_view.load(0, 0, 0, 0, 0, 0), fft_tile_shape {});
    auto filter_tile = ct::reshape(filter_view.load(0, 0, 0, 0, 0, 0), fft_tile_shape {});
    signal_tile = cufftdx::tiles::fft(signal_tile, cufftdx::tiles::axis<0>);
    signal_tile = cufftdx::tiles::fft(signal_tile, cufftdx::tiles::axis<1>);
    signal_tile = cufftdx::tiles::fft(signal_tile, cufftdx::tiles::axis<2>);
    filter_tile = cufftdx::tiles::fft(filter_tile, cufftdx::tiles::axis<0>);
    filter_tile = cufftdx::tiles::fft(filter_tile, cufftdx::tiles::axis<1>);
    filter_tile = cufftdx::tiles::fft(filter_tile, cufftdx::tiles::axis<2>);
    signal_stage_view.store(ct::reshape(signal_tile, logical_slice_shape {}), 0, 0, 0, 0, 0, 0);
    filter_stage_view.store(ct::reshape(filter_tile, logical_slice_shape {}), 0, 0, 0, 0, 0, 0);
}

__tile_global__ void convolve_axes_34_kernel(const float* signal_stage,
                                             const float* filter_stage,
                                             float* result_stage) {
    const size_t i = ct::bid().x;
    const size_t j = ct::bid().y;
    const size_t k = ct::bid().z;

    // This block owns one fixed (i,j,k) triple and takes the contiguous 2D
    // (l,m) plane. Combined with the first kernel's strided (i,j,k) slices,
    // this covers every axis of the original 5D tensor.
    using logical_slice_shape = ct::shape<1, 1, 1, N, N, 2>;
    using fft_tile_shape = ct::shape<N, N, 2>;
    using logical_slice_strides = ct::shape<N * N * N * N * 2, N * N * N * 2, N * N * 2, N * 2, 2, 1>;
    const auto mapping = ct::layout_strided_mapping { logical_slice_shape {}, logical_slice_strides {} };
    const size_t slice_offset = 2 * ((i * N + j) * N + k) * N * N;

    auto signal_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(signal_stage + slice_offset), mapping },
                             logical_slice_shape {} };
    auto filter_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(filter_stage + slice_offset), mapping },
                             logical_slice_shape {} };
    auto result_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<16>(result_stage + slice_offset), mapping },
                             logical_slice_shape {} };

    auto signal_tile = ct::reshape(signal_view.load(0, 0, 0, 0, 0, 0), fft_tile_shape {});
    auto filter_tile = ct::reshape(filter_view.load(0, 0, 0, 0, 0, 0), fft_tile_shape {});
    signal_tile = cufftdx::tiles::fft(signal_tile, cufftdx::tiles::axis<0>);
    signal_tile = cufftdx::tiles::fft(signal_tile, cufftdx::tiles::axis<1>);
    filter_tile = cufftdx::tiles::fft(filter_tile, cufftdx::tiles::axis<0>);
    filter_tile = cufftdx::tiles::fft(filter_tile, cufftdx::tiles::axis<1>);

    using scalar_shape = ct::shape<N, N, 1>;
    const auto signal_real = ct::extract(signal_tile, scalar_shape {}, 0, 0, 0);
    const auto signal_imag = ct::extract(signal_tile, scalar_shape {}, 0, 0, 1);
    const auto filter_real = ct::extract(filter_tile, scalar_shape {}, 0, 0, 0);
    const auto filter_imag = ct::extract(filter_tile, scalar_shape {}, 0, 0, 1);
    auto result = ct::cat<2>(signal_real * filter_real - signal_imag * filter_imag,
                             signal_real * filter_imag + signal_imag * filter_real);

    result = cufftdx::tiles::ifft(result, cufftdx::tiles::axis<1>);
    result = cufftdx::tiles::ifft(result, cufftdx::tiles::axis<0>);
    result_view.store(ct::reshape(result, logical_slice_shape {}), 0, 0, 0, 0, 0, 0);
}

__tile_global__ void inverse_axes_210_kernel(const float* result_stage, float* output) {
    const size_t l = ct::bid().x;
    const size_t m = ct::bid().y;

    // Return to the same fixed-(l,m), strided (i,j,k) geometry as the first
    // kernel. The N^5 scale accounts for all five unnormalized inverse FFTs.
    using logical_slice_shape = ct::shape<N, N, N, 1, 1, 2>;
    using fft_tile_shape = ct::shape<N, N, N, 2>;
    using logical_slice_strides = ct::shape<N * N * N * N * 2, N * N * N * 2, N * N * 2, N * 2, 2, 1>;
    const auto mapping = ct::layout_strided_mapping { logical_slice_shape {}, logical_slice_strides {} };
    const size_t slice_offset = 2 * (l * N + m);

    auto result_view =
        ct::partition_view { ct::tensor_span { ct::assume_aligned<8>(result_stage + slice_offset), mapping },
                             logical_slice_shape {} };
    auto output_view = ct::partition_view { ct::tensor_span { ct::assume_aligned<8>(output + slice_offset), mapping },
                                            logical_slice_shape {} };
    auto result = ct::reshape(result_view.load(0, 0, 0, 0, 0, 0), fft_tile_shape {});
    result = cufftdx::tiles::ifft(result, cufftdx::tiles::axis<2>);
    result = cufftdx::tiles::ifft(result, cufftdx::tiles::axis<1>);
    result = cufftdx::tiles::ifft(result, cufftdx::tiles::axis<0>);
    output_view.store(
        ct::reshape(result * (1.0f / static_cast<float>(element_count)), logical_slice_shape {}), 0, 0, 0, 0, 0, 0);
}

void cufft_reference(const std::vector<float>& signal,
                     const std::vector<float>& filter,
                     const std::vector<float>& tile_output,
                     double& relative_l2_error) {
    const auto signal_frequency = mathdx_examples::cufft::complex_from_interleaved(signal);
    const auto filter_frequency = mathdx_examples::cufft::complex_from_interleaved(filter);
    std::vector<cufftComplex> cufft_output(element_count);

    // cuFFT plans are limited to three dimensions. PlanMany supplies N^2
    // strided 3D transforms for axes 0..2, then N^3 contiguous 2D transforms
    // for axes 3..4; their composition is the full 5D DFT.
    int dimensions_012[] = { static_cast<int>(N), static_cast<int>(N), static_cast<int>(N) };
    int dimensions_34[] = { static_cast<int>(N), static_cast<int>(N) };
    cufftHandle axes_012_plan = 0;
    cufftHandle axes_34_plan = 0;
    cufftHandle signal_axes_012_forward_plan = 0;
    cufftComplex* device_signal = nullptr;
    cufftComplex* device_filter = nullptr;
    CUDA_CHECK(cudaMalloc(&device_signal, element_count * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&device_filter, element_count * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMemcpy(
        device_signal, signal_frequency.data(), element_count * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        device_filter, filter_frequency.data(), element_count * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    CUFFT_CHECK(cufftPlanMany(
        &axes_012_plan, 3, dimensions_012, dimensions_012, N * N, 1, dimensions_012, N * N, 1, CUFFT_C2C, N * N));
    CUFFT_CHECK(cufftPlanMany(
        &axes_34_plan, 2, dimensions_34, dimensions_34, 1, N * N, dimensions_34, 1, N * N, CUFFT_C2C, N * N * N));

    // cuFFT passes global scalar offsets for this batched 3D plan, so the
    // signal and filter spectra share the same `offset`. Applying the store
    // callback to the final forward axes-012 transform performs the complete
    // frequency-domain multiplication with one plan.
    CUFFT_CHECK(cufftCreate(&signal_axes_012_forward_plan));
    CUFFT_CHECK(cufftXtSetJITCallback(signal_axes_012_forward_plan,
                                      "cufftdx_tile_store_multiply",
                                      cufft_lto_store_multiply,
                                      sizeof(cufft_lto_store_multiply),
                                      CUFFT_CB_ST_COMPLEX,
                                      reinterpret_cast<void**>(&device_filter)));
    size_t work_size = 0;
    CUFFT_CHECK(cufftMakePlanMany(signal_axes_012_forward_plan,
                                  3,
                                  dimensions_012,
                                  dimensions_012,
                                  N * N,
                                  1,
                                  dimensions_012,
                                  N * N,
                                  1,
                                  CUFFT_C2C,
                                  N * N,
                                  &work_size));

    CUFFT_CHECK(cufftExecC2C(axes_34_plan, device_signal, device_signal, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(axes_34_plan, device_filter, device_filter, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(axes_012_plan, device_filter, device_filter, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(signal_axes_012_forward_plan, device_signal, device_signal, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(axes_34_plan, device_signal, device_signal, CUFFT_INVERSE));
    CUFFT_CHECK(cufftExecC2C(axes_012_plan, device_signal, device_signal, CUFFT_INVERSE));
    CUDA_CHECK(
        cudaMemcpy(cufft_output.data(), device_signal, element_count * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    CUFFT_CHECK(cufftDestroy(axes_012_plan));
    CUFFT_CHECK(cufftDestroy(axes_34_plan));
    CUFFT_CHECK(cufftDestroy(signal_axes_012_forward_plan));
    CUDA_CHECK(cudaFree(device_signal));
    CUDA_CHECK(cudaFree(device_filter));

    const auto reference_output =
        mathdx_examples::cufft::interleaved(cufft_output, 1.0f / static_cast<float>(element_count));
    relative_l2_error =
        mathdx_examples::relative_l2_error(tile_output.begin(), tile_output.end(), reference_output.begin());
}

int main() {
    std::vector<float> signal = make_signal();
    std::vector<float> filter = make_filter();
    std::vector<float> signal_stage(2 * element_count, 0.0f);
    std::vector<float> filter_stage(2 * element_count, 0.0f);
    std::vector<float> result_stage(2 * element_count, 0.0f);
    std::vector<float> output(2 * element_count, 0.0f);

    mathdx_examples::device_buffer<float> device_signal(signal.size());
    mathdx_examples::device_buffer<float> device_filter(filter.size());
    mathdx_examples::device_buffer<float> device_signal_stage(signal_stage.size());
    mathdx_examples::device_buffer<float> device_filter_stage(filter_stage.size());
    mathdx_examples::device_buffer<float> device_result_stage(result_stage.size());
    mathdx_examples::device_buffer<float> device_output(output.size());
    device_signal.copy_from_host(signal.data(), signal.size());
    device_filter.copy_from_host(filter.data(), filter.size());

    forward_axes_012_kernel<<<dim3(N, N), 1>>>(
        device_signal.data(), device_filter.data(), device_signal_stage.data(), device_filter_stage.data());
    convolve_axes_34_kernel<<<dim3(N, N, N), 1>>>(
        device_signal_stage.data(), device_filter_stage.data(), device_result_stage.data());
    inverse_axes_210_kernel<<<dim3(N, N), 1>>>(device_result_stage.data(), device_output.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    device_output.copy_to_host(output.data(), output.size());

    double cufft_relative_l2_error = 0.0;
    cufft_reference(signal, filter, output, cufft_relative_l2_error);
    std::printf("8^5 convolution; relative L2 error vs. cuFFT: %.6g\n", cufft_relative_l2_error);
    return cufft_relative_l2_error < 1e-5 ? EXIT_SUCCESS : EXIT_FAILURE;
}
