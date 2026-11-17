// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Timing and FLOP counts for the CUDA Tile C++ examples that report throughput
// against the batched host APIs. Everything else they share is in common.hpp.

#ifndef CUSOLVERDX_TILE_EXAMPLES_COMMON_PERF_MEASURE_HPP
#define CUSOLVERDX_TILE_EXAMPLES_COMMON_PERF_MEASURE_HPP

#include <cuda_runtime.h>

#include "common.hpp"

// Milliseconds per run, averaged after a warm-up. An in-place routine passes a
// reset that restores its input; the reset runs outside the timed region.
template <typename Launch, typename Reset>
double measure(Launch&& launch, Reset&& reset, cudaStream_t stream) {
    constexpr unsigned warmups = 1, runs = 3;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (unsigned i = 0; i < warmups; ++i) {
        reset(stream);
        launch(stream);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double total = 0.0;
    for (unsigned i = 0; i < runs; ++i) {
        reset(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch(stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        total += elapsed;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total / runs;
}

// The same, for a routine that leaves its input alone.
template <typename Launch>
double measure(Launch&& launch, cudaStream_t stream) {
    return measure(launch, [](cudaStream_t) {}, stream);
}

// Real-valued FLOP counts, matching common/measure.hpp in the cuSOLVERDx
// examples so the throughput numbers are comparable.
constexpr double flops_gemm(double rows, double cols, double depth) {
    return 2.0 * rows * cols * depth;
}
constexpr double flops_trsm(double order, double k) {
    return k * order * (order + 1) / 2.0 + k * order * (order - 1) / 2.0;
}
constexpr double flops_potrf(double order) {
    return order * (((1.0 / 6.0) * order + 0.5) * order + (1.0 / 3.0)) +
           order * (((1.0 / 6.0) * order) * order - (1.0 / 6.0));
}
constexpr double flops_potrs(double order, double k) {
    return 2.0 * flops_trsm(order, k);
}
constexpr double flops_geqrf(double rows, double cols) {
    return cols * (cols * (0.5 - (1.0 / 3.0) * cols + rows) + rows + 23.0 / 6.0) +
           cols * (cols * (0.5 - (1.0 / 3.0) * cols + rows) + 5.0 / 6.0);
}
constexpr double flops_unmqr_left(double rows, double cols, double k) {
    return 2.0 * k * (2.0 * rows - k) * cols;
}

#endif // CUSOLVERDX_TILE_EXAMPLES_COMMON_PERF_MEASURE_HPP
