/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <chrono>

#include <cuda_runtime.h>
#include <curand.h>

#include <nvcomp/crc32.h>

#include "benchmark_common.h"

#define CURAND_CHECK(call) do { \
    curandStatus_t err = call; \
    if (err != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "cuRAND error: %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

using nvcomp::cudaMallocSafe;

size_t next_lower_power_of_two(size_t n)
{
    if (n == 0) {
        return 0;
    }
    // Find the highest set bit
    size_t power = 1;
    while (power <= n / 2) {
        power <<= 1;
    }
    return power;
}

bool is_power_of_two(size_t n)
{
    return n > 0 && (n & (n - 1)) == 0;
}

// Binary size prefixes.
constexpr size_t Ki = 1024;
constexpr size_t Mi = 1024 * Ki;
constexpr size_t Gi = 1024 * Mi;

// Decimal size prefix, used for throughput reporting.
constexpr size_t G = 1000 * 1000 * 1000;

std::string human_readable_size(size_t size)
{
    if (size < Ki) {
        return std::to_string(size);
    }
    if (size < Mi) {
        return std::to_string(size / Ki) + "Ki";
    }
    if (size < Gi) {
        return std::to_string(size / Mi) + "Mi";
    }
    return std::to_string(size / Gi) + "Gi";
}

std::string human_readable_byte_size(size_t size)
{
    return human_readable_size(size) + "B";
}

std::pair<std::string, size_t> parse_command_line_args(int argc, char* argv[])
{
    // Get output filename
    std::string output_filename = "crc32_perf.csv";
    if (argc > 1) {
        output_filename = argv[1];
    }

    // Query GPU memory
    size_t free_mem = 0;
    size_t total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    std::cout << "Total GPU memory: " << total_mem << " (" << human_readable_byte_size(total_mem) << ")" << std::endl;
    std::cout << "Free GPU memory: " << free_mem << " (" << human_readable_byte_size(free_mem) << ")" << std::endl;

    // Reserve some memory for other allocations (kernels, temporary buffers, etc.)
    // Use 90% of total_mem to be safe.
    size_t considered_mem = size_t(double(total_mem) * 0.9);

    // Calculate storage size as next lower power of two
    size_t default_storage_size = next_lower_power_of_two(considered_mem);

    size_t storage_size = default_storage_size;

    if (argc > 2) {
        // Use override from command line argument
        storage_size = std::stoull(argv[2]);

        if (!is_power_of_two(storage_size)) {
            std::stringstream ss;
            ss << "Override storage size (" << human_readable_byte_size(storage_size)
               << ") is not a power of two";
            throw std::logic_error(ss.str());
        }

        if (storage_size > default_storage_size) {
            std::cerr << "Warning: Override storage size (" << human_readable_byte_size(storage_size)
                     << ") exceeds default storage size (" << human_readable_byte_size(default_storage_size) << ")" << std::endl;
        }
    }

    std::cout << "Using storage size: " << human_readable_byte_size(storage_size) << std::endl;
    return {output_filename, storage_size};
}

struct ThroughputTable
{
    std::vector<std::vector<double>> contents{};
    std::vector<size_t> buf_sizes{};
    std::vector<size_t> buf_counts{};
};

constexpr size_t MinBufSize = Ki;
constexpr size_t MaxBufSize = 4 * Gi;
constexpr size_t MinBufCount = 1;
constexpr size_t MaxBufCount = Mi;

ThroughputTable make_throughput_table(size_t storage_size)
{
    ThroughputTable table;
    // Generate all valid buf_size values (powers of two from 1 KiB to 4 GiB)
    for (size_t buf_size = MinBufSize; buf_size <= MaxBufSize; buf_size <<= 1) {
        if (buf_size * MinBufCount > storage_size) {
            break;
        }
        table.buf_sizes.push_back(buf_size);
    }

    // Generate all valid buf_count values (powers of two from 1 to 1M)
    for (size_t buf_count = MinBufCount; buf_count <= MaxBufCount; buf_count <<= 1) {
        if (buf_count * MinBufSize > storage_size) {
            break;
        }
        table.buf_counts.push_back(buf_count);
    }

    table.contents.resize(table.buf_sizes.size());
    for (auto& row : table.contents) {
        row.resize(table.buf_counts.size(), -1.0); // -1 indicates missing measurement
    }
    return table;
}

void initialize_data(void* storage_buf, size_t storage_size)
{
    constexpr size_t PseudorandomSeed = 42;

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, PseudorandomSeed));

    // Fill buffer with random data (treating as uint32_t array)
    size_t num_uint32 = storage_size / sizeof(uint32_t);
    CURAND_CHECK(curandGenerate(gen, static_cast<uint32_t*>(storage_buf), num_uint32));

    assert(storage_size % sizeof(uint32_t) == 0);
    CURAND_CHECK(curandDestroyGenerator(gen));
}

struct BenchmarkContext
{
    void* storage_buf_{};
    void** chunk_ptrs_h_{};
    void** chunk_ptrs_d_{};
    size_t* chunk_sizes_h_{};
    size_t* chunk_sizes_d_{};
    uint32_t* crc_results_d_{};
    cudaStream_t stream_{};
    cudaEvent_t start_{};
    cudaEvent_t stop_{};

    // Quell compiler warnings.
    BenchmarkContext(const BenchmarkContext&) = delete;
    BenchmarkContext& operator=(const BenchmarkContext&) = delete;

    BenchmarkContext(void* storage_buf, size_t max_buf_count)
    {
        storage_buf_ = storage_buf;
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));

        // Allocate chunk arrays
        CUDA_CHECK(cudaHostAlloc(&chunk_ptrs_h_, max_buf_count * sizeof(void*), cudaHostAllocDefault));
        CUDA_CHECK(cudaMallocSafe(&chunk_ptrs_d_, max_buf_count * sizeof(void*)));
        CUDA_CHECK(cudaHostAlloc(&chunk_sizes_h_, max_buf_count * sizeof(size_t), cudaHostAllocDefault));
        CUDA_CHECK(cudaMallocSafe(&chunk_sizes_d_, max_buf_count * sizeof(size_t)));
        CUDA_CHECK(cudaMallocSafe(&crc_results_d_, max_buf_count * sizeof(uint32_t)));
    }

    ~BenchmarkContext()
    {
        cudaFree(storage_buf_);
        cudaFreeHost(chunk_ptrs_h_);
        cudaFree(chunk_ptrs_d_);
        cudaFreeHost(chunk_sizes_h_);
        cudaFree(chunk_sizes_d_);
        cudaFree(crc_results_d_);
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
        cudaStreamDestroy(stream_);
    }

    double run_single_benchmark(size_t buf_size, size_t buf_count);
};

double BenchmarkContext::run_single_benchmark(size_t buf_size, size_t buf_count)
{
    std::cout << "Testing buf_size=" << human_readable_byte_size(buf_size) <<
        ", buf_count=" << human_readable_size(buf_count) << std::endl;

    // Set up chunk pointers (i-th chunk at storage_buf + i * buf_size)
    for (size_t i = 0; i < buf_count; ++i) {
        chunk_sizes_h_[i] = buf_size;
        chunk_ptrs_h_[i] = static_cast<uint8_t*>(storage_buf_) + i * buf_size;
    }

    CUDA_CHECK(cudaMemcpyAsync(chunk_ptrs_d_, chunk_ptrs_h_,
                         buf_count * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(chunk_sizes_d_, chunk_sizes_h_,
                         buf_count * sizeof(size_t), cudaMemcpyHostToDevice));

    // Search for optimal kernel configuration
    nvcompCRC32KernelConf_t kernel_conf;
    nvcompStatus_t status = nvcompBatchedCRC32SearchConf(
        chunk_ptrs_d_,
        chunk_sizes_d_,
        buf_count,
        crc_results_d_,
        nvcompCRC32,
        &kernel_conf,
        stream_);

    if (status != nvcompSuccess) {
        std::cerr << "Error in nvcompBatchedCRC32SearchConf: " << status << std::endl;
        return -std::numeric_limits<double>::infinity();
    }

    // Timing run
    CUDA_CHECK(cudaEventRecord(start_, stream_));

    status = nvcompBatchedCRC32Async(
        chunk_ptrs_d_,
        chunk_sizes_d_,
        buf_count,
        crc_results_d_,
        nvcompBatchedCRC32Opts_t{nvcompCRC32, kernel_conf, {}},
        nvcompCRC32OnlySegment,
        /*device_statuses=*/nullptr,
        stream_);

    CUDA_CHECK(cudaEventRecord(stop_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    if (status != nvcompSuccess) {
        std::cerr << "Error in nvcompBatchedCRC32Async: " << status << std::endl;
        return -std::numeric_limits<double>::infinity();
    }

    // Get timing result and calculate throughput
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));

    // Calculate throughput in GB/s
    size_t total_bytes = buf_size * buf_count;
    double elapsed_seconds = double(elapsed_ms) / 1000.0;
    double throughput_gb_per_sec = (total_bytes / static_cast<double>(G)) / elapsed_seconds;

    return throughput_gb_per_sec;
}

void write_throughput_table_to_csv(const std::string& output_filename, const ThroughputTable& table)
{
    std::ofstream csv_file(output_filename);
    if (!csv_file) {
        std::cerr << "Error: Could not open output file " << output_filename << std::endl;
        throw std::runtime_error("Failed to open output file");
    }

    // Write header row
    csv_file << "Size \\ Count";
    for (size_t buf_count : table.buf_counts) {
        csv_file << "," << human_readable_size(buf_count);
    }
    csv_file << "\n";

    // Write data rows
    for (size_t size_idx = 0; size_idx < table.buf_sizes.size(); ++size_idx) {
        size_t buf_size = table.buf_sizes[size_idx];

        csv_file << human_readable_byte_size(buf_size);
        for (size_t count_idx = 0; count_idx < table.buf_counts.size(); ++count_idx) {
            csv_file << "," << table.contents[size_idx][count_idx];
        }
        csv_file << "\n";
    }

    std::cout << "Results written to " << output_filename << std::endl;
}

int main(int argc, char* argv[])
{
    try {
        auto args = parse_command_line_args(argc, argv);
        const std::string& output_filename = args.first;
        const size_t storage_size = args.second;

        ThroughputTable table = make_throughput_table(storage_size);

        void* storage_buf = nullptr;
        CUDA_CHECK(cudaMallocSafe(&storage_buf, storage_size));

        initialize_data(storage_buf, storage_size);

        BenchmarkContext ctx{storage_buf, table.buf_counts.back()};

        std::cout << "Running benchmarks..." << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Main benchmark loop
        for (size_t size_idx = 0; size_idx < table.buf_sizes.size(); ++size_idx) {
            size_t buf_size = table.buf_sizes[size_idx];

            for (size_t count_idx = 0; count_idx < table.buf_counts.size(); ++count_idx) {
                size_t buf_count = table.buf_counts[count_idx];

                // Check if buf_size * buf_count <= storage_size
                if (buf_size > storage_size || buf_count > storage_size / buf_size) {
                    continue; // Skip this combination
                }

                table.contents[size_idx][count_idx] =
                    ctx.run_single_benchmark(buf_size, buf_count);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Total benchmark time: " << duration.count() / 1000.0 << " seconds" << std::endl;

        write_throughput_table_to_csv(output_filename, table);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}