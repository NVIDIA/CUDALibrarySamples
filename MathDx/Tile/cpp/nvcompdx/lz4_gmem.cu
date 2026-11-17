// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// LZ4 compression straight from and to global memory with the CUDA Tile
// C++ nvCOMPDx API. The chunks, the compressed payloads and the restored chunks
// stay in global memory. Compression reads the chunk and writes the payload slot through pointers.
// Decompression reads the payload slot and writes the restored chunk. Only byte counts are tiles.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include "device_buffer.hpp"
#include "error_checks.hpp"
#include "nvcompdx_tile.hpp"

namespace ct = cuda::tiles;
namespace cn = nvcompdx::tiles;
using mathdx_examples::device_buffer;

using element_t = std::uint8_t;
constexpr cn::algorithm algorithm = cn::lz4;
constexpr size_t chunks = 4;
constexpr size_t elements = 4096;
constexpr size_t chunk_bytes = elements * sizeof(element_t);
constexpr size_t compress_scratch_bytes = cn::compress_scratch_size<algorithm, element_t>(chunk_bytes);
constexpr size_t decompress_scratch_bytes = cn::decompress_scratch_size<algorithm, element_t>(chunk_bytes);
// Exact worst-case size per chunk. Padding keeps each slot aligned.
constexpr size_t capacity = cn::max_comp_chunk_size<algorithm>(chunk_bytes);
constexpr size_t slot_bytes = (capacity + cn::buffer_alignment() - 1) / cn::buffer_alignment() * cn::buffer_alignment();
static_assert(chunk_bytes % cn::buffer_alignment() == 0, "each chunk's input must stay aligned");
static_assert(compress_scratch_bytes % cn::scratch_alignment() == 0 &&
                  decompress_scratch_bytes % cn::scratch_alignment() == 0,
              "each chunk's scratch slices must stay aligned");

__tile_global__ void lz4_gmem_compress_kernel(element_t* input,
                                              std::uint8_t* payload,
                                              unsigned long long* sizes,
                                              std::uint8_t* scratch_base) {
    const unsigned int chunk = ct::bid().x;
    const auto size = cn::compress<algorithm, elements>(input + chunk * elements,
                                                        payload + chunk * slot_bytes,
                                                        cn::scratch { scratch_base + chunk * compress_scratch_bytes });
    ct::store(sizes + chunk, size);
}

__tile_global__ void lz4_gmem_decompress_kernel(std::uint8_t* payload,
                                                unsigned long long* sizes,
                                                element_t* restored,
                                                std::uint8_t* scratch_base) {
    const unsigned int chunk = ct::bid().x;
    cn::decompress<element_t, elements, algorithm>(payload + chunk * slot_bytes,
                                                   ct::load(sizes + chunk),
                                                   restored + chunk * elements,
                                                   cn::scratch { scratch_base + chunk * decompress_scratch_bytes });
}

std::vector<std::uint8_t> make_chunks() {
    // Low-entropy, chunk-distinct values, so every chunk compresses.
    std::vector<std::uint8_t> data(chunks * elements);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<std::uint8_t>(static_cast<unsigned>((i % elements) / 16 + 3 * (i / elements)) % 23);
    }
    return data;
}

int main() {
    const std::vector<element_t> data = make_chunks();
    device_buffer<element_t> d_data(data.size());
    device_buffer<std::uint8_t> d_payload(chunks * slot_bytes);
    device_buffer<unsigned long long> d_sizes(chunks);
    device_buffer<element_t> d_restored(data.size());
    device_buffer<std::uint8_t> d_compress_scratch(chunks * compress_scratch_bytes);
    device_buffer<std::uint8_t> d_decompress_scratch(chunks * decompress_scratch_bytes);
    d_data.copy_from_host(data.data(), data.size());

    lz4_gmem_compress_kernel<<<chunks, 1>>>(d_data.data(), d_payload.data(), d_sizes.data(), d_compress_scratch.data());
    CUDA_CHECK(cudaGetLastError());
    lz4_gmem_decompress_kernel<<<chunks, 1>>>(
        d_payload.data(), d_sizes.data(), d_restored.data(), d_decompress_scratch.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<element_t> restored(data.size());
    std::vector<unsigned long long> sizes(chunks, 0);
    d_restored.copy_to_host(restored.data(), restored.size());
    d_sizes.copy_to_host(sizes.data(), sizes.size());
    const unsigned long long compressed = std::accumulate(sizes.begin(), sizes.end(), 0ull);
    const bool exact = std::memcmp(restored.data(), data.data(), data.size() * sizeof(element_t)) == 0;
    std::printf("lz4 gmem: %zu x %zu B -> %llu B, round trip exact: %s\n",
                chunks,
                chunk_bytes,
                compressed,
                exact ? "yes" : "no");
    return exact && compressed < chunks * chunk_bytes ? EXIT_SUCCESS : EXIT_FAILURE;
}
