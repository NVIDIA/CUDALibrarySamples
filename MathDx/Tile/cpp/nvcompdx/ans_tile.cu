// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// ANS compression of tiles with the CUDA Tile C++ nvCOMPDx API. Each block
// loads one chunk into a tile, compresses it, decompresses the compressed
// tile, and stores both results.

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

using element_t = __half;
constexpr cn::algorithm algorithm = cn::ans;
constexpr size_t chunks = 4;
constexpr size_t elements = 2048;
constexpr size_t chunk_bytes = elements * sizeof(element_t);
constexpr size_t compress_scratch_bytes = cn::compress_scratch_size<algorithm, element_t>(chunk_bytes);
constexpr size_t decompress_scratch_bytes = cn::decompress_scratch_size<algorithm, element_t>(chunk_bytes);
constexpr size_t payload_bytes = cn::comp_tile_size<algorithm>(chunk_bytes);
static_assert(compress_scratch_bytes % cn::scratch_alignment() == 0 &&
                  decompress_scratch_bytes % cn::scratch_alignment() == 0,
              "each chunk's scratch slices must stay aligned");

__tile_global__ void ans_tile_kernel(element_t* input_data,
                                     std::uint8_t* payload_data,
                                     unsigned long long* sizes_data,
                                     element_t* restored_data,
                                     std::uint8_t* compress_scratch,
                                     std::uint8_t* decompress_scratch) {
    const unsigned int chunk = ct::bid().x;
    auto input =
        ct::partition_view { ct::tensor_span { input_data, ct::shape<chunks * elements> {} }, ct::shape<elements> {} };
    auto payload = ct::partition_view { ct::tensor_span { payload_data, ct::shape<chunks * payload_bytes> {} },
                                        ct::shape<payload_bytes> {} };
    auto restored = ct::partition_view { ct::tensor_span { restored_data, ct::shape<chunks * elements> {} },
                                         ct::shape<elements> {} };

    auto [compressed, size] =
        cn::compress<algorithm>(input.load(chunk), cn::scratch { compress_scratch + chunk * compress_scratch_bytes });
    payload.store(compressed, chunk);
    ct::store(sizes_data + chunk, size);
    restored.store(cn::decompress<element_t, elements, algorithm>(
                       compressed, size, cn::scratch { decompress_scratch + chunk * decompress_scratch_bytes }),
                   chunk);
}

std::vector<__half> make_chunks() {
    // Low-entropy, chunk-distinct values, so every chunk compresses.
    std::vector<__half> data(chunks * elements);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = __uint2half_rn(static_cast<unsigned>((i % elements) / 16 + 3 * (i / elements)) % 23);
    }
    return data;
}

int main() {
    const std::vector<element_t> data = make_chunks();
    device_buffer<element_t> d_data(data.size());
    device_buffer<std::uint8_t> d_payload(chunks * payload_bytes);
    device_buffer<unsigned long long> d_sizes(chunks);
    device_buffer<element_t> d_restored(data.size());
    device_buffer<std::uint8_t> d_compress_scratch(chunks * compress_scratch_bytes);
    device_buffer<std::uint8_t> d_decompress_scratch(chunks * decompress_scratch_bytes);
    d_data.copy_from_host(data.data(), data.size());

    ans_tile_kernel<<<chunks, 1>>>(d_data.data(),
                                   d_payload.data(),
                                   d_sizes.data(),
                                   d_restored.data(),
                                   d_compress_scratch.data(),
                                   d_decompress_scratch.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<element_t> restored(data.size());
    std::vector<unsigned long long> sizes(chunks, 0);
    d_restored.copy_to_host(restored.data(), restored.size());
    d_sizes.copy_to_host(sizes.data(), sizes.size());
    const unsigned long long compressed = std::accumulate(sizes.begin(), sizes.end(), 0ull);
    const bool exact = std::memcmp(restored.data(), data.data(), data.size() * sizeof(element_t)) == 0;
    std::printf("ans tile: %zu x %zu B -> %llu B, round trip exact: %s\n",
                chunks,
                chunk_bytes,
                compressed,
                exact ? "yes" : "no");
    return exact && compressed < chunks * chunk_bytes ? EXIT_SUCCESS : EXIT_FAILURE;
}
