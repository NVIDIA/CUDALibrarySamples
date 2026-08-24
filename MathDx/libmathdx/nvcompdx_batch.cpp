/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <libnvcompdx.h>
#include <nvJitLink.h>
#include <nvrtc.h>

#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "common_examples.hpp"
#include "macros.hpp"

using namespace examples;

struct nvcompdx_artifacts {
    std::vector<char> lto;
    std::vector<char> fatbin;
    long long shmem_size = 0;
    long long tmp_size = 0;
    long long shmem_alignment = 0;
    long long max_comp_chunk_size = 0; // 0 for decompress
};

// Build an nvCOMPDx descriptor, finalize it, and return its LTOIR + fatbin + traits.
nvcompdx_artifacts finalize(nvcompdxDirection direction, long long max_uncomp_chunk_size, const char* symbol) {
    nvcompdxDescriptor h { 0 };
    LIBMATHDX_CHECK(nvcompdxCreateDescriptor(&h));
    LIBMATHDX_CHECK(nvcompdxSetOperatorInt64(h, NVCOMPDX_OPERATOR_ALGORITHM, NVCOMPDX_ALGORITHM_LZ4));
    LIBMATHDX_CHECK(nvcompdxSetOperatorInt64(h, NVCOMPDX_OPERATOR_DIRECTION, direction));
    LIBMATHDX_CHECK(nvcompdxSetOperatorInt64(h, NVCOMPDX_OPERATOR_DATATYPE, COMMONDX_R_8UI));
    LIBMATHDX_CHECK(nvcompdxSetOperatorInt64(h, NVCOMPDX_OPERATOR_EXECUTION, COMMONDX_EXECUTION_WARP));
    LIBMATHDX_CHECK(nvcompdxSetOperatorInt64(h, NVCOMPDX_OPERATOR_SM, get_dx_sm().operator_sm()));
    if (direction == NVCOMPDX_DIRECTION_COMPRESS) {
        LIBMATHDX_CHECK(nvcompdxSetOperatorInt64(h, NVCOMPDX_OPERATOR_MAX_UNCOMP_CHUNK_SIZE, max_uncomp_chunk_size));
    }
    LIBMATHDX_CHECK(nvcompdxSetOptionStr(h, COMMONDX_OPTION_SYMBOL_NAME, symbol));

    nvcompdx_artifacts out {};
    LIBMATHDX_CHECK(nvcompdxGetTraitInt64(h, NVCOMPDX_TRAIT_SHMEM_SIZE_GROUP, &out.shmem_size));
    LIBMATHDX_CHECK(nvcompdxGetTraitInt64(h, NVCOMPDX_TRAIT_TMP_SIZE_GROUP, &out.tmp_size));
    LIBMATHDX_CHECK(nvcompdxGetTraitInt64(h, NVCOMPDX_TRAIT_SHMEM_ALIGNMENT, &out.shmem_alignment));
    if (direction == NVCOMPDX_DIRECTION_COMPRESS) {
        LIBMATHDX_CHECK(nvcompdxGetTraitInt64(h, NVCOMPDX_TRAIT_MAX_COMP_CHUNK_SIZE, &out.max_comp_chunk_size));
    }

    commondxCode code {};
    LIBMATHDX_CHECK(commondxCreateCode(&code));
    LIBMATHDX_CHECK(commondxSetCodeOptionInt64(code, COMMONDX_OPTION_TARGET_SM, get_target_sm().operator_sm()));
    LIBMATHDX_CHECK(nvcompdxFinalizeCode(code, h));
    size_t lto_size = 0;
    LIBMATHDX_CHECK(commondxGetCodeLTOIRSize(code, &lto_size));
    out.lto.resize(lto_size);
    LIBMATHDX_CHECK(commondxGetCodeLTOIR(code, out.lto.size(), out.lto.data()));
    LIBMATHDX_CHECK(commondxDestroyCode(code));

    size_t fatbin_size = 0;
    LIBMATHDX_CHECK(nvcompdxGetUniversalFATBINSize(h, &fatbin_size));
    out.fatbin.resize(fatbin_size);
    LIBMATHDX_CHECK(nvcompdxGetUniversalFATBIN(h, out.fatbin.size(), out.fatbin.data()));

    LIBMATHDX_CHECK(nvcompdxDestroyDescriptor(h));
    return out;
}

int main() {
    constexpr int num_chunks = 64;
    constexpr long long chunk_size = 32LL * 1024LL;
    constexpr int warps_per_block = 4;
    constexpr int threads_per_block = warps_per_block * 32;

    // The nvCOMPDx universal fatbin contains the pre-compiled algorithm implementations
    // (LZ4, ANS, ...) shipped with libmathdx. It must be linked alongside the per-descriptor
    // LTOIR for the device function to actually do anything at runtime.
    auto comp = finalize(NVCOMPDX_DIRECTION_COMPRESS, chunk_size, "lz4_compress");
    auto decomp = finalize(NVCOMPDX_DIRECTION_DECOMPRESS, 0, "lz4_decompress");

    const long long shmem_align = std::max(comp.shmem_alignment, decomp.shmem_alignment);

    // The nvCOMPDx-emitted device functions have signature:
    //   void(void* in, void* out, size_t* in_size, size_t* out_size, uint8_t* shmem, uint8_t* tmp)
    // For LZ4 decompress, tmp is unused and may be nullptr.
    const char kernel_template[] = R"(
extern "C" __device__ void lz4_compress(void*, void*, size_t*, size_t*, unsigned char*, unsigned char*);
extern "C" __device__ void lz4_decompress(void*, void*, size_t*, size_t*, unsigned char*, unsigned char*);

constexpr int       warps_per_block   = %d;
constexpr size_t    chunk_size        = %lld;
constexpr size_t    max_comp_size     = %lld;
constexpr long long comp_shmem_warp   = %lld;
constexpr long long decomp_shmem_warp = %lld;
constexpr long long comp_tmp_warp     = %lld;
constexpr long long shmem_align       = %lld;

extern "C" __global__ void compress_batch(
    int n, unsigned char* in_flat, size_t* in_size,
    unsigned char* out_flat, size_t* out_sizes, unsigned char* tmp)
{
    extern __shared__ __align__(shmem_align) unsigned char shmem[];
    int warp = threadIdx.x / 32;
    int chunk = blockIdx.x * warps_per_block + warp;
    if (chunk >= n) return;
    lz4_compress(in_flat + chunk * chunk_size,
                 out_flat + chunk * max_comp_size,
                 in_size, out_sizes + chunk,
                 shmem + warp * comp_shmem_warp,
                 tmp + chunk * comp_tmp_warp);
}

extern "C" __global__ void decompress_batch(
    int n, unsigned char* in_flat, size_t* in_sizes,
    unsigned char* out_flat, size_t* out_sizes)
{
    extern __shared__ __align__(shmem_align) unsigned char shmem[];
    int warp = threadIdx.x / 32;
    int chunk = blockIdx.x * warps_per_block + warp;
    if (chunk >= n) return;
    lz4_decompress(in_flat + chunk * max_comp_size,
                   out_flat + chunk * chunk_size,
                   in_sizes + chunk, out_sizes + chunk,
                   shmem + warp * decomp_shmem_warp,
                   nullptr); // LZ4 decompress takes no tmp scratch
}
)";
    std::string kernel = strprintf(kernel_template,
                                   warps_per_block,
                                   chunk_size,
                                   comp.max_comp_chunk_size,
                                   comp.shmem_size,
                                   decomp.shmem_size,
                                   comp.tmp_size,
                                   shmem_align);

    // Both descriptors are LZ4-with-uint8 at the same SM, so the universal nvCOMPDx fatbin
    // they emit is identical; we pass one of them. The per-direction LTOIRs are distinct.
    std::vector<lto_t> ltos = { { NVJITLINK_INPUT_FATBIN, comp.fatbin },
                                { NVJITLINK_INPUT_LTOIR, comp.lto },
                                { NVJITLINK_INPUT_LTOIR, decomp.lto } };
    auto cubin = compile_and_link(kernel, ltos, get_target_sm());

    CUDA_CHECK(cudaSetDevice(0));

    // Compressed-side stride per chunk: NVCOMPDX_TRAIT_MAX_COMP_CHUNK_SIZE is the
    // worst-case compressed size for any chunk_size-byte input under this configuration.
    const size_t comp_stride = static_cast<size_t>(comp.max_comp_chunk_size);

    // input/output: original and round-tripped data (chunk_size stride).
    // compressed: staging area between kernels (comp_stride).
    // in_size: one size_t = chunk_size, shared by every compress warp.
    // comp_sizes/out_sizes: per-chunk lengths written by compress/decompress.
    // comp_tmp: per-warp scratch (NVCOMPDX_TRAIT_TMP_SIZE_GROUP); LZ4 decompress doesn't need it.
    uint8_t *input {}, *compressed {}, *output {}, *comp_tmp {};
    size_t *in_size {}, *comp_sizes {}, *out_sizes {};
    CUDA_CHECK(cudaMallocManaged(&input, num_chunks * chunk_size));
    CUDA_CHECK(cudaMallocManaged(&compressed, num_chunks * comp_stride));
    CUDA_CHECK(cudaMallocManaged(&output, num_chunks * chunk_size));
    CUDA_CHECK(cudaMallocManaged(&in_size, sizeof(size_t)));
    CUDA_CHECK(cudaMallocManaged(&comp_sizes, num_chunks * sizeof(size_t)));
    CUDA_CHECK(cudaMallocManaged(&out_sizes, num_chunks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&comp_tmp, comp.tmp_size * num_chunks));
    *in_size = static_cast<size_t>(chunk_size);

    // Fill with sequential bytes (0,1,...,255,0,1,...); the cast wraps mod 256.
    for (size_t i = 0; i < num_chunks * chunk_size; ++i) {
        input[i] = static_cast<uint8_t>(i);
    }

    CUmodule mod {};
    CUfunction comp_fn {}, decomp_fn {};
    CU_CHECK(cuModuleLoadDataEx(&mod, cubin.data(), 0, 0, 0));
    CU_CHECK(cuModuleGetFunction(&comp_fn, mod, "compress_batch"));
    CU_CHECK(cuModuleGetFunction(&decomp_fn, mod, "decompress_batch"));

    const int blocks = (num_chunks + warps_per_block - 1) / warps_per_block;
    int n = num_chunks;
    void* comp_args[] = { &n, &input, &in_size, &compressed, &comp_sizes, &comp_tmp };
    void* decomp_args[] = { &n, &compressed, &comp_sizes, &output, &out_sizes };

    CU_CHECK(cuLaunchKernel(comp_fn,
                            blocks,
                            1,
                            1,
                            threads_per_block,
                            1,
                            1,
                            static_cast<unsigned>(comp.shmem_size * warps_per_block),
                            nullptr,
                            comp_args,
                            nullptr));
    CU_CHECK(cuLaunchKernel(decomp_fn,
                            blocks,
                            1,
                            1,
                            threads_per_block,
                            1,
                            1,
                            static_cast<unsigned>(decomp.shmem_size * warps_per_block),
                            nullptr,
                            decomp_args,
                            nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int c = 0; c < num_chunks; ++c) {
        ASSERT(out_sizes[c] == static_cast<size_t>(chunk_size));
        ASSERT(std::memcmp(input + c * chunk_size, output + c * chunk_size, chunk_size) == 0);
    }

    const size_t total_in = static_cast<size_t>(num_chunks) * chunk_size;
    const size_t total_comp = std::accumulate(comp_sizes, comp_sizes + num_chunks, size_t { 0 });
    printf("Compressed %d chunks (%zu B -> %zu B, %.2fx); decompressed output verified\n",
           num_chunks,
           total_in,
           total_comp,
           static_cast<double>(total_in) / static_cast<double>(total_comp));

    CU_CHECK(cuModuleUnload(mod));
    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(compressed));
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(in_size));
    CUDA_CHECK(cudaFree(comp_sizes));
    CUDA_CHECK(cudaFree(out_sizes));
    CUDA_CHECK(cudaFree(comp_tmp));
    return 0;
}
