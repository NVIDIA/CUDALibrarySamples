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

// nvTIFF-Batched-Region-Decode
// ----------------------------
// Decodes a reproducible random set of patches in one batch.

#include "nvtiff_samples_common.h"

#include <algorithm>
#include <chrono>
#include <random>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::printf("Usage: %s <input.tif> [--patch-width N] [--patch-height N] [--num-patches N]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string input_file = argv[1];
    uint32_t patch_w = 256;
    uint32_t patch_h = 256;
    uint32_t requested_patches = 8;
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", argv[i]);
            return EXIT_FAILURE;
        }
        uint32_t value = 0;
        if (!parse_positive_uint32(argv[i + 1], &value)) {
            std::fprintf(stderr, "invalid positive integer for %s: %s\n", argv[i], argv[i + 1]);
            return EXIT_FAILURE;
        }
        if (std::string(argv[i]) == "--patch-width") {
            patch_w = value;
        } else if (std::string(argv[i]) == "--patch-height") {
            patch_h = value;
        } else if (std::string(argv[i]) == "--num-patches") {
            requested_patches = value;
        } else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    nvtiffStream_t tiff_stream;
    CHECK_NVTIFF(nvtiffStreamOpenFromFile(input_file.c_str(), &tiff_stream));
    nvtiffImageInfo_t info;
    CHECK_NVTIFF(nvtiffStreamGetImageInfo(tiff_stream, 0, &info));

    const uint32_t max_patches = 256;
    uint32_t num_patches = std::min(requested_patches, max_patches);
    if (requested_patches > max_patches) {
        std::printf("requested %u patches; decoding only the first %u\n", requested_patches, max_patches);
    }
    const uint32_t max_patch_dim = 4096;
    const uint32_t random_w = std::min({patch_w, info.image_width, max_patch_dim});
    const uint32_t random_h = std::min({patch_h, info.image_height, max_patch_dim});
    if (random_w != patch_w || random_h != patch_h) {
        std::printf("requested patch size %u x %u clipped to %u x %u\n", patch_w, patch_h, random_w, random_h);
    }
    std::printf("%s: %u x %u, decoding %u random patch(es) of %u x %u in one batch\n", input_file.c_str(),
        info.image_width, info.image_height, num_patches, random_w, random_h);

    // Region descriptors. The fixed seed keeps sample output reproducible.
    std::vector<nvtiffDecodeRegion_t> regions(num_patches);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> random_x(0, info.image_width - random_w);
    std::uniform_int_distribution<uint32_t> random_y(0, info.image_height - random_h);
    for (uint32_t i = 0; i < num_patches; i++) {
        nvtiffDecodeRegion_t& region = regions[i];
        region.ifd_offset = 0; // all patches from the first image
        region.offset_x = (int32_t)random_x(rng);
        region.offset_y = (int32_t)random_y(rng);
        region.width = random_w;
        region.height = random_h;
    }

    const size_t patch_pitch = (size_t)random_w * 3;
    const size_t patch_bytes = patch_pitch * random_h;
    const size_t max_total_patch_bytes = (size_t)1 << 30;
    if (patch_bytes != 0 && num_patches > max_total_patch_bytes / patch_bytes) {
        std::fprintf(stderr, "requested patch outputs need more than %zu bytes; reduce patch size or count\n",
            max_total_patch_bytes);
        CHECK_NVTIFF(nvtiffStreamClose(tiff_stream));
        CHECK_CUDA(cudaStreamDestroy(stream));
        return EXIT_FAILURE;
    }

    nvtiffDecoder_t decoder;
    CHECK_NVTIFF(nvtiffDecoderCreateSimple(&decoder, stream));
    nvtiffDecodeParams_t decode_params;
    CHECK_NVTIFF(nvtiffDecodeParamsCreate(&decode_params));
    CHECK_NVTIFF(nvtiffDecodeParamsSetOutputFormat(decode_params, NVTIFF_OUTPUT_RGB_I_UINT8));
    CHECK_NVTIFF(nvtiffDecodeParamsSetRegions(decode_params, regions.data(), num_patches));

    // One contiguous RGB8 output allocation, with each region descriptor
    // pointing at its own patch-sized slice.
    void* patches_d = nullptr;
    CHECK_CUDA(cudaMalloc(&patches_d, patch_bytes * num_patches));
    std::vector<void*> patch_ptrs(num_patches, nullptr);
    std::vector<size_t> patch_pitches(num_patches, patch_pitch);
    std::vector<nvtiffImage_t> outputs(num_patches);
    for (uint32_t i = 0; i < num_patches; i++) {
        patch_ptrs[i] = static_cast<uint8_t*>(patches_d) + (size_t)i * patch_bytes;
        outputs[i].plane_data = &patch_ptrs[i];
        outputs[i].plane_pitch_bytes = &patch_pitches[i];
        outputs[i].num_planes = 1;
    }

    // Dry-run first: with output descriptors provided, this performs the
    // same pre-launch validation nvtiffDecode would, without any GPU work.
    const nvtiffStatus_t batch_status =
        nvtiffDecodeCheckSupported(tiff_stream, decoder, decode_params, outputs.data());

    if (batch_status == NVTIFF_STATUS_SUCCESS) {
        const auto start = std::chrono::high_resolution_clock::now();
        CHECK_NVTIFF(nvtiffDecode(tiff_stream, decoder, decode_params, outputs.data(), stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        const auto end = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::printf("decoded %u patches in one batch: %.3f ms (%.0f patches/s)\n", num_patches, ms,
            1000.0 * num_patches / ms);
    } else if (batch_status == NVTIFF_STATUS_BATCH_INCOMPATIBLE) {
        // Every region is individually decodable, but they cannot execute
        // as one batch (nvTIFF never silently serializes a batch).
        // Decode region by region instead. A decoder allows only one decode
        // in flight, so each call is followed by a synchronize.
        std::printf("batch rejected (NVTIFF_STATUS_BATCH_INCOMPATIBLE); decoding per region\n");
        for (uint32_t i = 0; i < num_patches; i++) {
            CHECK_NVTIFF(nvtiffDecodeParamsSetRegions(decode_params, &regions[i], 1));
            CHECK_NVTIFF(nvtiffDecode(tiff_stream, decoder, decode_params, &outputs[i], stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }
        std::printf("decoded %u patches individually\n", num_patches);
    } else {
        std::fprintf(stderr, "decode request rejected: %s\n", nvtiff_status_string(batch_status));
        CHECK_CUDA(cudaFree(patches_d));
        CHECK_NVTIFF(nvtiffDecodeParamsDestroy(decode_params));
        CHECK_NVTIFF(nvtiffDecoderDestroy(decoder, stream));
        CHECK_NVTIFF(nvtiffStreamClose(tiff_stream));
        CHECK_CUDA(cudaStreamDestroy(stream));
        return EXIT_FAILURE;
    }

    // Write the selected patches out for inspection.
    for (uint32_t i = 0; i < num_patches; i++) {
        std::vector<uint8_t> host(patch_bytes);
        CHECK_CUDA(cudaMemcpy(host.data(), patch_ptrs[i], host.size(), cudaMemcpyDeviceToHost));
        const std::string preview_file = base_name_no_ext(input_file) + "_patch" + std::to_string(i) + ".ppm";
        if (write_pnm(preview_file, host.data(), patch_pitches[i], random_w, random_h, 3, 1,
                NVTIFF_PHOTOMETRIC_RGB, NVTIFF_SAMPLEFORMAT_UINT)) {
            std::printf("  wrote %u x %u preview %s\n", random_w, random_h, preview_file.c_str());
        }
    }

    CHECK_CUDA(cudaFree(patches_d));
    CHECK_NVTIFF(nvtiffDecodeParamsDestroy(decode_params));
    CHECK_NVTIFF(nvtiffDecoderDestroy(decoder, stream));
    CHECK_NVTIFF(nvtiffStreamClose(tiff_stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
