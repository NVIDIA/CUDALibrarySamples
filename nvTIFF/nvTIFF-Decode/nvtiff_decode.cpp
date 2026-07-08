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

// nvTIFF-Decode
// -------------
// Decodes the first image and writes a PNM preview.

#include "nvtiff_samples_common.h"

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::printf("Usage: %s <input.tif>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string input_file = argv[1];

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    nvtiffStream_t tiff_stream;
    CHECK_NVTIFF(nvtiffStreamOpenFromFile(input_file.c_str(), &tiff_stream));

    // An ifd_offset of 0 selects the first IFD.
    nvtiffImageInfo_t info;
    CHECK_NVTIFF(nvtiffStreamGetImageInfo(tiff_stream, 0, &info));
    std::printf("%s: %u x %u, %u channel(s), %u bits per pixel, %s, %s\n", input_file.c_str(), info.image_width,
        info.image_height, info.samples_per_pixel, info.bits_per_pixel, photometric_string(info.photometric_int),
        compression_string(info.compression));

    nvtiffDecoder_t decoder;
    CHECK_NVTIFF(nvtiffDecoderCreateSimple(&decoder, stream));
    nvtiffDecodeParams_t decode_params;
    CHECK_NVTIFF(nvtiffDecodeParamsCreate(&decode_params));
    const DecodeTarget target = pick_decode_target(info);
    CHECK_NVTIFF(nvtiffDecodeParamsSetOutputFormat(decode_params, target.format));

    size_t pitch = decode_row_bytes(info, target, info.image_width);
    void* pixels_d = nullptr;
    CHECK_CUDA(cudaMalloc(&pixels_d, pitch * info.image_height));
    nvtiffImage_t image = {};
    image.plane_data = &pixels_d;
    image.plane_pitch_bytes = &pitch;
    image.num_planes = 1;

    CHECK_NVTIFF(nvtiffDecode(tiff_stream, decoder, decode_params, &image, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::printf("decoded first image\n");

    const std::string preview_file =
        base_name_no_ext(input_file) + (target.channels == 1 ? "_decoded.pgm" : "_decoded.ppm");
    write_decoded_preview(preview_file, info, target, pixels_d, pitch, info.image_width, info.image_height);

    CHECK_NVTIFF(nvtiffDecodeParamsDestroy(decode_params));
    CHECK_NVTIFF(nvtiffDecoderDestroy(decoder, stream));
    CHECK_NVTIFF(nvtiffStreamClose(tiff_stream));
    CHECK_CUDA(cudaFree(pixels_d));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
