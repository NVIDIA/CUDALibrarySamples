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

// nvTIFF-Decode-Encode
// --------------------
// Decodes one image, writes a preview, then re-encodes it.

#include "nvtiff_samples_common.h"

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3) {
        std::printf("Usage: %s <input.tif> [output.tif]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string input_file = argv[1];
    const std::string output_file = (argc == 3) ? argv[2] : "nvtiff_reencoded.tif";

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Open the TIFF. Only the file header is validated here; each IFD is
    // parsed lazily on first access.
    nvtiffStream_t tiff_stream;
    CHECK_NVTIFF(nvtiffStreamOpenFromFile(input_file.c_str(), &tiff_stream));

    // Metadata of the first image (an ifd_offset of 0 selects the first IFD).
    nvtiffImageInfo_t info;
    CHECK_NVTIFF(nvtiffStreamGetImageInfo(tiff_stream, 0, &info));
    std::printf("%s: %u x %u, %u channel(s), %u bits per pixel, %s, %s\n", input_file.c_str(), info.image_width,
        info.image_height, info.samples_per_pixel, info.bits_per_pixel, photometric_string(info.photometric_int),
        compression_string(info.compression));

    const DecodeTarget target = pick_decode_target(info);
    nvtiffImageInfo_t encode_info = {};
    if (!make_lzw_encode_info(info, target, encode_info)) {
        CHECK_NVTIFF(nvtiffStreamClose(tiff_stream));
        CHECK_CUDA(cudaStreamDestroy(stream));
        return EXIT_FAILURE;
    }

    nvtiffDecoder_t decoder;
    CHECK_NVTIFF(nvtiffDecoderCreateSimple(&decoder, stream));
    nvtiffDecodeParams_t decode_params;
    CHECK_NVTIFF(nvtiffDecodeParamsCreate(&decode_params));
    CHECK_NVTIFF(nvtiffDecodeParamsSetOutputFormat(decode_params, target.format));

    // One interleaved output plane with a dense pitch.
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

    // Re-encode the decoded raster as an LZW-compressed TIFF.
    nvtiffEncoder_t encoder;
    CHECK_NVTIFF(nvtiffEncoderCreate(&encoder, nullptr, nullptr, stream));
    nvtiffEncodeParams_t encode_params;
    CHECK_NVTIFF(nvtiffEncodeParamsCreate(&encode_params));
    CHECK_NVTIFF(nvtiffEncodeParamsSetImageInfo(encode_params, &encode_info));
    uint8_t* encode_inputs[1] = {static_cast<uint8_t*>(pixels_d)};
    CHECK_NVTIFF(nvtiffEncodeParamsSetInputs(encode_params, encode_inputs, 1));

    CHECK_NVTIFF(nvtiffEncode(encoder, &encode_params, 1, stream));
    CHECK_NVTIFF(nvtiffEncodeFinalize(encoder, &encode_params, 1, stream));

    size_t metadata_bytes = 0;
    size_t bitstream_bytes = 0;
    CHECK_NVTIFF(nvtiffGetBitstreamSize(encoder, &encode_params, 1, &metadata_bytes, &bitstream_bytes));
    const size_t raw_bytes = pitch * info.image_height;
    std::printf("re-encoded with LZW: %zu raw -> %zu compressed bytes (%.2fx)\n", raw_bytes, bitstream_bytes,
        (double)raw_bytes / (double)bitstream_bytes);

    // nvtiffWriteTiffFile synchronizes the stream internally.
    CHECK_NVTIFF(nvtiffWriteTiffFile(encoder, &encode_params, 1, output_file.c_str(), stream));
    std::printf("wrote %s\n", output_file.c_str());

    CHECK_NVTIFF(nvtiffEncodeParamsDestroy(encode_params, stream));
    CHECK_NVTIFF(nvtiffEncoderDestroy(encoder, stream));
    CHECK_NVTIFF(nvtiffDecodeParamsDestroy(decode_params));
    CHECK_NVTIFF(nvtiffDecoderDestroy(decoder, stream));
    CHECK_NVTIFF(nvtiffStreamClose(tiff_stream));
    CHECK_CUDA(cudaFree(pixels_d));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
