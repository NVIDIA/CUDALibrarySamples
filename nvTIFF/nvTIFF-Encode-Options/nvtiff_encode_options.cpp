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

// nvTIFF-Encode-Options
// ---------------------
// Re-encodes one image with several encoder options.

#include "nvtiff_samples_common.h"

#include <functional>

static nvtiffImageInfo_t rgb8_info(uint32_t width, uint32_t height, nvtiffCompression_t compression)
{
    nvtiffImageInfo_t info = {};
    info.image_width = width;
    info.image_height = height;
    info.samples_per_pixel = 3;
    info.bits_per_pixel = 24;
    for (uint32_t c = 0; c < 3; c++) {
        info.bits_per_sample[c] = 8;
        info.sample_format[c] = NVTIFF_SAMPLEFORMAT_UINT;
    }
    info.photometric_int = NVTIFF_PHOTOMETRIC_RGB;
    info.planar_config = NVTIFF_PLANARCONFIG_CONTIG;
    info.compression = compression;
    return info;
}

// Encodes `num_images` pages of the same dense RGB8 raster, applying
// `customize` to the encode params before encoding, and writes the result
// to `out_path` (through a host buffer when `write_via_buffer` is set).
// Returns the total encoded file size in bytes.
static size_t encode_variant(const std::string& out_path, uint8_t* rgb_d, uint32_t width, uint32_t height,
    uint32_t num_images, nvtiffCompression_t compression,
    const std::function<void(nvtiffEncodeParams_t)>& customize, bool write_via_buffer, cudaStream_t stream)
{
    nvtiffEncoder_t encoder;
    CHECK_NVTIFF(nvtiffEncoderCreate(&encoder, nullptr, nullptr, stream));
    nvtiffEncodeParams_t params;
    CHECK_NVTIFF(nvtiffEncodeParamsCreate(&params));

    const nvtiffImageInfo_t info = rgb8_info(width, height, compression);
    CHECK_NVTIFF(nvtiffEncodeParamsSetImageInfo(params, &info));
    std::vector<uint8_t*> inputs(num_images, rgb_d);
    CHECK_NVTIFF(nvtiffEncodeParamsSetInputs(params, inputs.data(), num_images));
    if (customize) {
        customize(params);
    }

    CHECK_NVTIFF(nvtiffEncode(encoder, &params, 1, stream));
    CHECK_NVTIFF(nvtiffEncodeFinalize(encoder, &params, 1, stream));

    size_t metadata_bytes = 0;
    size_t bitstream_bytes = 0;
    CHECK_NVTIFF(nvtiffGetBitstreamSize(encoder, &params, 1, &metadata_bytes, &bitstream_bytes));
    const size_t file_bytes = metadata_bytes + bitstream_bytes;

    if (write_via_buffer) {
        // Encode-to-memory: retrieve the complete TIFF (metadata included)
        // into a host buffer. Useful for sending encoded images over the
        // network or embedding them without touching the filesystem.
        std::vector<uint8_t> tiff_data(file_bytes);
        CHECK_NVTIFF(nvtiffWriteTiffBuffer(encoder, &params, 1, tiff_data.data(), tiff_data.size(), stream));
        std::ofstream out(out_path, std::ios::binary);
        if (!out.write(reinterpret_cast<const char*>(tiff_data.data()), (std::streamsize)tiff_data.size())) {
            std::fprintf(stderr, "cannot write output file %s\n", out_path.c_str());
            std::exit(EXIT_FAILURE);
        }
    } else {
        CHECK_NVTIFF(nvtiffWriteTiffFile(encoder, &params, 1, out_path.c_str(), stream));
    }

    CHECK_NVTIFF(nvtiffEncodeParamsDestroy(params, stream));
    CHECK_NVTIFF(nvtiffEncoderDestroy(encoder, stream));
    return file_bytes;
}

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3) {
        std::printf("Usage: %s <input.tif> [out_dir]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string input_file = argv[1];
    const std::string out_dir = (argc == 3) ? argv[2] : ".";

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Decode the first image of the input to 8-bit interleaved RGB, the one
    // layout every encode variant below accepts (JPEG requires it).
    nvtiffStream_t tiff_stream;
    CHECK_NVTIFF(nvtiffStreamOpenFromFile(input_file.c_str(), &tiff_stream));
    nvtiffImageInfo_t info;
    CHECK_NVTIFF(nvtiffStreamGetImageInfo(tiff_stream, 0, &info));

    nvtiffDecoder_t decoder;
    CHECK_NVTIFF(nvtiffDecoderCreateSimple(&decoder, stream));
    nvtiffDecodeParams_t decode_params;
    CHECK_NVTIFF(nvtiffDecodeParamsCreate(&decode_params));
    CHECK_NVTIFF(nvtiffDecodeParamsSetOutputFormat(decode_params, NVTIFF_OUTPUT_RGB_I_UINT8));

    const uint32_t width = info.image_width;
    const uint32_t height = info.image_height;
    size_t pitch = (size_t)width * 3;
    void* rgb_d = nullptr;
    CHECK_CUDA(cudaMalloc(&rgb_d, pitch * height));
    nvtiffImage_t image = {};
    image.plane_data = &rgb_d;
    image.plane_pitch_bytes = &pitch;
    image.num_planes = 1;
    CHECK_NVTIFF(nvtiffDecode(tiff_stream, decoder, decode_params, &image, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::printf("decoded %s: %u x %u -> RGB8\n", input_file.c_str(), width, height);

    uint8_t* rgb8_d = static_cast<uint8_t*>(rgb_d);
    struct Variant
    {
        const char* label;
        const char* file;
        uint32_t pages;
        size_t bytes;
    };
    std::vector<Variant> results;

    // 1. LZW in explicit 128-row strips, encoded twice -> a two-page file.
    //    For striped images strile_width must equal the image width.
    results.push_back({"LZW, 128-row strips, 2 pages", "striped_lzw_multipage.tif", 2,
        encode_variant(out_dir + "/striped_lzw_multipage.tif", rgb8_d, width, height, 2, NVTIFF_COMPRESSION_LZW,
            [&](nvtiffEncodeParams_t params) {
                nvtiffImageGeometry_t geometry = {};
                geometry.type = NVTIFF_IMAGE_STRIPED;
                geometry.strile_width = width;
                geometry.strile_height = 128;
                CHECK_NVTIFF(nvtiffEncodeParamsSetImageGeometry(params, &geometry));
            },
            false, stream)});

    // 2. LZW in 256x256 tiles. Tile dimensions must be multiples of 16.
    results.push_back({"LZW, 256x256 tiles", "tiled_lzw.tif", 1,
        encode_variant(out_dir + "/tiled_lzw.tif", rgb8_d, width, height, 1, NVTIFF_COMPRESSION_LZW,
            [&](nvtiffEncodeParams_t params) {
                nvtiffImageGeometry_t geometry = {};
                geometry.type = NVTIFF_IMAGE_TILED;
                geometry.strile_width = 256;
                geometry.strile_height = 256;
                CHECK_NVTIFF(nvtiffEncodeParamsSetImageGeometry(params, &geometry));
            },
            false, stream)});

    // 3. JPEG-in-TIFF, quality 85, no chroma subsampling, optimized Huffman
    //    tables. Requires the nvJPEG library at runtime.
    results.push_back({"JPEG q85 4:4:4 optimized Huffman", "jpeg_q85.tif", 1,
        encode_variant(out_dir + "/jpeg_q85.tif", rgb8_d, width, height, 1, NVTIFF_COMPRESSION_JPEG,
            [&](nvtiffEncodeParams_t params) {
                nvtiffJpegEncodeOptions_t jpeg_options = {};
                jpeg_options.quality = 85;
                jpeg_options.optimized_huffman = 1;
                jpeg_options.chroma_subsampling = NVTIFF_JPEG_CHROMA_SUBSAMPLING_444;
                CHECK_NVTIFF(nvtiffEncodeParamsSetJpegOptions(params, &jpeg_options));
            },
            false, stream)});

    // 4. Uncompressed BigTIFF (64-bit offsets; required for files > 4 GB,
    //    valid at any size).
    results.push_back({"uncompressed BigTIFF", "bigtiff_uncompressed.tif", 1,
        encode_variant(out_dir + "/bigtiff_uncompressed.tif", rgb8_d, width, height, 1, NVTIFF_COMPRESSION_NONE,
            [&](nvtiffEncodeParams_t params) {
                CHECK_NVTIFF(nvtiffEncodeParamsSetTiffVariant(params, NVTIFF_BIG_TIFF));
            },
            false, stream)});

    // 5. LZW with custom numeric tags written via nvtiffEncodeParamsSetTag and
    //    a host memory buffer instead of directly to a file.
    //
    // NOTE: nvTIFF 0.8.0's nvtiffEncodeParamsSetTag accepts only
    // ModelPixelScale (33550), ModelTiepoint (33922),
    // ModelTransformation (34264), and RPCCoefficientTag (50844).
    // Other TIFF tags, including Software (305) and
    // ImageDescription (270), return NVTIFF_STATUS_INVALID_PARAMETER.
    results.push_back({"LZW, custom tags, via buffer", "custom_tags_from_buffer.tif", 1,
        encode_variant(out_dir + "/custom_tags_from_buffer.tif", rgb8_d, width, height, 1, NVTIFF_COMPRESSION_LZW,
            [&](nvtiffEncodeParams_t params) {
                // ModelPixelScale (33550): pixel size in world units (x, y, z).
                static double pixel_scale[3] = {1.0, 1.0, 0.0};
                CHECK_NVTIFF(nvtiffEncodeParamsSetTag(
                    params, NVTIFF_TAG_MODEL_PIXEL_SCALE, NVTIFF_TAG_TYPE_DOUBLE,
                    pixel_scale, 3));
                // ModelTiePoint (33922): raster (i,j,k) -> world (x,y,z).
                static double tie_point[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                CHECK_NVTIFF(nvtiffEncodeParamsSetTag(
                    params, NVTIFF_TAG_MODEL_TIE_POINT, NVTIFF_TAG_TYPE_DOUBLE,
                    tie_point, 6));
            },
            true, stream)});

    std::printf("\n%-36s %-28s %12s %8s\n", "variant", "file", "bytes", "ratio");
    for (const Variant& v : results) {
        const size_t raw_bytes = (size_t)width * height * 3 * v.pages;
        std::printf("%-36s %-28s %12zu %7.2fx\n", v.label, v.file, v.bytes, (double)raw_bytes / (double)v.bytes);
    }

    CHECK_NVTIFF(nvtiffDecodeParamsDestroy(decode_params));
    CHECK_NVTIFF(nvtiffDecoderDestroy(decoder, stream));
    CHECK_NVTIFF(nvtiffStreamClose(tiff_stream));
    CHECK_CUDA(cudaFree(rgb_d));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
