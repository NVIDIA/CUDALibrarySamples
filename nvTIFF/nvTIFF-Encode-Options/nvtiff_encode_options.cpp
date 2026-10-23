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

// Selects striped or tiled output. For striped images strile_width must equal
// the image width; for tiled images width and height must be multiples of 16.
static void set_geometry(nvtiffEncodeParams_t params, nvtiffImageGeometryType_t type, uint32_t strile_width,
    uint32_t strile_height)
{
    nvtiffImageGeometry_t geometry = {};
    geometry.type = type;
    geometry.strile_width = strile_width;
    geometry.strile_height = strile_height;
    CHECK_NVTIFF(nvtiffEncodeParamsSetImageGeometry(params, &geometry));
}

// Encodes `num_images` pages of the same dense RGB8 raster, applying
// `customize` to the encode params before encoding, and writes the result
// to `out_path` (through a host buffer when `write_via_buffer` is set).
// Returns the total encoded file size in bytes. If `compression` needs an
// optional library that is not installed (nvCOMP 6 for Deflate/Zstd, nvJPEG for
// JPEG), the variant is skipped: 0 is returned and `*missing_library` names the
// library.
static size_t encode_variant(const std::string& out_path, uint8_t* rgb_d, uint32_t width, uint32_t height,
    uint32_t num_images, nvtiffCompression_t compression,
    const std::function<void(nvtiffEncodeParams_t)>& customize, bool write_via_buffer, cudaStream_t stream,
    const char** missing_library)
{
    nvtiffEncoder_t encoder;
    CHECK_NVTIFF(nvtiffEncoderCreate(&encoder, nullptr, nullptr, stream));
    nvtiffEncodeParams_t params;
    CHECK_NVTIFF(nvtiffEncodeParamsCreate(&params));

    const nvtiffImageInfo_t info = rgb8_info(width, height, compression);
    const nvtiffStatus_t info_status = nvtiffEncodeParamsSetImageInfo(params, &info);
    if (info_status == NVTIFF_STATUS_NVCOMP_NOT_FOUND || info_status == NVTIFF_STATUS_NVJPEG_NOT_FOUND) {
        *missing_library = (info_status == NVTIFF_STATUS_NVCOMP_NOT_FOUND) ? "nvCOMP 6" : "nvJPEG";
        CHECK_NVTIFF(nvtiffEncodeParamsDestroy(params, stream));
        CHECK_NVTIFF(nvtiffEncoderDestroy(encoder, stream));
        return 0;
    }
    CHECK_NVTIFF(info_status);
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

    // The decode resources are no longer needed: only the RGB buffer is used
    // from here on. Releasing them now lowers peak memory while encoding.
    CHECK_NVTIFF(nvtiffDecodeParamsDestroy(decode_params));
    CHECK_NVTIFF(nvtiffDecoderDestroy(decoder, stream));
    CHECK_NVTIFF(nvtiffStreamClose(tiff_stream));

    uint8_t* rgb8_d = static_cast<uint8_t*>(rgb_d);
    struct Variant
    {
        const char* label;
        const char* file;
        uint32_t pages;
        size_t bytes;                // 0 if the variant was skipped
        const char* missing_library; // set if the variant was skipped
    };
    std::vector<Variant> results;

    // Encodes the decoded image with one set of options and records the result.
    // `customize` sets the encode options being demonstrated.
    auto add_variant = [&](const char* label, const char* file, uint32_t pages, nvtiffCompression_t compression,
                           const std::function<void(nvtiffEncodeParams_t)>& customize, bool via_buffer = false) {
        const char* missing_library = nullptr;
        const size_t bytes = encode_variant(out_dir + "/" + file, rgb8_d, width, height, pages, compression,
            customize, via_buffer, stream, &missing_library);
        results.push_back({label, file, pages, bytes, missing_library});
    };

    // 1. LZW in explicit 128-row strips, encoded twice -> a two-page file.
    add_variant("LZW, 128-row strips, 2 pages", "striped_lzw_multipage.tif", 2, NVTIFF_COMPRESSION_LZW,
        [&](nvtiffEncodeParams_t params) { set_geometry(params, NVTIFF_IMAGE_STRIPED, width, 128); });

    // 2. LZW in 256x256 tiles.
    add_variant("LZW, 256x256 tiles", "tiled_lzw.tif", 1, NVTIFF_COMPRESSION_LZW,
        [&](nvtiffEncodeParams_t params) { set_geometry(params, NVTIFF_IMAGE_TILED, 256, 256); });

    // 3. JPEG-in-TIFF, quality 85, no chroma subsampling, optimized Huffman
    //    tables. Requires the nvJPEG library at runtime; skipped if it is
    //    not found.
    add_variant("JPEG q85 4:4:4 optimized Huffman", "jpeg_q85.tif", 1, NVTIFF_COMPRESSION_JPEG,
        [&](nvtiffEncodeParams_t params) {
            nvtiffJpegEncodeOptions_t jpeg_options = {};
            jpeg_options.quality = 85;
            jpeg_options.optimized_huffman = 1;
            jpeg_options.chroma_subsampling = NVTIFF_JPEG_CHROMA_SUBSAMPLING_444;
            CHECK_NVTIFF(nvtiffEncodeParamsSetJpegOptions(params, &jpeg_options));
        });

    // 4. Uncompressed BigTIFF (64-bit offsets; required for files > 4 GB,
    //    valid at any size).
    add_variant("uncompressed BigTIFF", "bigtiff_uncompressed.tif", 1, NVTIFF_COMPRESSION_NONE,
        [&](nvtiffEncodeParams_t params) { CHECK_NVTIFF(nvtiffEncodeParamsSetTiffVariant(params, NVTIFF_BIG_TIFF)); });

    // 5. LZW with custom ASCII and numeric tags set via nvtiffEncodeParamsSetTag.
    //    nvTIFF 0.9 accepts arbitrary TIFF tags. Tags nvTIFF writes itself
    //    (ImageWidth, Compression, StripOffsets, ...) are rejected with
    //    NVTIFF_STATUS_INVALID_PARAMETER.
    add_variant("LZW, custom tags", "custom_tags.tif", 1, NVTIFF_COMPRESSION_LZW, [&](nvtiffEncodeParams_t params) {
        // ASCII tags. Per the TIFF spec the count includes the null terminator;
        // nvTIFF writes the bytes as provided.
        auto set_ascii_tag = [&](uint16_t tag, const char* text) {
            std::string value(text);
            CHECK_NVTIFF(nvtiffEncodeParamsSetTag(
                params, tag, NVTIFF_TAG_TYPE_ASCII, &value[0], (uint32_t)value.size() + 1));
        };
        set_ascii_tag(270, "Re-encoded by the nvTIFF Encode-Options sample"); // ImageDescription
        set_ascii_tag(271, "NVIDIA");                                         // Make
        set_ascii_tag(272, "nvTIFF sample");                                  // Model
        set_ascii_tag(305, "nvTIFF encode-options sample");                   // Software

        // Numeric tags. A RATIONAL value is a numerator/denominator pair of
        // uint32 (count = number of rationals); SHORT is a plain uint16.
        uint32_t resolution[2] = {300, 1}; // 300/1
        CHECK_NVTIFF(nvtiffEncodeParamsSetTag(params, 282, NVTIFF_TAG_TYPE_RATIONAL, resolution, 1)); // XResolution
        CHECK_NVTIFF(nvtiffEncodeParamsSetTag(params, 283, NVTIFF_TAG_TYPE_RATIONAL, resolution, 1)); // YResolution
        uint16_t resolution_unit = 2;                                                                 // inch
        CHECK_NVTIFF(nvtiffEncodeParamsSetTag(params, 296, NVTIFF_TAG_TYPE_SHORT, &resolution_unit, 1)); // ResolutionUnit
    });

    // 6. LZW encoded to memory: nvtiffWriteTiffBuffer returns the complete TIFF
    //    in a host buffer instead of writing a file.
    add_variant("LZW, via host buffer", "lzw_from_buffer.tif", 1, NVTIFF_COMPRESSION_LZW, nullptr,
        /*via_buffer=*/true);

    // 7. LZW with the predictor explicitly disabled. By default nvTIFF selects
    //    horizontal differencing for compatible LZW, Deflate and Zstandard input
    //    (as in the other variants). Whether differencing helps is data
    //    dependent, so compare file sizes on your own images. Use
    //    NVTIFF_PREDICTOR_HORIZONTAL to request differencing explicitly.
    add_variant("LZW, predictor disabled", "lzw_no_predictor.tif", 1, NVTIFF_COMPRESSION_LZW,
        [&](nvtiffEncodeParams_t params) {
            CHECK_NVTIFF(nvtiffEncodeParamsSetPredictor(params, NVTIFF_PREDICTOR_NONE));
        });

    // 8. Deflate (Adobe Deflate, tag 8) and Zstandard (tag 50000) through
    //    nvCOMP. These need nvCOMP 6.0 or later at runtime; without it the
    //    variants are skipped. nvCOMP's sweet spot is about 64 KiB of
    //    uncompressed data per strip or tile; larger striles work but compress
    //    and decode more slowly and need more temporary device memory. We reuse
    //    the 256x256 tiles of variant 2 (192 KiB for RGB8) so the sizes are
    //    directly comparable with LZW.
    add_variant("Deflate, 256x256 tiles", "tiled_deflate.tif", 1, NVTIFF_COMPRESSION_ADOBE_DEFLATE,
        [&](nvtiffEncodeParams_t params) { set_geometry(params, NVTIFF_IMAGE_TILED, 256, 256); });
    add_variant("Zstandard, 256x256 tiles", "tiled_zstd.tif", 1, NVTIFF_COMPRESSION_ZSTD,
        [&](nvtiffEncodeParams_t params) { set_geometry(params, NVTIFF_IMAGE_TILED, 256, 256); });

    std::printf("\n%-36s %-28s %12s %8s\n", "variant", "file", "bytes", "ratio");
    for (const Variant& v : results) {
        if (v.bytes == 0) {
            std::printf("%-36s skipped (%s not found)\n", v.label, v.missing_library);
            continue;
        }
        const size_t raw_bytes = (size_t)width * height * 3 * v.pages;
        std::printf("%-36s %-28s %12zu %7.2fx\n", v.label, v.file, v.bytes, (double)raw_bytes / (double)v.bytes);
    }

    CHECK_CUDA(cudaFree(rgb_d));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
